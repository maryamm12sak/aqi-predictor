"""
Backfill using Open-Meteo Air Quality + Weather APIs
- Pulls historical US AQI, PM2.5, PM10, O3, NO2, SO2, CO from Open-Meteo
- Pulls historical weather from Open-Meteo Archive
- Stores in MongoDB (replaces all previous data)
- Same data source as live feature pipeline = fully consistent
"""

import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
LAT         = 24.8607
LON         = 67.0011
START_DATE  = "2022-01-01"
END_DATE    = datetime.now().strftime("%Y-%m-%d")


def fetch_air_quality(start_date, end_date):
    """Fetch historical hourly AQI + pollutants from Open-Meteo Air Quality API."""
    print(f"📡 Fetching air quality: {start_date} → {end_date}")
    resp = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude":   LAT,
            "longitude":  LON,
            "hourly":     ",".join([
                "us_aqi", "pm2_5", "pm10",
                "ozone", "nitrogen_dioxide",
                "sulphur_dioxide", "carbon_monoxide"
            ]),
            "start_date": start_date,
            "end_date":   end_date,
            "timezone":   "Asia/Karachi",
        },
        timeout=60
    )
    resp.raise_for_status()
    data = resp.json()["hourly"]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["time"], utc=True)
    df.rename(columns={
        "us_aqi":           "aqi",
        "pm2_5":            "pm25",
        "ozone":            "o3",
        "nitrogen_dioxide": "no2",
        "sulphur_dioxide":  "so2",
        "carbon_monoxide":  "co",
    }, inplace=True)
    df.drop(columns=["time"], inplace=True)
    print(f"   Got {len(df)} air quality records")
    return df


def fetch_weather(start_date, end_date):
    """Fetch historical hourly weather from Open-Meteo Archive API."""
    # Archive API only supports up to yesterday
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    end_date  = min(end_date, yesterday)
    print(f"🌤️  Fetching weather: {start_date} → {end_date}")
    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude":   LAT,
            "longitude":  LON,
            "hourly":     ",".join([
                "temperature_2m", "relative_humidity_2m",
                "wind_speed_10m", "wind_direction_10m",
                "precipitation", "surface_pressure", "uv_index"
            ]),
            "start_date": start_date,
            "end_date":   end_date,
            "timezone":   "Asia/Karachi",
        },
        timeout=60
    )
    resp.raise_for_status()
    data = resp.json()["hourly"]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["time"], utc=True)
    df.rename(columns={
        "temperature_2m":       "temperature",
        "relative_humidity_2m": "humidity",
        "wind_speed_10m":       "wind_speed",
        "wind_direction_10m":   "wind_dir",
        "surface_pressure":     "pressure",
    }, inplace=True)
    df.drop(columns=["time"], inplace=True)
    print(f"   Got {len(df)} weather records")
    return df


def run_backfill():
    print("🚀 Starting Open-Meteo backfill...")
    print(f"   Date range: {START_DATE} → {END_DATE}")

    # Connect to MongoDB
    client = MongoClient(MONGODB_URI)
    col    = client["aqi_db"]["features"]

    # Clear old data
    old = col.count_documents({})
    print(f"\n🗑️  Removing {old} old records...")
    col.delete_many({})
    print("   Done!")

    # Fetch data
    aq_df = fetch_air_quality(START_DATE, END_DATE)
    wx_df = fetch_weather(START_DATE, END_DATE)

    # Merge on timestamp
    aq_df["hour_key"] = aq_df["timestamp"].dt.floor("h")
    wx_df["hour_key"] = wx_df["timestamp"].dt.floor("h")
    merged = pd.merge(aq_df, wx_df, on="hour_key", how="left")
    merged["timestamp"] = merged["hour_key"]
    merged = merged.drop(columns=["hour_key", "timestamp_x", "timestamp_y"], errors="ignore")
    merged = merged.dropna(subset=["aqi"])
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    print(f"\n📊 Merged records: {len(merged)}")

    # Build docs and insert
    print("💾 Inserting into MongoDB...")
    bulk_ops = []
    aqi_list = []
    prev_aqi = None

    for _, row in merged.iterrows():
        ts      = row["timestamp"]
        aqi_val = int(row["aqi"]) if pd.notna(row["aqi"]) else None
        if aqi_val is None:
            continue

        aqi_change = (aqi_val - prev_aqi) if prev_aqi is not None else 0.0
        ws     = float(row["wind_speed"]) if pd.notna(row.get("wind_speed")) else 0.0
        wd     = float(row["wind_dir"])   if pd.notna(row.get("wind_dir"))   else 0.0
        wd_rad = np.radians(wd)

        doc = {
            "timestamp":       ts,
            "hour":            ts.hour,
            "day":             ts.day,
            "month":           ts.month,
            "weekday":         ts.weekday(),
            "is_weekend":      int(ts.weekday() >= 5),
            "aqi":             aqi_val,
            "pm25":            round(float(row["pm25"]), 2) if pd.notna(row.get("pm25")) else None,
            "pm10":            round(float(row["pm10"]), 2) if pd.notna(row.get("pm10")) else None,
            "o3":              round(float(row["o3"]),   2) if pd.notna(row.get("o3"))   else None,
            "no2":             round(float(row["no2"]),  2) if pd.notna(row.get("no2"))  else None,
            "so2":             round(float(row["so2"]),  2) if pd.notna(row.get("so2"))  else None,
            "co":              round(float(row["co"]),   2) if pd.notna(row.get("co"))   else None,
            "temperature":     float(row["temperature"]) if pd.notna(row.get("temperature")) else None,
            "humidity":        float(row["humidity"])    if pd.notna(row.get("humidity"))    else None,
            "wind_speed":      ws,
            "wind_dir":        wd,
            "precipitation":   float(row["precipitation"]) if pd.notna(row.get("precipitation")) else 0.0,
            "pressure":        float(row["pressure"])    if pd.notna(row.get("pressure"))    else None,
            "uv_index":        float(row["uv_index"])    if pd.notna(row.get("uv_index"))    else None,
            "aqi_change_rate": aqi_change,
            "wind_u":          ws * np.cos(wd_rad),
            "wind_v":          ws * np.sin(wd_rad),
            "target_aqi_24h":  aqi_val,
            "target_aqi_48h":  aqi_val,
            "target_aqi_72h":  aqi_val,
        }

        bulk_ops.append(UpdateOne({"timestamp": ts}, {"$set": doc}, upsert=True))
        aqi_list.append((ts, aqi_val))
        prev_aqi = aqi_val

        if len(bulk_ops) >= 1000:
            col.bulk_write(bulk_ops)
            print(f"   Inserted {len(aqi_list)}/{len(merged)}...")
            bulk_ops = []

    if bulk_ops:
        col.bulk_write(bulk_ops)
        print(f"   Inserted {len(aqi_list)}/{len(merged)}...")

    # Fix targets using bulk operations
    print("\n🔧 Fixing targets (bulk)...")
    aqi_map  = {ts: aqi for ts, aqi in aqi_list}
    ts_set   = set(aqi_map.keys())
    bulk_ops = []

    for ts, _ in aqi_list:
        updates = {}
        for hours, key in [(24,"target_aqi_24h"),(48,"target_aqi_48h"),(72,"target_aqi_72h")]:
            future_ts = ts + timedelta(hours=hours)
            if future_ts in ts_set:
                updates[key] = aqi_map[future_ts]
        if updates:
            bulk_ops.append(UpdateOne({"timestamp": ts}, {"$set": updates}))
        if len(bulk_ops) >= 1000:
            col.bulk_write(bulk_ops)
            bulk_ops = []

    if bulk_ops:
        col.bulk_write(bulk_ops)

    total = col.count_documents({})
    lo    = col.find_one(sort=[("aqi", 1)])["aqi"]
    hi    = col.find_one(sort=[("aqi",-1)])["aqi"]
    print(f"\n✅ Backfill complete!")
    print(f"   Total records: {total}")
    print(f"   AQI range: {lo} – {hi}")
    print(f"   Date range: {START_DATE} → {END_DATE}")
    client.close()


if __name__ == "__main__":
    run_backfill()