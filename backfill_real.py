"""
Real Backfill using OpenAQ
- Pulls real PM2.5 data from Karachi US Consulate station (Sensor 23747)
- Converts PM2.5 to AQI using US EPA formula
- Combines with Open-Meteo historical weather
- Stores in MongoDB (replaces synthetic data)
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
OPENAQ_KEY  = "670c5f5780e3f5da454f3ffa1ce100e8435c992f3a9b1d10a879da793d6ef881"
SENSOR_ID   = 23747
LAT, LON    = 24.8607, 67.0011


def pm25_to_aqi(pm25):
    if pm25 is None or np.isnan(pm25) or pm25 < 0:
        return None
    breakpoints = [
        (0.0,   12.0,   0,   50),
        (12.1,  35.4,   51,  100),
        (35.5,  55.4,   101, 150),
        (55.5,  150.4,  151, 200),
        (150.5, 250.4,  201, 300),
        (250.5, 350.4,  301, 400),
        (350.5, 500.4,  401, 500),
    ]
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm25 <= c_high:
            return round(((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low)
    return 500


def fetch_openaq_pm25(days_back=2000):
    print(f"📡 Fetching {days_back} days of real PM2.5 data from OpenAQ (chunked)...")
    end_date   = datetime(2025, 3, 4, tzinfo=timezone.utc)
    start_date = end_date - timedelta(days=days_back)
    all_records = []
    chunk_start = start_date

    while chunk_start < end_date:
        chunk_end = min(chunk_start + timedelta(days=180), end_date)
        print(f"   Chunk: {chunk_start.strftime('%Y-%m-%d')} → {chunk_end.strftime('%Y-%m-%d')}")
        page = 1
        while True:
            try:
                r = requests.get(
                    f"https://api.openaq.org/v3/sensors/{SENSOR_ID}/hours",
                    params={
                        "datetime_from": chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "datetime_to":   chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "limit": 1000,
                        "page":  page,
                    },
                    headers={"X-API-Key": OPENAQ_KEY},
                    timeout=60
                )
                if r.status_code != 200:
                    print(f"   API error: {r.status_code}, skipping chunk")
                    break
                results = r.json().get("results", [])
                if not results:
                    break
                all_records.extend(results)
                print(f"     Page {page}: +{len(results)} (total: {len(all_records)})")
                if len(results) < 1000:
                    break
                page += 1
            except Exception as e:
                print(f"   Error: {e}, skipping")
                break
        chunk_start = chunk_end

    print(f"✅ Total PM2.5 records: {len(all_records)}")
    return all_records


def fetch_historical_weather(start_date, end_date):
    print("🌤️  Fetching historical weather from Open-Meteo...")
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=temperature_2m,relative_humidity_2m,"
        "wind_speed_10m,wind_direction_10m,precipitation,surface_pressure,uv_index"
        "&timezone=Asia%2FKarachi"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()["hourly"])
    df["timestamp"] = pd.to_datetime(df["time"], utc=True)
    df.rename(columns={
        "temperature_2m":       "temperature",
        "relative_humidity_2m": "humidity",
        "wind_speed_10m":       "wind_speed",
        "wind_direction_10m":   "wind_dir",
        "surface_pressure":     "pressure",
    }, inplace=True)
    return df.drop(columns=["time"])


def run_real_backfill(days_back=2000):
    print(f"🚀 Starting REAL backfill for last {days_back} days...")

    client = MongoClient(MONGODB_URI)
    col    = client["aqi_db"]["features"]

    old_count = col.count_documents({})
    print(f"\n🗑️  Removing {old_count} old records...")
    col.delete_many({})
    print("   Done!")

    pm25_records = fetch_openaq_pm25(days_back)
    if not pm25_records:
        print("❌ No PM2.5 data found!")
        client.close()
        return

    pm25_data = []
    for rec in pm25_records:
        try:
            ts  = pd.to_datetime(rec["period"]["datetimeFrom"]["utc"], utc=True)
            val = rec.get("value")
            if val is not None and val >= 0:
                pm25_data.append({"timestamp": ts, "pm25_raw": float(val)})
        except:
            continue

    pm25_df = pd.DataFrame(pm25_data).drop_duplicates("timestamp").sort_values("timestamp")
    print(f"\n📊 Valid PM2.5 records: {len(pm25_df)}")
    print(f"   Date range: {pm25_df['timestamp'].min()} → {pm25_df['timestamp'].max()}")

    start_str  = pm25_df["timestamp"].min().strftime("%Y-%m-%d")
    end_str    = pm25_df["timestamp"].max().strftime("%Y-%m-%d")
    weather_df = fetch_historical_weather(start_str, end_str)

    pm25_df["hour_key"]    = pm25_df["timestamp"].dt.floor("h")
    weather_df["hour_key"] = weather_df["timestamp"].dt.floor("h")
    merged = pd.merge(pm25_df, weather_df, on="hour_key", how="left")
    merged["timestamp"] = merged["hour_key"]
    print(f"   Merged records: {len(merged)}")

    # ── Insert using bulk_write ──────────────────────────────────────────────
    print("\n💾 Inserting real data into MongoDB...")
    bulk_insert = []
    prev_aqi    = None
    aqi_list    = []  # store (timestamp, aqi) for target fixing later

    for _, row in merged.iterrows():
        ts      = row["timestamp"]
        pm25    = row["pm25_raw"]
        aqi_val = pm25_to_aqi(pm25)
        if aqi_val is None:
            continue

        aqi_change = (aqi_val - prev_aqi) if prev_aqi is not None else 0.0
        wd_rad = np.radians(row["wind_dir"]) if pd.notna(row.get("wind_dir")) else 0
        ws     = float(row["wind_speed"]) if pd.notna(row.get("wind_speed")) else 0.0

        doc = {
            "timestamp":       ts,
            "hour":            ts.hour,
            "day":             ts.day,
            "month":           ts.month,
            "weekday":         ts.weekday(),
            "is_weekend":      int(ts.weekday() >= 5),
            "aqi":             aqi_val,
            "pm25":            round(pm25, 2),
            "pm10":            None, "o3": None, "no2": None, "so2": None, "co": None,
            "temperature":     row["temperature"]  if pd.notna(row.get("temperature"))  else None,
            "humidity":        row["humidity"]     if pd.notna(row.get("humidity"))     else None,
            "wind_speed":      ws,
            "wind_dir":        row["wind_dir"]     if pd.notna(row.get("wind_dir"))     else None,
            "precipitation":   row["precipitation"] if pd.notna(row.get("precipitation")) else 0,
            "pressure":        row["pressure"]     if pd.notna(row.get("pressure"))     else None,
            "uv_index":        row["uv_index"]     if pd.notna(row.get("uv_index"))     else None,
            "aqi_change_rate": aqi_change,
            "wind_u":          ws * np.cos(wd_rad),
            "wind_v":          ws * np.sin(wd_rad),
            "target_aqi_24h":  aqi_val,
            "target_aqi_48h":  aqi_val,
            "target_aqi_72h":  aqi_val,
        }

        bulk_insert.append(UpdateOne({"timestamp": ts}, {"$set": doc}, upsert=True))
        aqi_list.append((ts, aqi_val))
        prev_aqi = aqi_val

        if len(bulk_insert) >= 1000:
            col.bulk_write(bulk_insert)
            print(f"   Inserted {len(aqi_list)}/{len(merged)}...")
            bulk_insert = []

    if bulk_insert:
        col.bulk_write(bulk_insert)
        print(f"   Inserted {len(aqi_list)}/{len(merged)}...")

    # ── Fix targets using bulk_write ─────────────────────────────────────────
    print("\n🔧 Fixing targets (bulk)...")
    aqi_map = {ts: aqi for ts, aqi in aqi_list}
    ts_set  = set(aqi_map.keys())
    bulk_ops = []

    for ts, _ in aqi_list:
        updates = {}
        for hours, key in [(24, "target_aqi_24h"), (48, "target_aqi_48h"), (72, "target_aqi_72h")]:
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

    print(f"\n✅ Real backfill complete!")
    print(f"   Total records: {col.count_documents({})}")
    lo = col.find_one(sort=[("aqi", 1)])["aqi"]
    hi = col.find_one(sort=[("aqi", -1)])["aqi"]
    print(f"   AQI range: {lo} – {hi}")
    client.close()


if __name__ == "__main__":
    run_real_backfill(days_back=2000)