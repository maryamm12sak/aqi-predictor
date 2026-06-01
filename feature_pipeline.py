"""
Feature Pipeline
- Fetches AQI + pollutants from Open-Meteo Air Quality API (free, no key needed)
- Fetches weather from Open-Meteo Weather API (free, no key needed)
- Engineers features
- Stores in MongoDB
- Backfills targets for past 24h/48h/72h records
"""

import os
import requests
import numpy as np
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
CITY        = os.getenv("CITY", "Karachi")
LAT         = 24.8607
LON         = 67.0011


def fetch_aqi_data() -> dict:
    """
    Fetch current AQI and pollutants from Open-Meteo Air Quality API.
    No API key required. Returns US AQI directly.
    """
    resp = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude":  LAT,
            "longitude": LON,
            "current":   ",".join([
                "us_aqi",
                "pm2_5",
                "pm10",
                "ozone",
                "nitrogen_dioxide",
                "sulphur_dioxide",
                "carbon_monoxide",
            ]),
            "timezone": "Asia/Karachi",
        },
        timeout=15
    )
    resp.raise_for_status()
    c = resp.json()["current"]
    return {
        "aqi":  c.get("us_aqi"),
        "pm25": c.get("pm2_5"),
        "pm10": c.get("pm10"),
        "o3":   c.get("ozone"),
        "no2":  c.get("nitrogen_dioxide"),
        "so2":  c.get("sulphur_dioxide"),
        "co":   c.get("carbon_monoxide"),
    }


def fetch_weather_data() -> dict:
    """
    Fetch current weather from Open-Meteo Weather API.
    No API key required.
    """
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude":  LAT,
            "longitude": LON,
            "current":   ",".join([
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "wind_direction_10m",
                "precipitation",
            ]),
            "timezone": "Asia/Karachi",
        },
        timeout=15
    )
    resp.raise_for_status()
    c = resp.json()["current"]
    return {
        "temperature":   c.get("temperature_2m"),
        "humidity":      c.get("relative_humidity_2m"),
        "wind_speed":    c.get("wind_speed_10m"),
        "wind_dir":      c.get("wind_direction_10m"),
        "precipitation": c.get("precipitation"),
    }


def engineer_features(aqi_data: dict, weather_data: dict, prev_aqi: float = None) -> dict:
    """Combine raw data into ML-ready features."""
    now = datetime.now(timezone.utc)
    features = {
        "timestamp":  now,
        "hour":       now.hour,
        "day":        now.day,
        "month":      now.month,
        "weekday":    now.weekday(),
        "is_weekend": int(now.weekday() >= 5),
        **aqi_data,
        **weather_data,
    }

    # AQI change rate
    if prev_aqi is not None and aqi_data["aqi"] is not None:
        features["aqi_change_rate"] = aqi_data["aqi"] - prev_aqi
    else:
        features["aqi_change_rate"] = 0.0

    # Wind components
    ws = features.get("wind_speed") or 0.0
    wd = features.get("wind_dir")   or 0.0
    wd_rad = np.radians(wd)
    features["wind_u"] = ws * np.cos(wd_rad)
    features["wind_v"] = ws * np.sin(wd_rad)

    return features


def get_prev_aqi(collection) -> float:
    """Get the most recent AQI from MongoDB."""
    doc = collection.find_one(sort=[("timestamp", -1)])
    return doc["aqi"] if doc and "aqi" in doc else None


def store_features(features: dict, collection):
    """Upsert features into MongoDB, rounded to the hour."""
    ts = features["timestamp"].replace(minute=0, second=0, microsecond=0)
    features["timestamp"] = ts
    collection.update_one(
        {"timestamp": ts},
        {"$set": features},
        upsert=True
    )
    print(f"✅ Stored features for {ts} | AQI: {features['aqi']}")


def backfill_targets(collection, now_ts, current_aqi):
    """
    Now that we know today's AQI, update past records' targets.
    Record from 24h ago → target_aqi_24h = today's AQI
    Record from 48h ago → target_aqi_48h = today's AQI
    Record from 72h ago → target_aqi_72h = today's AQI
    """
    for hours, key in [
        (24, "target_aqi_24h"),
        (48, "target_aqi_48h"),
        (72, "target_aqi_72h"),
    ]:
        past_ts = now_ts - timedelta(hours=hours)
        result  = collection.update_one(
            {"timestamp": past_ts},
            {"$set": {key: current_aqi}}
        )
        if result.modified_count:
            print(f"🎯 Updated {key} for {past_ts} → {current_aqi}")


def run_feature_pipeline():
    print(f"🚀 Running feature pipeline for {CITY}...")

    client     = MongoClient(MONGODB_URI)
    collection = client["aqi_db"]["features"]

    # Get previous AQI for change rate calculation
    prev_aqi = get_prev_aqi(collection)

    # Fetch data
    print("📡 Fetching air quality from Open-Meteo Air Quality API...")
    aqi_data = fetch_aqi_data()
    print(f"   AQI: {aqi_data['aqi']} | PM2.5: {aqi_data['pm25']} | PM10: {aqi_data['pm10']}")

    print("🌤️  Fetching weather from Open-Meteo Weather API...")
    weather_data = fetch_weather_data()
    print(f"   Temp: {weather_data['temperature']}°C | Humidity: {weather_data['humidity']}% | Wind: {weather_data['wind_speed']} km/h")

    # Engineer and store
    features = engineer_features(aqi_data, weather_data, prev_aqi)
    store_features(features, collection)

    # Backfill targets for past records
    if aqi_data["aqi"] is not None:
        backfill_targets(collection, features["timestamp"], aqi_data["aqi"])

    client.close()
    print("✅ Feature pipeline complete!")
    return features


if __name__ == "__main__":
    run_feature_pipeline()