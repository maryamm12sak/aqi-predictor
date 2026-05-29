"""
Feature Pipeline
- Fetches AQI data from AQICN
- Fetches weather data from Open-Meteo (free, no key needed)
- Engineers features
- Stores in MongoDB
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

AQICN_TOKEN = os.getenv("AQICN_TOKEN")
MONGODB_URI = os.getenv("MONGODB_URI")
CITY = os.getenv("CITY", "Karachi")

# Karachi coordinates
LAT = 24.8607
LON = 67.0011


def fetch_aqi_data(city: str) -> dict:
    """Fetch current AQI and pollutant data from AQICN."""
    url = f"https://api.waqi.info/feed/{city}/?token={AQICN_TOKEN}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data["status"] != "ok":
        raise ValueError(f"AQICN error: {data}")
    d = data["data"]
    iaqi = d.get("iaqi", {})
    return {
        "aqi": d.get("aqi"),
        "pm25": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "o3":   iaqi.get("o3",   {}).get("v"),
        "no2":  iaqi.get("no2",  {}).get("v"),
        "so2":  iaqi.get("so2",  {}).get("v"),
        "co":   iaqi.get("co",   {}).get("v"),
    }


def fetch_weather_data(lat: float, lon: float) -> dict:
    """Fetch current weather from Open-Meteo (free, no API key)."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,"
        "wind_speed_10m,wind_direction_10m,precipitation"
        "&timezone=Asia%2FKarachi"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    c = resp.json()["current"]
    return {
        "temperature":  c.get("temperature_2m"),
        "humidity":     c.get("relative_humidity_2m"),
        "wind_speed":   c.get("wind_speed_10m"),
        "wind_dir":     c.get("wind_direction_10m"),
        "precipitation":c.get("precipitation"),
    }


def engineer_features(aqi_data: dict, weather_data: dict, prev_aqi: float = None) -> dict:
    """Combine raw data into ML-ready features."""
    now = datetime.now(timezone.utc)
    features = {
        "timestamp": now,
        "hour":      now.hour,
        "day":       now.day,
        "month":     now.month,
        "weekday":   now.weekday(),
        "is_weekend": int(now.weekday() >= 5),
        **aqi_data,
        **weather_data,
    }
    # Derived features
    if prev_aqi is not None and aqi_data["aqi"] is not None:
        features["aqi_change_rate"] = aqi_data["aqi"] - prev_aqi
    else:
        features["aqi_change_rate"] = 0.0

    # Wind components
    if features["wind_speed"] and features["wind_dir"]:
        wd_rad = np.radians(features["wind_dir"])
        features["wind_u"] = features["wind_speed"] * np.cos(wd_rad)
        features["wind_v"] = features["wind_speed"] * np.sin(wd_rad)
    else:
        features["wind_u"] = 0.0
        features["wind_v"] = 0.0

    return features


def get_prev_aqi(collection) -> float:
    """Get the most recent AQI from MongoDB."""
    doc = collection.find_one(sort=[("timestamp", -1)])
    return doc["aqi"] if doc and "aqi" in doc else None


def store_features(features: dict, collection):
    """Upsert features into MongoDB."""
    # Round timestamp to hour to avoid duplicates
    ts = features["timestamp"].replace(minute=0, second=0, microsecond=0)
    features["timestamp"] = ts
    collection.update_one(
        {"timestamp": ts},
        {"$set": features},
        upsert=True
    )
    print(f"✅ Stored features for {ts} | AQI: {features['aqi']}")


def run_feature_pipeline():
    print(f"🚀 Running feature pipeline for {CITY}...")

    # Connect to MongoDB
    client = MongoClient(MONGODB_URI)
    db = client["aqi_db"]
    collection = db["features"]

    # Get previous AQI for change rate
    prev_aqi = get_prev_aqi(collection)

    # Fetch data
    print("📡 Fetching AQI data...")
    aqi_data = fetch_aqi_data(CITY)

    print("🌤️  Fetching weather data...")
    weather_data = fetch_weather_data(LAT, LON)

    # Engineer features
    features = engineer_features(aqi_data, weather_data, prev_aqi)

    # Store in MongoDB
    store_features(features, collection)

    client.close()
    print("✅ Feature pipeline complete!")
    return features


if __name__ == "__main__":
    run_feature_pipeline()
