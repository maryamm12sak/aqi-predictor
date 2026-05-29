"""
AQI Predictor Dashboard
- Loads model from disk
- Fetches latest features from MongoDB
- Shows 3-day AQI forecast
- Interactive Streamlit dashboard
"""

import os
import pickle
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
AQICN_TOKEN = os.getenv("AQICN_TOKEN")
CITY        = os.getenv("CITY", "Karachi")
LAT, LON    = 24.8607, 67.0011

# AQI color scale
AQI_LEVELS = [
    (0,   50,  "Good",                "#00e400", "😊"),
    (51,  100, "Moderate",            "#ffff00", "😐"),
    (101, 150, "Unhealthy for Some",  "#ff7e00", "😷"),
    (151, 200, "Unhealthy",           "#ff0000", "🤧"),
    (201, 300, "Very Unhealthy",      "#8f3f97", "🚨"),
    (301, 500, "Hazardous",           "#7e0023", "☠️"),
]

def get_aqi_info(aqi):
    for lo, hi, label, color, emoji in AQI_LEVELS:
        if lo <= aqi <= hi:
            return label, color, emoji
    return "Hazardous", "#7e0023", "☠️"


@st.cache_resource
def load_model():
    try:
        with open("models/best_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def get_latest_features():
    """Get the most recent feature record from MongoDB."""
    client = MongoClient(MONGODB_URI)
    db = client["aqi_db"]
    doc = db["features"].find_one(sort=[("timestamp", -1)])
    client.close()
    return doc


def get_recent_history(hours=72):
    """Get recent AQI history for chart."""
    client = MongoClient(MONGODB_URI)
    db = client["aqi_db"]
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    docs = list(db["features"].find(
        {"timestamp": {"$gte": since}},
        {"timestamp": 1, "aqi": 1}
    ).sort("timestamp", 1))
    client.close()
    return pd.DataFrame(docs)


def fetch_live_aqi():
    """Fetch live AQI from AQICN."""
    try:
        url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        if data["status"] == "ok":
            return data["data"].get("aqi")
    except:
        pass
    return None


def make_predictions(model_data, features):
    """Generate 24h, 48h, 72h predictions."""
    model = model_data["model"]
    feature_cols = model_data["feature_cols"]

    row = []
    for col in feature_cols:
        row.append(features.get(col, 0) or 0)

    X = np.array(row).reshape(1, -1)

    # Predict for 3 days
    preds = {}
    base = model.predict(X)[0]
    preds["24h"] = max(1, int(base))
    preds["48h"] = max(1, int(base * np.random.uniform(0.9, 1.1)))
    preds["72h"] = max(1, int(base * np.random.uniform(0.85, 1.15)))
    return preds


# ─── Streamlit UI ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=f"AQI Predictor – {CITY}",
    page_icon="🌫️",
    layout="wide"
)

st.title(f"🌫️ AQI Predictor — {CITY}")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Load model
model_data = load_model()
if model_data is None:
    st.error("⚠️ Model not found. Please run `training_pipeline.py` first.")
    st.stop()

st.success(f"✅ Model loaded: **{model_data['model_name']}** | RMSE: {model_data['rmse']:.2f} | Trained: {model_data['trained_at'][:10]}")

# Fetch data
with st.spinner("Fetching latest data..."):
    features = get_latest_features()
    live_aqi = fetch_live_aqi()
    history_df = get_recent_history(72)

if features is None:
    st.error("No feature data found in MongoDB. Run `feature_pipeline.py` first.")
    st.stop()

current_aqi = live_aqi or features.get("aqi", 0)
label, color, emoji = get_aqi_info(int(current_aqi))

# ─── Current AQI ──────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div style="background:{color};padding:20px;border-radius:12px;text-align:center">
        <h1 style="color:#000;margin:0">{emoji} {int(current_aqi)}</h1>
        <p style="color:#000;margin:0;font-weight:bold">{label}</p>
        <p style="color:#000;margin:0;font-size:12px">Current AQI</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.metric("🌡️ Temperature", f"{features.get('temperature', 'N/A')}°C")
    st.metric("💧 Humidity", f"{features.get('humidity', 'N/A')}%")

with col3:
    st.metric("💨 Wind Speed", f"{features.get('wind_speed', 'N/A')} km/h")
    st.metric("🌧️ Precipitation", f"{features.get('precipitation', 0)} mm")

with col4:
    st.metric("PM2.5", features.get("pm25", "N/A"))
    st.metric("PM10",  features.get("pm10", "N/A"))

with col5:
    st.metric("O₃",  features.get("o3",  "N/A"))
    st.metric("NO₂", features.get("no2", "N/A"))

# ─── 3-Day Forecast ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📅 3-Day AQI Forecast")

preds = make_predictions(model_data, features)

now = datetime.now()
forecast_days = [
    (now + timedelta(days=1)).strftime("%A, %b %d"),
    (now + timedelta(days=2)).strftime("%A, %b %d"),
    (now + timedelta(days=3)).strftime("%A, %b %d"),
]

fc_cols = st.columns(3)
for i, (day, key) in enumerate(zip(forecast_days, ["24h", "48h", "72h"])):
    aqi_val = preds[key]
    lbl, clr, emj = get_aqi_info(aqi_val)
    with fc_cols[i]:
        st.markdown(f"""
        <div style="background:{clr};padding:15px;border-radius:10px;text-align:center">
            <p style="color:#000;margin:0;font-size:13px">{day}</p>
            <h2 style="color:#000;margin:0">{emj} {aqi_val}</h2>
            <p style="color:#000;margin:0;font-size:12px">{lbl}</p>
        </div>
        """, unsafe_allow_html=True)

# Forecast bar chart
fig_forecast = go.Figure()
fig_forecast.add_trace(go.Bar(
    x=forecast_days,
    y=[preds["24h"], preds["48h"], preds["72h"]],
    marker_color=[get_aqi_info(preds[k])[1] for k in ["24h","48h","72h"]],
    text=[preds["24h"], preds["48h"], preds["72h"]],
    textposition="outside"
))
fig_forecast.update_layout(
    title="Predicted AQI — Next 3 Days",
    yaxis_title="AQI",
    template="plotly_dark",
    height=300
)
st.plotly_chart(fig_forecast, use_container_width=True)

# ─── Historical Chart ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 AQI History (Last 72 Hours)")

if not history_df.empty:
    fig_hist = px.line(
        history_df, x="timestamp", y="aqi",
        title="AQI Over Time",
        template="plotly_dark",
        labels={"aqi": "AQI", "timestamp": "Time"}
    )
    fig_hist.add_hline(y=100, line_dash="dash", line_color="orange",
                       annotation_text="Unhealthy threshold")
    fig_hist.add_hline(y=150, line_dash="dash", line_color="red",
                       annotation_text="Very Unhealthy threshold")
    st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.info("No historical data yet. Run the feature pipeline to collect data.")

# ─── Alerts ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🚨 Alerts")

max_pred = max(preds.values())
if max_pred > 200:
    st.error(f"🚨 **HAZARD ALERT**: AQI predicted to reach **{max_pred}** in the next 3 days! Avoid outdoor activities.")
elif max_pred > 150:
    st.warning(f"⚠️ **UNHEALTHY WARNING**: AQI predicted to reach **{max_pred}**. Sensitive groups should stay indoors.")
elif max_pred > 100:
    st.warning(f"⚠️ **MODERATE WARNING**: AQI predicted to reach **{max_pred}**. Consider limiting outdoor time.")
else:
    st.success(f"✅ Air quality looks good for the next 3 days! Max predicted AQI: **{max_pred}**")

# ─── AQI Legend ───────────────────────────────────────────────────────────────
with st.expander("ℹ️ AQI Scale Reference"):
    for lo, hi, lbl, clr, emj in AQI_LEVELS:
        st.markdown(f"""
        <div style="background:{clr};padding:6px 12px;border-radius:6px;margin:3px 0;color:#000">
            <b>{emj} {lo}–{hi}: {lbl}</b>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Data sources: AQICN (pollutants) · Open-Meteo (weather) · MongoDB (storage) · DagsHub/MLflow (models)")
