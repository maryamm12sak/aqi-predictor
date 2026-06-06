import os, pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
CITY        = os.getenv("CITY", "Karachi")

def get_aqi_info(aqi):
    aqi = int(aqi)
    if aqi <= 50:   return "Good",               "#10b981", "linear-gradient(135deg, #064e3b, #10b981)"
    if aqi <= 100:  return "Moderate",           "#f59e0b", "linear-gradient(135deg, #451a03, #f59e0b)"
    if aqi <= 150:  return "Unhealthy for Some", "#f97316", "linear-gradient(135deg, #431407, #f97316)"
    if aqi <= 200:  return "Unhealthy",          "#ef4444", "linear-gradient(135deg, #450a0a, #ef4444)"
    if aqi <= 300:  return "Very Unhealthy",     "#8b5cf6", "linear-gradient(135deg, #1e1b4b, #8b5cf6)"
    return "Hazardous",                          "#d946ef", "linear-gradient(135deg, #3b0764, #d946ef)"

@st.cache_resource
def load_models():
    import pathlib
    base = pathlib.Path(__file__).parent
    models = {}
    for horizon in ["24h", "48h", "72h"]:
        p = base / "models" / f"best_model_{horizon}.pkl"
        try:
            with open(p, "rb") as f:
                models[horizon] = pickle.load(f)
        except:
            # fallback to best_model.pkl for all horizons
            p2 = base / "models" / "best_model.pkl"
            try:
                with open(p2, "rb") as f:
                    models[horizon] = pickle.load(f)
            except:
                models[horizon] = None
    return models

@st.cache_data(ttl=300)
def get_latest_features():
    c = MongoClient(MONGODB_URI)
    doc = c["aqi_db"]["features"].find_one(sort=[("timestamp", -1)])
    c.close()
    return doc

@st.cache_data(ttl=300)
def get_history(hours=72):
    c = MongoClient(MONGODB_URI)
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    docs = list(c["aqi_db"]["features"].find(
        {"timestamp": {"$gte": since}},
        {"timestamp": 1, "aqi": 1}
    ).sort("timestamp", 1))
    c.close()
    return pd.DataFrame(docs)

def predict(models, features):
    preds = {}
    for horizon in ["24h", "48h", "72h"]:
        md = models.get(horizon)
        if md:
            fc = md["feature_cols"]
            X  = np.array([features.get(c, 0) or 0 for c in fc]).reshape(1, -1)
            preds[horizon] = max(1, int(float(md["model"].predict(X)[0])))
        else:
            preds[horizon] = 0
    return preds

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"AirWatch · {CITY}",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700&family=Space+Grotesk:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(circle at 10% 20%, #0a0f1a, #03060c); }
[data-testid="stAppViewContainer"] { background: transparent; }
[data-testid="stHeader"] { background: rgba(0,0,0,0.2); backdrop-filter: blur(12px); }
[data-testid="stSidebar"] { background: rgba(10,15,26,0.75) !important; backdrop-filter: blur(16px); border-right: 1px solid rgba(255,255,255,0.05); }
[data-testid="stSidebar"] * { color: #e2e8f0; }
.card { background: rgba(18,22,32,0.6); backdrop-filter: blur(12px); border-radius: 28px; border: 1px solid rgba(255,255,255,0.05); padding: 1.5rem; transition: all 0.25s ease; box-shadow: 0 8px 20px rgba(0,0,0,0.2); }
.card:hover { border-color: rgba(6,182,212,0.3); box-shadow: 0 12px 28px rgba(0,0,0,0.3); }
h1, h2, h3, .section-title { font-family: 'Space Grotesk', sans-serif; font-weight: 500; letter-spacing: -0.01em; background: linear-gradient(135deg, #f0f9ff, #94a3b8); background-clip: text; -webkit-background-clip: text; color: transparent; margin-bottom: 1rem; }
.big-number { font-family: 'JetBrains Mono', monospace; font-size: 3.8rem; font-weight: 500; line-height: 1; background: linear-gradient(135deg, #ffffff, #a5f3fc); background-clip: text; -webkit-background-clip: text; color: transparent; }
.forecast-card { background: rgba(15,20,30,0.5); backdrop-filter: blur(8px); border-radius: 24px; padding: 1.2rem; text-align: center; border: 1px solid rgba(6,182,212,0.2); transition: 0.2s; }
.forecast-card:hover { border-color: #06b6d4; transform: translateY(-3px); }
.forecast-day { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.06em; }
.forecast-value { font-family: 'JetBrains Mono', monospace; font-size: 2.8rem; font-weight: 500; margin: 0.5rem 0; }
.pollutant-tile { background: rgba(0,0,0,0.2); border-radius: 20px; padding: 0.9rem; text-align: center; transition: 0.2s; border: 1px solid rgba(255,255,255,0.03); }
.pollutant-tile:hover { background: rgba(6,182,212,0.1); border-color: rgba(6,182,212,0.3); }
.model-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.model-table th { text-align: left; padding: 0.7rem 0.5rem; color: #94a3b8; font-weight: 500; border-bottom: 1px solid rgba(255,255,255,0.05); }
.model-table td { padding: 0.6rem 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.02); }
.model-table tr.best td { background: linear-gradient(90deg, rgba(6,182,212,0.15), transparent); font-weight: 500; }
.best-badge { background: #06b6d4; color: #0a0f1a; font-size: 0.65rem; font-weight: 700; padding: 0.2rem 0.6rem; border-radius: 40px; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
models = load_models()
md = models.get("24h")
if not md:
    st.error("❌ Model not found. Run `training_pipeline.py` first.")
    st.stop()

feat = get_latest_features()
if not feat:
    st.error("❌ No feature data. Run `feature_pipeline.py` first.")
    st.stop()

# Use AQI from MongoDB (fetched by Open-Meteo in feature pipeline)
aqi            = int(feat.get("aqi") or 0)
label, color, gradient = get_aqi_info(aqi)
preds          = predict(models, feat)
now            = datetime.now()
hist           = get_history(72)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; margin-bottom:1.5rem;">
        <div class="big-number" style="font-size:4.5rem;color:{color};">{aqi}</div>
        <div style="background:{gradient}; padding:0.2rem 1rem; border-radius:60px; display:inline-block; margin-top:0.3rem;">
            <span style="color:white; font-weight:600; font-size:0.8rem;">{label}</span>
        </div>
        <div style="height:4px; background:rgba(255,255,255,0.1); border-radius:4px; margin:1rem 0; overflow:hidden;">
            <div style="width:{min(aqi/500*100,100)}%; height:100%; background:{color}; border-radius:4px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🌦️ Weather now")
    for name, key in [("Temp", "temperature"), ("Humidity", "humidity"),
                       ("Wind", "wind_speed"), ("Rain", "precipitation")]:
        val = feat.get(key, "—")
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;'>"
            f"<span style='color:#94a3b8'>{name}</span>"
            f"<span style='font-family:monospace'>{val}</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("""
    <div style="margin-top:1.25rem;">
      <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;color:#64748b;margin-bottom:0.6rem;">AQI Scale</div>
      <div style="display:flex;flex-direction:column;gap:5px;">
        <div style="display:flex;align-items:center;gap:8px;"><div style="width:10px;height:10px;border-radius:2px;background:#10b981;flex-shrink:0;"></div><span style="font-size:0.72rem;color:#94a3b8;">0–50 &nbsp;&nbsp; Good</span></div>
        <div style="display:flex;align-items:center;gap:8px;"><div style="width:10px;height:10px;border-radius:2px;background:#f59e0b;flex-shrink:0;"></div><span style="font-size:0.72rem;color:#94a3b8;">51–100 &nbsp; Moderate</span></div>
        <div style="display:flex;align-items:center;gap:8px;"><div style="width:10px;height:10px;border-radius:2px;background:#f97316;flex-shrink:0;"></div><span style="font-size:0.72rem;color:#94a3b8;">101–150 · Unhealthy for Some</span></div>
        <div style="display:flex;align-items:center;gap:8px;"><div style="width:10px;height:10px;border-radius:2px;background:#ef4444;flex-shrink:0;"></div><span style="font-size:0.72rem;color:#94a3b8;">151–200 · Unhealthy</span></div>
        <div style="display:flex;align-items:center;gap:8px;"><div style="width:10px;height:10px;border-radius:2px;background:#8b5cf6;flex-shrink:0;"></div><span style="font-size:0.72rem;color:#94a3b8;">201–300 · Very Unhealthy</span></div>
        <div style="display:flex;align-items:center;gap:8px;"><div style="width:10px;height:10px;border-radius:2px;background:#d946ef;flex-shrink:0;"></div><span style="font-size:0.72rem;color:#94a3b8;">301+ &nbsp;&nbsp;&nbsp; Hazardous</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"**Model**  \n`{md['model_name']}`  \n**RMSE**  `{md['rmse']:.2f}`")
    st.caption(f"Last update: {now.strftime('%d %b %H:%M')}  \nGitHub Actions · hourly")

# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:1.5rem;">
    <h1 style="margin:0;">🌬️ AirWatch</h1>
    <span style="color:#64748b;font-size:0.8rem;">{now.strftime('%A, %d %b %Y · %H:%M')}</span>
</div>
""", unsafe_allow_html=True)

# Refresh button
if st.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()

# Alert
max_pred = max(preds.values())
if max_pred > 200:
    st.markdown(f'<div style="background:rgba(239,68,68,0.1);border-left:4px solid #ef4444;border-radius:16px;padding:1rem;margin-bottom:1rem;">🚨 <strong>Hazard alert</strong> – AQI forecast reaches {max_pred}. Avoid outdoor activity.</div>', unsafe_allow_html=True)
elif max_pred > 100:
    st.markdown(f'<div style="background:rgba(245,158,11,0.1);border-left:4px solid #f59e0b;border-radius:16px;padding:1rem;margin-bottom:1rem;">⚠️ Unhealthy air expected – limit prolonged exposure (peak {max_pred}).</div>', unsafe_allow_html=True)

# ── Forecast ──────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">🔮 3‑day outlook</p>', unsafe_allow_html=True)
cols     = st.columns(3)
day_names = [(now + timedelta(days=i+1)).strftime("%A") for i in range(3)]
for col, key, day in zip(cols, ["24h","48h","72h"], day_names):
    val = preds[key]
    lbl, colr, _ = get_aqi_info(val)
    col.markdown(f"""
    <div class="forecast-card">
        <div class="forecast-day">{day}</div>
        <div class="forecast-value" style="color:{colr}">{val}</div>
        <div style="font-size:0.7rem;font-weight:500;">{lbl}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Pollutants ────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title" style="margin-top:1.2rem;">🧪 Key pollutants</p>', unsafe_allow_html=True)
pcols = st.columns(6)
for col, (name, key, unit) in zip(pcols, [
    ("PM₂.₅","pm25","µg/m³"), ("PM₁₀","pm10","µg/m³"),
    ("O₃","o3","ppb"),         ("NO₂","no2","ppb"),
    ("SO₂","so2","ppb"),       ("CO","co","ppm")
]):
    val = feat.get(key)
    col.markdown(f"""
    <div class="pollutant-tile">
        <div style="font-size:0.7rem;color:#94a3b8;">{name}</div>
        <div style="font-size:1.6rem;font-weight:500;font-family:'JetBrains Mono';">{round(val,1) if val is not None else "—"}</div>
        <div style="font-size:0.65rem;">{unit}</div>
    </div>
    """, unsafe_allow_html=True)

# ── History chart ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">📈 Last 72 hours · AQI trend</p>', unsafe_allow_html=True)
if not hist.empty:
    hist["timestamp"] = pd.to_datetime(hist["timestamp"])
    hist = hist.sort_values("timestamp")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["timestamp"], y=hist["aqi"],
        fill='tozeroy',
        line=dict(color="#06b6d4", width=2),
        marker=dict(size=0),
        hoverinfo="x+y"
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=20,b=20), height=260,
        xaxis=dict(showgrid=False, showline=False, tickformat="%d %b\n%H:%M", color="#64748b"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   title=dict(text="AQI", font=dict(color="#94a3b8"))),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Additional weather + Model leaderboard ────────────────────────────────────
left, right = st.columns([1, 1.2])

with left:
    st.markdown('<p class="section-title">🌡️ Additional weather</p>', unsafe_allow_html=True)
    for name, val in [
        ("Wind direction", f"{feat.get('wind_dir', '—')}°"),
        ("AQI change/hr",  f"{round(feat.get('aqi_change_rate', 0), 1)}"),
        ("Wind U",         f"{round(feat.get('wind_u', 0), 1)}"),
        ("Wind V",         f"{round(feat.get('wind_v', 0), 1)}"),
    ]:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;padding:0.3rem 0;'>"
            f"<span>{name}</span><span style='font-family:monospace'>{val}</span></div>",
            unsafe_allow_html=True
        )

with right:
    st.markdown('<p class="section-title">📊 Best models per horizon</p>', unsafe_allow_html=True)
    rows = ""
    for horizon in ["24h", "48h", "72h"]:
        md_h = models.get(horizon)
        if md_h:
            rows += (
                f'<tr class="best"><td>{horizon}</td>'
                f"<td>{md_h['model_name']}</td>"
                f"<td>{md_h['rmse']:.2f}</td></tr>"
            )
    st.markdown(
        '<table class="model-table"><thead><tr>'
        '<th>Horizon</th><th>Best Model</th><th>RMSE</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>',
        unsafe_allow_html=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:2rem;padding:1rem;border-top:1px solid rgba(255,255,255,0.03);">
    <span style="color:#475569;font-size:0.7rem;">
        Data: Open-Meteo Air Quality &amp; Weather · MongoDB Atlas · DagsHub/MLflow · GitHub Actions
    </span>
</div>
""", unsafe_allow_html=True)
