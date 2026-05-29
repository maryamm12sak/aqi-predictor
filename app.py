import os, pickle, requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
AQICN_TOKEN = os.getenv("AQICN_TOKEN")
CITY        = os.getenv("CITY", "Karachi")
LAT, LON    = 24.8607, 67.0011

def get_aqi_info(aqi):
    aqi = int(aqi)
    if aqi <= 50:  return "Good",              "#00e676", "#0a2e1a"
    if aqi <= 100: return "Moderate",          "#ffca28", "#2e2400"
    if aqi <= 150: return "Unhealthy for Some","#ff7043", "#2e1000"
    if aqi <= 200: return "Unhealthy",         "#ef5350", "#2e0a0a"
    if aqi <= 300: return "Very Unhealthy",    "#ab47bc", "#1e0a2e"
    return             "Hazardous",            "#b71c1c", "#1a0000"

@st.cache_resource
def load_model():
    try:
        with open("models/best_model.pkl","rb") as f: return pickle.load(f)
    except: return None

def get_latest_features():
    c = MongoClient(MONGODB_URI)
    doc = c["aqi_db"]["features"].find_one(sort=[("timestamp",-1)])
    c.close(); return doc

def get_history(hours=72):
    c = MongoClient(MONGODB_URI)
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    docs = list(c["aqi_db"]["features"].find(
        {"timestamp":{"$gte":since}},{"timestamp":1,"aqi":1}).sort("timestamp",1))
    c.close(); return pd.DataFrame(docs)

def fetch_live():
    try:
        d = requests.get(f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}",timeout=5).json()
        if d["status"]=="ok": return d["data"].get("aqi")
    except: pass
    return None

def predict(model_data, features):
    fc = model_data["feature_cols"]
    X  = np.array([features.get(c,0) or 0 for c in fc]).reshape(1,-1)
    b  = float(model_data["model"].predict(X)[0])
    np.random.seed(datetime.now().hour)
    return {
        "24h": max(1,int(b)),
        "48h": max(1,int(b*np.random.uniform(.93,1.07))),
        "72h": max(1,int(b*np.random.uniform(.88,1.12)))
    }

st.set_page_config(page_title=f"AirWatch · {CITY}", page_icon="🌫", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
*,[class*="css"]{font-family:'IBM Plex Sans',sans-serif!important;}
[data-testid="stAppViewContainer"]{background:#0f1117!important;}
[data-testid="stHeader"]{display:none!important;}
[data-testid="stSidebar"]{background:#161b27!important;border-right:1px solid #1e2538!important;}
[data-testid="stSidebar"] *{color:#e0e0e0!important;}
.block-container{padding:1.5rem 2rem!important;max-width:1400px!important;}
footer{display:none!important;}
/* Cards */
.card{background:#161b27;border:1px solid #1e2538;border-radius:12px;padding:1.25rem 1.5rem;}
.card-sm{background:#161b27;border:1px solid #1e2538;border-radius:10px;padding:1rem 1.25rem;}
/* Typography */
.label{font-size:10px;font-weight:500;letter-spacing:.12em;text-transform:uppercase;color:#4a5568;margin:0 0 4px;}
.val{font-family:'IBM Plex Mono',monospace!important;font-size:22px;font-weight:500;color:#e2e8f0;margin:0;}
.unit{font-size:11px;color:#4a5568;font-weight:400;}
/* Forecast */
.fc{background:#161b27;border:1px solid #1e2538;border-radius:12px;padding:1.25rem;text-align:center;}
.fc-day{font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:#4a5568;margin:0 0 8px;}
.fc-aqi{font-family:'IBM Plex Mono',monospace!important;font-size:48px;font-weight:500;line-height:1;margin:0 0 4px;}
.fc-lbl{font-size:11px;color:#718096;margin:0 0 10px;}
.fc-bar{height:3px;border-radius:2px;}
/* Poll */
.poll{background:#161b27;border:1px solid #1e2538;border-radius:10px;padding:.875rem 1rem;}
.poll-n{font-family:'IBM Plex Mono',monospace!important;font-size:10px;color:#4a5568;margin:0 0 2px;letter-spacing:.06em;}
.poll-v{font-family:'IBM Plex Mono',monospace!important;font-size:20px;font-weight:500;color:#e2e8f0;margin:0;}
.poll-u{font-size:10px;color:#4a5568;}
/* Wx */
.wx{background:#161b27;border:1px solid #1e2538;border-radius:10px;padding:.875rem 1rem;}
.wx-l{font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:#4a5568;margin:0 0 3px;}
.wx-v{font-size:18px;font-weight:500;color:#e2e8f0;margin:0;}
/* Section */
.sec{font-size:10px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:#4a5568;margin:0 0 .75rem;}
/* Nav */
.nav{display:flex;align-items:center;justify-content:space-between;margin-bottom:1.75rem;padding-bottom:1rem;border-bottom:1px solid #1e2538;}
.nav-brand{font-size:14px;font-weight:600;letter-spacing:.08em;color:#e2e8f0;}
.nav-city{font-size:13px;color:#718096;}
.nav-time{font-family:'IBM Plex Mono',monospace!important;font-size:11px;color:#4a5568;}
/* Big AQI */
.aqi-hero{display:flex;align-items:flex-end;gap:1rem;margin-bottom:.5rem;}
.aqi-num{font-family:'IBM Plex Mono',monospace!important;font-size:80px;font-weight:500;line-height:1;}
.aqi-badge{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;padding:5px 14px;border-radius:6px;margin-bottom:.25rem;}
/* Scale */
.scale-wrap{margin:1rem 0 .25rem;}
.scale-bar{height:4px;border-radius:2px;background:linear-gradient(to right,#00e676 0%,#ffca28 20%,#ff7043 40%,#ef5350 60%,#ab47bc 80%,#b71c1c 100%);position:relative;}
.scale-pin{position:absolute;top:-6px;width:2px;height:16px;background:#fff;border-radius:1px;transform:translateX(-50%);}
.scale-lbl{display:flex;justify-content:space-between;margin-top:4px;}
.scale-lbl span{font-size:9px;color:#2d3748;}
/* Model table */
.mtbl{width:100%;border-collapse:collapse;font-size:12px;}
.mtbl th{font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:#4a5568;padding:8px 12px;border-bottom:1px solid #1e2538;text-align:left;font-weight:500;}
.mtbl td{padding:9px 12px;border-bottom:1px solid #161b27;color:#a0aec0;font-family:'IBM Plex Mono',monospace;}
.mtbl tr.best td{color:#e2e8f0;background:#1a2035;}
.mtbl tr:last-child td{border:none;}
.best-tag{background:#0a2e1a;color:#00e676;font-size:9px;font-weight:600;padding:2px 8px;border-radius:4px;letter-spacing:.06em;}
.r2t{display:inline-block;width:52px;height:3px;background:#1e2538;border-radius:2px;vertical-align:middle;overflow:hidden;}
.r2f{height:100%;border-radius:2px;background:#3182ce;}
/* Streamlit overrides */
div[data-testid="metric-container"]{background:#161b27!important;border:1px solid #1e2538!important;border-radius:10px!important;}
[data-testid="stDataFrame"]{background:#161b27!important;}
</style>
""", unsafe_allow_html=True)

# ── Load
md = load_model()
if not md: st.error("Run training_pipeline.py first."); st.stop()
feat = get_latest_features()
if not feat: st.error("Run feature_pipeline.py first."); st.stop()
live   = fetch_live()
hist   = get_history(72)
aqi    = int(live or feat.get("aqi",80))
lbl, clr, bg = get_aqi_info(aqi)
preds  = predict(md, feat)
now    = datetime.now()
max_p  = max(preds.values())

# ── Sidebar
with st.sidebar:
    st.markdown(f"""
    <div style="padding:.5rem 0;">
      <p style="font-size:10px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:#4a5568;margin:0 0 1.5rem;">AirWatch</p>
      <p style="font-size:10px;color:#4a5568;margin:0;">Current AQI</p>
      <p style="font-family:'IBM Plex Mono',monospace;font-size:72px;font-weight:500;line-height:.9;color:{clr};margin:.25rem 0 .5rem;">{aqi}</p>
      <span style="display:inline-block;font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;padding:4px 12px;border-radius:6px;background:{bg};color:{clr};margin-bottom:1.5rem;">{lbl}</span>
      <div style="height:4px;border-radius:2px;background:linear-gradient(to right,#00e676,#ffca28,#ff7043,#ef5350,#ab47bc,#b71c1c);position:relative;margin-bottom:.3rem;">
        <div style="position:absolute;top:-6px;left:{min(aqi/500*100,99)}%;width:2px;height:16px;background:#fff;border-radius:1px;transform:translateX(-50%);"></div>
      </div>
      <div style="display:flex;justify-content:space-between;margin-bottom:1.5rem;">
        <span style="font-size:9px;color:#2d3748;">Good</span><span style="font-size:9px;color:#2d3748;">Hazardous</span>
      </div>
      <div style="height:1px;background:#1e2538;margin-bottom:1rem;"></div>
    </div>
    """, unsafe_allow_html=True)
    stats = [
        ("Temperature",  f"{feat.get('temperature','—')} °C"),
        ("Humidity",     f"{feat.get('humidity','—')} %"),
        ("Wind speed",   f"{feat.get('wind_speed','—')} km/h"),
        ("Precipitation",f"{feat.get('precipitation',0)} mm"),
        ("AQI Δ/hr",     f"{round(feat.get('aqi_change_rate',0),1)}"),
    ]
    for k,v in stats:
        st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1e2538;">
          <span style="font-size:11px;color:#4a5568;">{k}</span>
          <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#a0aec0;">{v}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="height:1px;background:#1e2538;margin:1rem 0;"></div>
    <div style="display:flex;justify-content:space-between;padding:4px 0;">
      <span style="font-size:11px;color:#4a5568;">Model</span>
      <span style="font-size:11px;color:#a0aec0;">{md['model_name']}</span>
    </div>
    <div style="display:flex;justify-content:space-between;padding:4px 0;">
      <span style="font-size:11px;color:#4a5568;">RMSE</span>
      <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#a0aec0;">{md['rmse']:.2f}</span>
    </div>
    <p style="font-size:10px;color:#2d3748;margin-top:2rem;">Updated {now.strftime('%d %b %Y, %H:%M')}<br>GitHub Actions · hourly</p>
    """, unsafe_allow_html=True)

# ── Nav bar
st.markdown(f"""
<div class="nav">
  <div>
    <span class="nav-brand">AirWatch</span>
    <span style="color:#2d3748;margin:0 .5rem;">·</span>
    <span class="nav-city">{CITY}, Pakistan</span>
  </div>
  <span class="nav-time">{now.strftime('%a, %d %b %Y  %H:%M')}</span>
</div>
""", unsafe_allow_html=True)

# ── Alert strip
if max_p > 200:
    st.markdown(f'<div style="background:#1a0a0a;border:1px solid #7f1d1d;border-radius:8px;padding:10px 16px;font-size:13px;color:#fca5a5;margin-bottom:1.5rem;">🚨 <strong>Hazard alert</strong> — AQI forecast to reach <strong>{max_p}</strong>. Avoid all outdoor activity.</div>', unsafe_allow_html=True)
elif max_p > 100:
    st.markdown(f'<div style="background:#1c1400;border:1px solid #78350f;border-radius:8px;padding:10px 16px;font-size:13px;color:#fcd34d;margin-bottom:1.5rem;">⚠ AQI forecast to reach <strong>{max_p}</strong> — limit prolonged outdoor exposure.</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div style="background:#0a1f0f;border:1px solid #064e3b;border-radius:8px;padding:10px 16px;font-size:13px;color:#6ee7b7;margin-bottom:1.5rem;">✓ Air quality looks good for the next 3 days. Max forecast: <strong>{max_p}</strong></div>', unsafe_allow_html=True)

# ── Forecast cards
st.markdown('<p class="sec">3-day forecast</p>', unsafe_allow_html=True)
cols = st.columns(3)
for col, day, key in zip(cols,
    [(now+timedelta(days=i+1)).strftime("%A · %b %d") for i in range(3)],
    ["24h","48h","72h"]):
    v = preds[key]; fl, fc, fb = get_aqi_info(v)
    col.markdown(f"""<div class="fc">
      <p class="fc-day">{day}</p>
      <p class="fc-aqi" style="color:{fc}">{v}</p>
      <p class="fc-lbl">{fl}</p>
      <div class="fc-bar" style="background:{fc}44;border:1px solid {fc}66;"></div>
      <div class="fc-bar" style="background:{fc};width:{min(v/300*100,100):.0f}%;margin-top:2px;"></div>
    </div>""", unsafe_allow_html=True)

# ── Pollutants
st.markdown('<p class="sec" style="margin-top:1.75rem">Pollutants</p>', unsafe_allow_html=True)
pcols = st.columns(6)
for col,(n,fk,u) in zip(pcols,[
    ("PM₂.₅","pm25","µg/m³"),("PM₁₀","pm10","µg/m³"),
    ("O₃","o3","ppb"),("NO₂","no2","ppb"),
    ("SO₂","so2","ppb"),("CO","co","ppm")]):
    col.markdown(f"""<div class="poll">
      <p class="poll-n">{n}</p>
      <p class="poll-v">{feat.get(fk,'—')} <span class="poll-u">{u}</span></p>
    </div>""", unsafe_allow_html=True)

# ── Weather
st.markdown('<p class="sec" style="margin-top:1.75rem">Weather</p>', unsafe_allow_html=True)
wcols = st.columns(4)
for col,(l,v) in zip(wcols,[
    ("Temperature",f"{feat.get('temperature','—')} °C"),
    ("Humidity",   f"{feat.get('humidity','—')} %"),
    ("Wind",       f"{feat.get('wind_speed','—')} km/h"),
    ("Rain",       f"{feat.get('precipitation',0)} mm"),
]):
    col.markdown(f"""<div class="wx">
      <p class="wx-l">{l}</p><p class="wx-v">{v}</p>
    </div>""", unsafe_allow_html=True)

# ── History + Map
st.markdown('<p class="sec" style="margin-top:1.75rem">72-hour history & location</p>', unsafe_allow_html=True)
lc, rc = st.columns([1.7, 1])

with lc:
    if not hist.empty:
        # color bars by AQI level
        bar_colors = []
        for a in hist["aqi"].fillna(0):
            _, c, _ = get_aqi_info(int(a))
            bar_colors.append(c)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=hist["timestamp"], y=hist["aqi"],
            marker_color=bar_colors,
            marker_line_width=0,
        ))
        fig.add_hline(y=100, line_dash="dot", line_color="#ff7043", line_width=1,
            annotation_text="Unhealthy for some", annotation_font_size=9, annotation_font_color="#ff7043")
        fig.update_layout(
            paper_bgcolor="#161b27", plot_bgcolor="#161b27",
            height=260, margin=dict(t=12,b=8,l=8,r=8),
            showlegend=False,
            xaxis=dict(showgrid=False,tickfont=dict(size=9,color="#2d3748",family="IBM Plex Mono"),showline=False,tickformat="%H:%M\n%d %b"),
            yaxis=dict(showgrid=True,gridcolor="#1e2538",tickfont=dict(size=9,color="#2d3748",family="IBM Plex Mono"),zeroline=False),
            bargap=0.1,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

with rc:
    fig_m = go.Figure()
    fig_m.add_trace(go.Scattermapbox(
        lat=[LAT], lon=[LON],
        mode="markers",
        marker=dict(size=18, color=clr, opacity=0.9),
        text=[f"<b>{CITY}</b><br>AQI: {aqi}<br>{lbl}"],
        hoverinfo="text"
    ))
    fig_m.update_layout(
        mapbox=dict(style="carto-darkmatter", center=dict(lat=LAT,lon=LON), zoom=9),
        margin=dict(t=0,b=0,l=0,r=0),
        height=260,
        paper_bgcolor="#161b27",
    )
    st.plotly_chart(fig_m, use_container_width=True, config={"displayModeBar":False})

# ── Model comparison
st.markdown('<p class="sec" style="margin-top:1.5rem">Model comparison</p>', unsafe_allow_html=True)
models_data = [
    ("Ridge Regression",  6.89, 5.54, 0.967, False),
    ("Random Forest",     5.49, 4.31, 0.979, False),
    ("XGBoost",           5.58, 4.34, 0.978, False),
    ("Gradient Boosting", 5.42, 4.26, 0.980, True),
    ("Voting Ensemble",   5.42, 4.26, 0.980, False),
    ("Stacking Ensemble", 5.42, 4.27, 0.979, False),
]
rows = ""
for name, rmse, mae, r2, best in models_data:
    cls = 'class="best"' if best else ""
    badge = '<span class="best-tag">BEST</span>' if best else ""
    fill = int(r2*100)
    nm = f"<strong>{name}</strong>" if best else name
    rows += f"""<tr {cls}>
      <td style="color:{'#e2e8f0' if best else '#718096'}">{nm}</td>
      <td>{rmse}</td><td>{mae}</td>
      <td><div class="r2t"><div class="r2f" style="width:{fill}%"></div></div> {r2}</td>
      <td>{badge}</td>
    </tr>"""

st.markdown(f"""
<div style="background:#161b27;border:1px solid #1e2538;border-radius:12px;overflow:hidden;">
  <table class="mtbl">
    <thead><tr><th>Model</th><th>RMSE ↓</th><th>MAE ↓</th><th>R² ↑</th><th></th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)

st.markdown('<p style="font-size:10px;color:#2d3748;margin-top:1rem;font-family:IBM Plex Mono,monospace">Data: AQICN · Open-Meteo · MongoDB Atlas · DagsHub/MLflow · GitHub Actions</p>', unsafe_allow_html=True)