"""
Exploratory Data Analysis (EDA)
- AQI distribution
- Time-of-day patterns
- Monthly trends
- Correlation heatmap
- Pollutant analysis
- AQI change rate
Saves plots as PNG and displays in Streamlit
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#161b27",
    "axes.edgecolor":    "#1e2538",
    "axes.labelcolor":   "#a0aec0",
    "xtick.color":       "#4a5568",
    "ytick.color":       "#4a5568",
    "text.color":        "#e2e8f0",
    "grid.color":        "#1e2538",
    "grid.linewidth":    0.8,
    "font.family":       "monospace",
    "axes.titlesize":    13,
    "axes.titlecolor":   "#e2e8f0",
    "axes.titleweight":  "bold",
})

AQI_COLORS = {
    "Good":              "#00e676",
    "Moderate":          "#ffca28",
    "Unhealthy for Some":"#ff7043",
    "Unhealthy":         "#ef5350",
    "Very Unhealthy":    "#ab47bc",
    "Hazardous":         "#b71c1c",
}

def classify_aqi(aqi):
    if aqi <= 50:  return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Some"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"


def load_data():
    print("📦 Loading data from MongoDB...")
    client = MongoClient(MONGODB_URI)
    docs = list(client["aqi_db"]["features"].find(
        {}, {"timestamp":1,"aqi":1,"pm25":1,"pm10":1,"o3":1,"no2":1,
             "so2":1,"co":1,"temperature":1,"humidity":1,"wind_speed":1,
             "precipitation":1,"aqi_change_rate":1,"hour":1,"month":1,"weekday":1}
    ))
    client.close()
    df = pd.DataFrame(docs).drop(columns=["_id"], errors="ignore")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["category"]  = df["aqi"].apply(classify_aqi)
    print(f"   {len(df)} records loaded")
    return df.dropna(subset=["aqi"])


def save(fig, name):
    os.makedirs("eda_plots", exist_ok=True)
    path = f"eda_plots/{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150, facecolor="#0f1117")
    plt.close(fig)
    print(f"   Saved {path}")
    return path


def plot_aqi_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("AQI Distribution — Karachi", fontsize=14, color="#e2e8f0", fontweight="bold")

    # Histogram
    ax = axes[0]
    colors = [AQI_COLORS[classify_aqi(v)] for v in df["aqi"]]
    ax.hist(df["aqi"], bins=40, color="#3182ce", edgecolor="#0f1117", linewidth=0.3)
    ax.axvline(df["aqi"].mean(),  color="#ffca28", linewidth=1.5, linestyle="--", label=f"Mean: {df['aqi'].mean():.1f}")
    ax.axvline(df["aqi"].median(),color="#00e676", linewidth=1.5, linestyle=":",  label=f"Median: {df['aqi'].median():.1f}")
    ax.set_xlabel("AQI"); ax.set_ylabel("Count")
    ax.set_title("AQI Frequency Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y")

    # Category pie
    ax2 = axes[1]
    counts = df["category"].value_counts()
    pie_colors = [AQI_COLORS[c] for c in counts.index]
    wedges, texts, autotexts = ax2.pie(
        counts.values, labels=counts.index,
        colors=pie_colors, autopct="%1.1f%%",
        startangle=140, pctdistance=0.8,
        textprops={"color":"#a0aec0","fontsize":9}
    )
    for at in autotexts: at.set_color("#0f1117"); at.set_fontsize(8)
    ax2.set_title("AQI Category Breakdown")

    return save(fig, "01_aqi_distribution")


def plot_hourly_pattern(df):
    fig, ax = plt.subplots(figsize=(13, 5))
    hourly = df.groupby("hour")["aqi"].agg(["mean","std"]).reset_index()

    bar_colors = [AQI_COLORS[classify_aqi(v)] for v in hourly["mean"]]
    bars = ax.bar(hourly["hour"], hourly["mean"], color=bar_colors,
                  edgecolor="#0f1117", linewidth=0.3, alpha=0.85)
    ax.errorbar(hourly["hour"], hourly["mean"], yerr=hourly["std"],
                fmt="none", color="#4a5568", capsize=3, linewidth=0.8)
    ax.axhline(100, color="#ff7043", linewidth=1, linestyle="--", alpha=0.6, label="Unhealthy threshold (100)")
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("Average AQI")
    ax.set_title("Average AQI by Hour of Day — Karachi")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" if h % 3 == 0 else "" for h in range(24)], rotation=45, fontsize=8)
    ax.legend(fontsize=9); ax.grid(True, axis="y")

    # Annotate peak
    peak_hour = hourly.loc[hourly["mean"].idxmax(), "hour"]
    peak_val  = hourly["mean"].max()
    ax.annotate(f"Peak: {peak_val:.0f}\n{peak_hour:02d}:00",
                xy=(peak_hour, peak_val), xytext=(peak_hour+1.5, peak_val+5),
                arrowprops=dict(arrowstyle="->", color="#ffca28", lw=1),
                color="#ffca28", fontsize=9)

    return save(fig, "02_hourly_pattern")


def plot_monthly_trend(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Seasonal AQI Patterns — Karachi", fontsize=14, color="#e2e8f0", fontweight="bold")

    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    # Monthly average
    ax = axes[0]
    monthly = df.groupby("month")["aqi"].mean().reindex(range(1,13))
    bar_colors = [AQI_COLORS[classify_aqi(v)] for v in monthly.fillna(0)]
    ax.bar(range(1,13), monthly, color=bar_colors, edgecolor="#0f1117", linewidth=0.3, alpha=0.85)
    ax.set_xticks(range(1,13)); ax.set_xticklabels(month_names, fontsize=9)
    ax.set_xlabel("Month"); ax.set_ylabel("Average AQI")
    ax.set_title("Monthly Average AQI"); ax.grid(True, axis="y")

    # Weekday pattern
    ax2 = axes[1]
    day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    weekly = df.groupby("weekday")["aqi"].mean().reindex(range(7))
    wd_colors = [AQI_COLORS[classify_aqi(v)] for v in weekly.fillna(0)]
    ax2.bar(range(7), weekly, color=wd_colors, edgecolor="#0f1117", linewidth=0.3, alpha=0.85)
    ax2.set_xticks(range(7)); ax2.set_xticklabels(day_names, fontsize=9)
    ax2.set_xlabel("Day of Week"); ax2.set_ylabel("Average AQI")
    ax2.set_title("Weekday vs Weekend AQI"); ax2.grid(True, axis="y")
    ax2.axvspan(4.5, 6.5, alpha=0.07, color="#3182ce", label="Weekend")
    ax2.legend(fontsize=9)

    return save(fig, "03_seasonal_trends")


def plot_correlation(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    cols = ["aqi","pm25","pm10","o3","no2","so2","co","temperature","humidity","wind_speed"]
    corr = df[cols].dropna().corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, ax=ax,
        cmap="RdYlGn", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size":9, "color":"#0f1117"},
        linewidths=0.5, linecolor="#0f1117",
        cbar_kws={"shrink":0.8}
    )
    ax.set_title("Feature Correlation Heatmap", pad=15)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)

    return save(fig, "04_correlation_heatmap")


def plot_pollutant_vs_aqi(df):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Pollutant vs AQI Relationships — Karachi", fontsize=14, color="#e2e8f0", fontweight="bold")
    axes = axes.flatten()

    pollutants = [("pm25","PM2.5 (µg/m³)"),("pm10","PM10 (µg/m³)"),
                  ("o3","O₃ (ppb)"),("no2","NO₂ (ppb)"),
                  ("so2","SO₂ (ppb)"),("co","CO (ppm)")]

    for ax, (col, label) in zip(axes, pollutants):
        sub = df[[col,"aqi","category"]].dropna()
        if sub.empty: continue
        scatter_colors = [AQI_COLORS[c] for c in sub["category"]]
        ax.scatter(sub[col], sub["aqi"], c=scatter_colors, alpha=0.4, s=8, linewidths=0)
        # Trend line
        try:
            z = np.polyfit(sub[col], sub["aqi"], 1)
            p = np.poly1d(z)
            xs = np.linspace(sub[col].min(), sub[col].max(), 100)
            ax.plot(xs, p(xs), color="#3182ce", linewidth=1.5, alpha=0.8)
        except: pass
        ax.set_xlabel(label, fontsize=9); ax.set_ylabel("AQI", fontsize=9)
        ax.set_title(f"{label.split(' ')[0]} vs AQI")
        ax.grid(True, alpha=0.3)
        r = sub[[col,"aqi"]].corr().iloc[0,1]
        ax.text(0.97, 0.05, f"r = {r:.2f}", transform=ax.transAxes,
                ha="right", fontsize=9, color="#ffca28")

    fig.tight_layout()
    return save(fig, "05_pollutant_vs_aqi")


def plot_aqi_change_rate(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("AQI Change Rate Analysis — Karachi", fontsize=14, color="#e2e8f0", fontweight="bold")

    cr = df["aqi_change_rate"].dropna()

    # Distribution
    ax = axes[0]
    ax.hist(cr, bins=50, color="#3182ce", edgecolor="#0f1117", linewidth=0.3)
    ax.axvline(0, color="#ffca28", linewidth=1.5, linestyle="--")
    ax.axvline(cr.mean(), color="#00e676", linewidth=1.5, linestyle=":",
               label=f"Mean: {cr.mean():.2f}")
    ax.set_xlabel("AQI Change per Hour"); ax.set_ylabel("Count")
    ax.set_title("Hourly AQI Change Rate Distribution")
    ax.legend(fontsize=9); ax.grid(True, axis="y")
    pct_improving = (cr < 0).mean() * 100
    ax.text(0.97, 0.95, f"{pct_improving:.1f}% hours improving",
            transform=ax.transAxes, ha="right", fontsize=9, color="#00e676")

    # Change rate by hour
    ax2 = axes[1]
    hourly_cr = df.groupby("hour")["aqi_change_rate"].mean()
    colors = ["#00e676" if v < 0 else "#ef5350" for v in hourly_cr]
    ax2.bar(hourly_cr.index, hourly_cr.values, color=colors,
            edgecolor="#0f1117", linewidth=0.3, alpha=0.85)
    ax2.axhline(0, color="#a0aec0", linewidth=0.8)
    ax2.set_xlabel("Hour of Day"); ax2.set_ylabel("Avg AQI Change/hr")
    ax2.set_title("AQI Change Rate by Hour")
    ax2.set_xticks(range(0,24,3))
    ax2.set_xticklabels([f"{h:02d}:00" for h in range(0,24,3)], rotation=45, fontsize=8)
    ax2.grid(True, axis="y")

    return save(fig, "06_aqi_change_rate")


def print_summary(df):
    print("\n" + "="*50)
    print("📊 EDA SUMMARY — KARACHI AQI")
    print("="*50)
    print(f"Total records:     {len(df):,}")
    print(f"Date range:        {df['timestamp'].min().strftime('%Y-%m-%d')} → {df['timestamp'].max().strftime('%Y-%m-%d')}")
    print(f"\nAQI Statistics:")
    print(f"  Mean:            {df['aqi'].mean():.1f}")
    print(f"  Median:          {df['aqi'].median():.1f}")
    print(f"  Std Dev:         {df['aqi'].std():.1f}")
    print(f"  Min:             {df['aqi'].min():.1f}")
    print(f"  Max:             {df['aqi'].max():.1f}")
    print(f"\nCategory breakdown:")
    for cat, cnt in df["category"].value_counts().items():
        pct = cnt/len(df)*100
        print(f"  {cat:<22} {cnt:>5} ({pct:.1f}%)")
    print(f"\nPeak pollution hour: {df.groupby('hour')['aqi'].mean().idxmax():02d}:00")
    print(f"Cleanest hour:       {df.groupby('hour')['aqi'].mean().idxmin():02d}:00")
    print(f"\nStrongest AQI correlations:")
    cols = ["pm25","pm10","o3","no2","so2","co","temperature","humidity","wind_speed"]
    corrs = df[cols+["aqi"]].corr()["aqi"].drop("aqi").abs().sort_values(ascending=False)
    for feat, val in corrs.head(5).items():
        print(f"  {feat:<15} r = {val:.3f}")
    print("="*50)


def run_eda():
    df = load_data()
    print_summary(df)

    print("\n🎨 Generating plots...")
    p1 = plot_aqi_distribution(df)
    p2 = plot_hourly_pattern(df)
    p3 = plot_monthly_trend(df)
    p4 = plot_correlation(df)
    p5 = plot_pollutant_vs_aqi(df)
    p6 = plot_aqi_change_rate(df)

    print(f"\n✅ EDA complete! 6 plots saved to eda_plots/")
    return df, [p1, p2, p3, p4, p5, p6]


if __name__ == "__main__":
    run_eda()