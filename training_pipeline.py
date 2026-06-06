"""
Training Pipeline
- Loads features from MongoDB
- Trains multiple ML models for 24h, 48h, 72h targets
- Evaluates with RMSE, MAE, R²
- Logs to DagsHub (MLflow)
- Saves best models for each horizon
- Generates SHAP feature importance plots
"""

import os
import pickle
import shutil
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    VotingRegressor, StackingRegressor
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

load_dotenv()

MONGODB_URI      = os.getenv("MONGODB_URI")
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO     = os.getenv("DAGSHUB_REPO")
DAGSHUB_TOKEN    = os.getenv("DAGSHUB_TOKEN")

FEATURE_COLS = [
    "hour", "day", "month", "weekday", "is_weekend",
    "pm25", "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "wind_speed", "wind_u", "wind_v",
    "precipitation", "aqi_change_rate", "aqi"
]

HORIZONS = {
    "24h": "target_aqi_24h",
    "48h": "target_aqi_48h",
    "72h": "target_aqi_72h",
}


def load_data(target_col):
    client = MongoClient(MONGODB_URI)
    docs = list(client["aqi_db"]["features"].find(
        {target_col: {"$exists": True, "$ne": None}},
        {col: 1 for col in FEATURE_COLS + [target_col]}
    ))
    client.close()
    df = pd.DataFrame(docs)
    df.drop(columns=["_id"], errors="ignore", inplace=True)
    df.dropna(subset=["aqi", target_col], inplace=True)
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    return df


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def get_models():
    return {
        "ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "random_forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "xgboost": XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        "voting_ensemble": VotingRegressor(estimators=[
            ("rf",  RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
            ("xgb", XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0)),
            ("gb",  GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)),
        ]),
        "stacking_ensemble": StackingRegressor(
            estimators=[
                ("rf",  RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
                ("xgb", XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0)),
                ("gb",  GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)),
            ],
            final_estimator=Ridge(),
            cv=5,
            n_jobs=-1
        ),
    }


def generate_shap_plot(X_train, y_train, feature_names, horizon, output_dir="shap_plots"):
    """Train a standalone XGBoost for SHAP and save summary plot."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"   Generating SHAP plot for {horizon}...")

        sample = X_train[:500]
        y_sample = y_train[:500]

        # Train a simple XGBoost just for SHAP explanation
        xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05,
                           random_state=42, verbosity=0)
        xgb.fit(X_train, y_train)

        explainer   = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(sample)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, sample,
            feature_names=feature_names,
            show=False, plot_type="bar"
        )
        plt.title(f"SHAP Feature Importance — {horizon} Horizon", fontsize=13, pad=12)
        plt.tight_layout()
        path = os.path.join(output_dir, f"shap_{horizon}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   ✅ Saved {path}")
        return path
    except Exception as e:
        print(f"   ⚠️ SHAP plot failed: {e}")
        return None


def run_training_pipeline():
    print("🚀 Starting training pipeline...")

    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")
    mlflow.set_experiment("aqi_prediction")
    print(f"✅ MLflow → DagsHub/{DAGSHUB_REPO}")

    os.makedirs("models", exist_ok=True)
    all_results = {}

    for horizon, target_col in HORIZONS.items():
        print(f"\n{'='*50}")
        print(f"📦 Training models for {horizon} horizon (target: {target_col})")
        print(f"{'='*50}")

        df = load_data(target_col)
        available = [c for c in FEATURE_COLS if c in df.columns]
        df[available] = df[available].fillna(0)

        X = df[available].values.astype(float)
        y = df[target_col].values.astype(float)
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]
        print(f"   Records: {len(X)} | Train: {int(len(X)*0.8)} | Test: {int(len(X)*0.2)}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        best_model = None
        best_rmse  = float("inf")
        best_name  = ""

        for name, model in get_models().items():
            print(f"\n🔧 Training {name} [{horizon}]...")
            with mlflow.start_run(run_name=f"{name}_{horizon}"):
                model.fit(X_train, y_train)
                metrics = evaluate_model(model, X_test, y_test)
                print(f"   RMSE: {metrics['rmse']:.2f} | MAE: {metrics['mae']:.2f} | R²: {metrics['r2']:.3f}")

                mlflow.log_params({"model": name, "horizon": horizon, "features": len(available)})
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, artifact_path=f"model_{name}_{horizon}")

                if metrics["rmse"] < best_rmse:
                    best_rmse  = metrics["rmse"]
                    best_model = model
                    best_name  = name

        print(f"\n🏆 Best for {horizon}: {best_name} (RMSE: {best_rmse:.2f})")

        # Save model
        model_path = f"models/best_model_{horizon}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model":        best_model,
                "model_name":   best_name,
                "feature_cols": available,
                "horizon":      horizon,
                "trained_at":   datetime.now().isoformat(),
                "rmse":         best_rmse,
            }, f)
        print(f"✅ Saved {model_path}")
        all_results[horizon] = {"name": best_name, "rmse": best_rmse}

        # Generate SHAP plot using standalone XGBoost
        generate_shap_plot(X_train, y_train, available, horizon)

    # Backward compatibility
    shutil.copy("models/best_model_24h.pkl", "models/best_model.pkl")

    # Save feature cols
    with open("models/feature_cols.txt", "w") as f:
        f.write("\n".join(available))

    print("\n📊 Final Results:")
    for h, r in all_results.items():
        print(f"   {h}: {r['name']} | RMSE: {r['rmse']:.2f}")

    print("\n✅ Training pipeline complete!")


if __name__ == "__main__":
    run_training_pipeline()