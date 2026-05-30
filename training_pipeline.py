"""
Training Pipeline
- Loads features from MongoDB
- Trains multiple ML models
- Evaluates with RMSE, MAE, R²
- Logs to DagsHub (MLflow)
- Saves best model
"""

import os
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import shap
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

MONGODB_URI     = os.getenv("MONGODB_URI")
DAGSHUB_USERNAME= os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO    = os.getenv("DAGSHUB_REPO")
DAGSHUB_TOKEN   = os.getenv("DAGSHUB_TOKEN")

FEATURE_COLS = [
    "hour", "day", "month", "weekday", "is_weekend",
    "pm25", "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "wind_speed", "wind_u", "wind_v",
    "precipitation", "aqi_change_rate", "aqi"
]
TARGET_COL = "target_aqi_24h"


def load_data_from_mongodb():
    """Load feature data from MongoDB."""
    print("📦 Loading data from MongoDB...")
    client = MongoClient(MONGODB_URI)
    db = client["aqi_db"]
    collection = db["features"]

    docs = list(collection.find(
        {TARGET_COL: {"$exists": True}},
        {col: 1 for col in FEATURE_COLS + [TARGET_COL]}
    ))
    client.close()

    df = pd.DataFrame(docs)
    df.drop(columns=["_id"], errors="ignore", inplace=True)
    df.dropna(subset=["aqi", TARGET_COL], inplace=True)

    # Fill remaining NAs with median
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    print(f"   Loaded {len(df)} records")
    return df


def evaluate_model(model, X_test, y_test):
    """Calculate RMSE, MAE, R²."""
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    return {"rmse": rmse, "mae": mae, "r2": r2}, preds


def plot_shap(model, X_train, feature_names, model_name):
    """Generate SHAP feature importance plot."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train[:100])
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_train[:100],
                         feature_names=feature_names, show=False)
        path = f"shap_{model_name}.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        return path
    except Exception as e:
        print(f"  SHAP plot error: {e}")
        return None


def run_training_pipeline():
    print("🚀 Starting training pipeline...")

    # Setup MLflow → DagsHub using token directly (no OAuth)
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
    mlflow.set_tracking_uri(
        f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
    )
    print(f"✅ MLflow → DagsHub/{DAGSHUB_REPO}")

    # Load data
    df = load_data_from_mongodb()

    # Available feature cols
    available = [c for c in FEATURE_COLS if c in df.columns]

    # Fill ALL NaNs with 0 to be safe
    df[available] = df[available].fillna(0)
    df[TARGET_COL] = df[TARGET_COL].fillna(df[TARGET_COL].mean())

    X = df[available].values.astype(float)
    y = df[TARGET_COL].values.astype(float)

    # Drop any remaining NaN rows
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[mask], y[mask]
    print(f"   Clean records: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

    # Base models (reused in ensembles)
    rf  = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0)
    gb  = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)

    # Define models to try
    models = {
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),
        "random_forest": rf,
        "xgboost": xgb,
        "gradient_boosting": gb,

        # Voting Ensemble: averages predictions of RF + XGB + GB
        "voting_ensemble": VotingRegressor(estimators=[
            ("rf",  RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
            ("xgb", XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0)),
            ("gb",  GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)),
        ]),

        # Stacking: tree models feed into Ridge meta-learner
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

    best_model = None
    best_rmse = float("inf")
    best_name = ""

    mlflow.set_experiment("aqi_prediction")

    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            metrics, preds = evaluate_model(model, X_test, y_test)

            print(f"   RMSE: {metrics['rmse']:.2f} | MAE: {metrics['mae']:.2f} | R²: {metrics['r2']:.3f}")

            # Log to MLflow
            mlflow.log_params({"model": name, "features": len(available)})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path=f"model_{name}")

            # SHAP for tree models (not ensembles or pipelines)
            raw_model = model.named_steps["model"] if hasattr(model, "named_steps") else model
            if name in ["random_forest", "xgboost", "gradient_boosting"]:
                shap_path = plot_shap(raw_model, X_train, available, name)
                if shap_path:
                    mlflow.log_artifact(shap_path)

            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_model = model
                best_name = name

    print(f"\n🏆 Best model: {best_name} (RMSE: {best_rmse:.2f})")

    # Save best model locally
    os.makedirs("models", exist_ok=True)
    model_path = "models/best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": best_model,
            "model_name": best_name,
            "feature_cols": available,
            "trained_at": datetime.now().isoformat(),
            "rmse": best_rmse
        }, f)
    print(f"✅ Best model saved to {model_path}")

    # Also save feature list
    with open("models/feature_cols.txt", "w") as f:
        f.write("\n".join(available))

    print("✅ Training pipeline complete!")
    return best_model, available


if __name__ == "__main__":
    run_training_pipeline()