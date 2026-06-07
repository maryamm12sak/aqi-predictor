#  AQI Predictor — Karachi

An end-to-end serverless ML system that predicts Air Quality Index (AQI) for Karachi for the next 24h, 48h, and 72h — fully automated with GitHub Actions.

## Live Demo
(https://karachi-aqi.streamlit.app)

## DagsHub MLflow Experiments
(https://dagshub.com/maryamsultan800/aqi-predictor/experiments)

##Architecture

```
Open-Meteo Air Quality API + Open-Meteo Weather API ──► Feature Pipeline ──► MongoDB Atlas ──► Training Pipeline ──► DagsHub/MLflow ──► Streamlit App

                        
                                                        
##  Project Structure

```
aqi-predictor/
├── feature_pipeline.py     # Fetch + engineer + store features hourly
├── backfill_openmeteo.py   # Fill MongoDB with historical data
├── training_pipeline.py    # Train models + log to DagsHub/MLflow + SHAP
├── app.py                  # Streamlit dashboard
├── eda.py                  # Exploratory data analysis + plots
├── requirements.txt
├── .env                    # Credentials (never commit this)
├── models/                 # Saved model files (.pkl)
├── eda_plots/              # EDA + SHAP visualizations
└── .github/workflows/
    ├── feature_pipeline.yml    # Runs every hour
    └── training_pipeline.yml   # Runs every day
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up `.env` file
```
MONGODB_URI=mongodb+srv://...
DAGSHUB_USERNAME=maryamsultan800
DAGSHUB_REPO=aqi-predictor
DAGSHUB_TOKEN=your_token
CITY=Karachi
```

### 3. Backfill historical data
```bash
python backfill_openmeteo.py
```

### 4. Train the models
```bash
python training_pipeline.py
```

### 5. Run the dashboard
```bash
streamlit run app.py
```

##  Automated Pipelines

Add these GitHub Secrets (**Settings → Secrets → Actions**):
- `MONGODB_URI`
- `DAGSHUB_USERNAME`
- `DAGSHUB_REPO`
- `DAGSHUB_TOKEN`

Then push to GitHub — pipelines run automatically. Feature pipeline runs every hour, training pipeline runs daily at 2 AM UTC.

##  Model Leaderboard (24h horizon)
## Model Leaderboard (24h horizon)

| Model                 | RMSE ↓    | MAE ↓    | R² ↑  |
|-----------------------|-----------|----------|-------|
| Ridge                 | 19.08     | 12.48    | 0.647 |
| Random Forest         | 15.77     | 10.36    | 0.760 |
| XGBoost               | 14.75     | 9.66     | 0.789 |
| Gradient Boosting     | 15.91     | 10.46    | 0.755 |
| Voting Ensemble       | 15.27     | 10.23    | 0.774 |
| **Stacking Ensemble** | **14.36** | **9.39** | **0.800** |

##  SHAP Feature Importance

SHAP analysis reveals feature importance shifts meaningfully across horizons:

| Horizon | Top Features  | Insight                                                                      |
|-----|-------------------|--------------------------------------------------|
| 24h | PM2.5, AQI, SO₂   | Current pollutant levels dominate at short range |
| 48h | AQI, Month, PM2.5 | Seasonal patterns start emerging                 |
| 72h | Month, AQI, SO₂   | Seasonal cycles take over at longer range        |

Precipitation and is_weekend contribute negligibly across all horizons.

##  Tech Stack

| Layer               | Tool |
|---------------------|------------------------------------------------------------|
| Data                | Open-Meteo Air Quality + Weather API (free, no key needed) |
| Feature Store       | MongoDB Atlas                                              |
| Experiment Tracking | DagsHub + MLflow                                           |
| CI/CD               | GitHub Actions                                             |
| Dashboard           | Streamlit + Plotly                                         |
| Explainability      | SHAP (TreeExplainer on XGBoost base learner)               |

##  Limitations

- Open-Meteo provides CAMS model-based data, not ground sensor measurements
- MongoDB feature store is credentials-protected — recreate dataset with `backfill_openmeteo.py`
- No deep learning models (LSTM/Transformer) — planned for future work
- Flask/FastAPI not used; Streamlit handles the full web layer
