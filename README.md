#   AQI Predictor — Karachi

An end-to-end serverless ML system that predicts Air Quality Index (AQI) for the next 3 days.


##  Live Demo
(https://karachi-aqi.streamlit.app)

## DagsHub MLflow link 
https://dagshub.com/maryamsultan800/aqi-predictor/experiments



##  Architecture


```
AQICN API ──┐
            ├──► Feature Pipeline ──► MongoDB Atlas ──► Training Pipeline ──► DagsHub/MLflow
Open-Meteo ─┘                                                                       │
                                                                                    ▼
                                                                              Streamlit App
```

##  Project Structure

```
aqi-predictor/
├── feature_pipeline.py     # Fetch + engineer + store features hourly
├── backfill_openmeteo.py   # Fill MongoDB with historical data
├── training_pipeline.py    # Train models + log to DagsHub/MLflow
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── .env                    #  credentials
├── models/                 # Saved model files (.pkl)
└── .github/workflows/
    ├── feature_pipeline.yml    # Runs every hour
    └── training_pipeline.yml   # Runs every day
```

##  Quick Start

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

### 4. Train the model
```bash
python training_pipeline.py
```

### 5. Run the dashboard
```bash
streamlit run app.py
```

##  Automated Pipelines

Add these GitHub Secrets in your repo settings:
- `MONGODB_URI`
- `DAGSHUB_USERNAME`
- `DAGSHUB_REPO`
- `DAGSHUB_TOKEN`

Then push to GitHub — pipelines run automatically!



## Model Leaderboard (24h horizon)

| Model                 | RMSE ↓    | MAE ↓    | R² ↑  |
|-----------------------|-----------|----------|-------|
| Ridge                 | 19.08     | 12.48    | 0.647 |
| Random Forest         | 15.77     | 10.36    | 0.760 |
| XGBoost               | 14.75     | 9.66     | 0.789 |
| Gradient Boosting     | 15.91     | 10.46    | 0.755 |
| Voting Ensemble       | 15.27     | 10.23    | 0.774 |
| **Stacking Ensemble** | **14.36** | **9.39** | **0.800** |


## 🛠️ Tech Stack
- **Data**:  Open-Meteo
- **Feature Store**: MongoDB Atlas
- **Model Registry**: DagsHub (MLflow)
- **CI/CD**: GitHub Actions
- **Dashboard**: Streamlit + Plotly
- **Explainability**: SHAP
