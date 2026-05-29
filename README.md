# 🌫️ AQI Predictor — Karachi

An end-to-end serverless ML system that predicts Air Quality Index (AQI) for the next 3 days.

## 🏗️ Architecture

```
AQICN API ──┐
             ├──► Feature Pipeline ──► MongoDB Atlas ──► Training Pipeline ──► DagsHub/MLflow
Open-Meteo ─┘                                                                       │
                                                                                     ▼
                                                                              Streamlit App
```

## 📁 Project Structure

```
aqi_predictor/
├── feature_pipeline.py     # Fetch + engineer + store features
├── backfill.py             # Fill MongoDB with historical data
├── training_pipeline.py    # Train models + log to DagsHub
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── .env                    # Your credentials (never commit this!)
├── models/                 # Saved model files
└── .github/workflows/
    ├── feature_pipeline.yml   # Runs every hour
    └── training_pipeline.yml  # Runs every day
```

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up `.env` file
```
AQICN_TOKEN=your_token
MONGODB_URI=mongodb+srv://...
DAGSHUB_USERNAME=your_username
DAGSHUB_REPO=your_repo
DAGSHUB_TOKEN=your_token
CITY=Karachi
```

### 3. Backfill historical data
```bash
python backfill.py
```

### 4. Train the model
```bash
python training_pipeline.py
```

### 5. Run the dashboard
```bash
streamlit run app.py
```

## 🤖 Automated Pipelines

Add these GitHub Secrets in your repo settings:
- `AQICN_TOKEN`
- `MONGODB_URI`
- `DAGSHUB_USERNAME`
- `DAGSHUB_REPO`
- `DAGSHUB_TOKEN`

Then push to GitHub — pipelines run automatically!

## 📊 Models Trained
- Ridge Regression
- Random Forest
- XGBoost ⭐ (usually best)
- Gradient Boosting

## 🛠️ Tech Stack
- **Data**: AQICN + Open-Meteo
- **Feature Store**: MongoDB Atlas
- **Model Registry**: DagsHub (MLflow)
- **CI/CD**: GitHub Actions
- **Dashboard**: Streamlit + Plotly
- **Explainability**: SHAP
