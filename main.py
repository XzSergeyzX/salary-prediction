# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# --- Загрузка артефактов ---
model = joblib.load("models/xgb_model.pkl")
vectorizer = joblib.load("models/job_title_vectorizer.pkl")
kmeans = joblib.load("models/job_title_kmeans.pkl")

# --- FastAPI instance ---
app = FastAPI(title="💰 Salary Prediction API", version="1.0")

# --- Pydantic модель ---
class JobInput(BaseModel):
    job_title: str
    remote_ratio: int
    work_year: int
    experience_level: str
    employment_type: str
    company_size: str
    company_location: str

# --- Prediction route ---
@app.get("/")
def root():
    return {"message": "Welcome to the Salary Prediction API. Go to /docs to use it."}

@app.post("/predict")
def predict_salary(data: JobInput):
    job_vec = vectorizer.transform([data.job_title])
    cluster = kmeans.predict(job_vec)[0]

    model_features = model.feature_names_in_
    input_dict = {
        'remote_ratio': data.remote_ratio,
        'work_year': data.work_year,
        'job_cluster_nlp': cluster,
        f'experience_level_{data.experience_level}': 1,
        f'employment_type_{data.employment_type}': 1,
        f'company_size_{data.company_size}': 1,
        f'company_location_{data.company_location.upper()}': 1
    }

    input_df = pd.DataFrame([{feat: 0 for feat in model_features}])
    for key, value in input_dict.items():
        if key in input_df.columns:
            input_df.loc[0, key] = value

    input_df = input_df[model_features]
    pred = model.predict(input_df)[0]
    print("Input dict:", input_dict)
    print("Input DF before prediction:", input_df.head())

    return {"predicted_salary_k": float(round(pred, 2))}

