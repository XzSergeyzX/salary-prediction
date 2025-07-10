import gradio as gr
import joblib
import pandas as pd

# Загружаем всё священное
model = joblib.load("models/xgb_model.pkl")
vectorizer = joblib.load("models/job_title_vectorizer.pkl")
kmeans = joblib.load("models/job_title_kmeans.pkl")
model_features = model.feature_names_in_

# Основная функция
def predict_salary(job_title, remote_ratio, work_year, experience_level, employment_type, company_size, company_location):
    company_location = company_location.strip().upper()

    if len(company_location) != 2:
        return "❌ Invalid country code. Use 2-letter codes like US, DE, FR."

    job_vec = vectorizer.transform([job_title])
    cluster = kmeans.predict(job_vec)[0]

    user_input = {
        'remote_ratio': remote_ratio,
        'work_year': work_year,
        'job_cluster_nlp': cluster,
        f'experience_level_{experience_level}': 1,
        f'employment_type_{employment_type}': 1,
        f'company_size_{company_size}': 1,
        f'company_location_{company_location}': 1
    }

    input_df = pd.DataFrame([{feat: 0 for feat in model_features}])

    for key, value in user_input.items():
        if key in input_df.columns:
            input_df.loc[0, key] = value

    pred = model.predict(input_df)[0]
    return f"💸 Estimated Salary: {pred:.2f}K USD"

# UI компоненты
iface = gr.Interface(
    fn=predict_salary,
    inputs=[
        gr.Textbox(label="Job title", value="Data Scientist"),
        gr.Slider(0, 100, value=50, label="Remote ratio"),
        gr.Number(label="Work year", value=2023),
        gr.Dropdown(["EN", "MI", "SE", "EX"], label="Experience level"),
        gr.Dropdown(["FT", "PT", "CT", "FL"], label="Employment type"),
        gr.Dropdown(["S", "M", "L"], label="Company size"),
        gr.Textbox(label="Company location (2-letter code)", value="US")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="💰 IT Salary Prediction App (Gradio)",
    description="Enter job info and get an estimated salary in K USD."
)

iface.launch()