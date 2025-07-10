import streamlit as st
import joblib
import pandas as pd

# Загрузка модели и векторайзера
model = joblib.load('models/xgb_model.pkl')
vectorizer = joblib.load('models/job_title_vectorizer.pkl')
kmeans = joblib.load('models/job_title_kmeans.pkl')

st.title("💰 IT Salary Prediction App")

job_title = st.text_input("Job title", "Data Scientist")
remote_ratio = st.slider("Remote ratio", 0, 100, 50)
work_year = st.number_input("Work year", 2020, 2025, 2023)
experience_level = st.selectbox("Experience level", ["EN", "MI", "SE", "EX"])
employment_type = st.selectbox("Employment type", ["FT", "PT", "CT", "FL"])
company_size = st.selectbox("Company size", ["S", "M", "L"])
company_location = st.text_input("Company location", "US")
company_location = company_location.strip().upper()

if len(company_location) != 2:
    st.warning("⚠️ Please enter a valid 2-letter country code (e.g. US, DE, IN)")

# Векторизация
job_vec = vectorizer.transform([job_title])
cluster = kmeans.predict(job_vec)[0]

# Собираем фичи как бог велел
model_features = model.feature_names_in_

user_input = {
    'remote_ratio': remote_ratio,
    'work_year': work_year,
    'job_cluster_nlp': cluster,
    f'experience_level_{experience_level}': 1,
    f'employment_type_{employment_type}': 1,
    f'company_size_{company_size}': 1,
    f'company_location_{company_location}': 1
}

# Пустой фрейм на основе фичей модели
input_df = pd.DataFrame([{feat: 0 for feat in model_features}])

# Заполняем тем, что ввёл пользователь
for key, value in user_input.items():
    if key in input_df.columns:
        input_df.loc[0, key] = value


input_df = input_df[model.feature_names_in_]

# Предсказание
pred = model.predict(input_df)[0]
st.success(f"💸 Estimated Salary: **{pred:.2f}K USD**")

# print(input_df.T)