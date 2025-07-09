# 📦 Базa
import pandas as pd
import numpy as np

# 📊 Visual
import matplotlib.pyplot as plt
import seaborn as sns

# 🔍 Пpreproc + modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# 📐 metrics + clustering
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 🔧 Pipeline + preprocessing
from config import Config
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import logging
import os

config = Config()

# Set up logging
log_path = os.path.join("logs", "run_log.txt")
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename=log_path, level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

print("=== Reading data ===")
df = pd.read_csv(config.data_path)

# Початковий аналіз / Initial exploration
print(df.info())
print(df.describe())
print(df.isnull().sum())
sns.histplot(df['salary_in_usd'], kde=True)
plt.title("Розподіл зарплат / Salary distribution")
plt.show()

# 2 Кластеризація job_title / Job title clustering

def cluster_job_titles(df, n_clusters=10):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_jobs = vectorizer.fit_transform(df['job_title'])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['job_cluster_nlp'] = kmeans.fit_predict(X_jobs)
    return vectorizer, kmeans


vectorizer, kmeans = cluster_job_titles(df, n_clusters=10)

# Перевірка представника кластера / Check dominant title per cluster
cluster_map = df.groupby('job_cluster_nlp')['job_title'].agg(
    lambda x: x.value_counts().index[0])
print(cluster_map)


df = pd.get_dummies(df, columns=['experience_level', 'employment_type', 'company_size', 'company_location'])

# Сталі та one-hot колонки / Fixed + One-hot columns
base_features = ['remote_ratio', 'work_year', 'job_cluster_nlp']
one_hot_features = [col for col in df.columns if col.startswith(('experience_', 'employment_', 'company_'))]
X = df[base_features + one_hot_features]
df['salary_in_usd'] = df['salary_in_usd'] / 1000  # Convert to thousands
y = df['salary_in_usd']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info(f"Training set size: {X_train.shape}, Validation set size: {X_val.shape}")


# Видалити кластер за потреби / Drop cluster if needed
# df = df[df['job_cluster_nlp'] != 3]  #  Приклад: видалити кластер 3 / Example: drop cluster 3

# 3 Обрізка хвоста і повторне навчання / Truncating outliers and retraining
cut_threshold = 320000  # 🔪 Рубаємо по 320 к / Cut at 350k

df_cut = df[df['salary_in_usd'] <= cut_threshold].copy()
df_cut['salary_in_usd'] = df_cut['salary_in_usd'] / 1000  # Normalize *before* extracting X

X_cut = df_cut[X.columns]
y_cut = df_cut['salary_in_usd']


X_train_cut, X_val_cut, y_train_cut, y_val_cut = train_test_split(X_cut, y_cut, test_size=0.2, random_state=42)


# 🔹 Лінійна регресія / Linear Regression
start = time.time()
logging.info("Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train_cut, y_train_cut)
pred_lr_cut = lr.predict(X_val_cut)
logging.info(f"Linear Regression trained in {time.time() - start:.2f}s")

# 🔹 Random Forest
start = time.time()
logging.info("Training Random Forest...")
rf = RandomForestRegressor(n_estimators=20, random_state=42)
rf.fit(X_train_cut, y_train_cut)
pred_rf_cut = rf.predict(X_val_cut)
logging.info(f"Random Forest trained in {time.time() - start:.2f}s")

# 🔹 XGBoost
start = time.time()
logging.info("Training XGBoost...")
xgb_cut = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
xgb_cut.fit(X_train_cut, y_train_cut)
pred_xgb_cut = xgb_cut.predict(X_val_cut)
logging.info(f"XGBoost trained in {time.time() - start:.2f}s")

# 📊 Метрики для обрізаних / Metrics after cutoff
mae_cut = mean_absolute_error(y_val_cut, pred_xgb_cut)
rmse_cut = mean_squared_error(y_val_cut, pred_xgb_cut)
# 📊 Метрики / Metrics
metrics = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
    'MAE': [mean_absolute_error(y_val_cut, pred_lr_cut),
            mean_absolute_error(y_val_cut, pred_rf_cut),
            mean_absolute_error(y_val_cut, pred_xgb_cut)],
    'RMSE': [mean_squared_error(y_val_cut, pred_lr_cut),
             mean_squared_error(y_val_cut, pred_rf_cut),
             mean_squared_error(y_val_cut, pred_xgb_cut)]
})

print(metrics)
for i, row in metrics.iterrows():
    logging.info(f"{row['Model']} — MAE: {row['MAE']:.5f}, RMSE: {row['RMSE']:.5f}")
