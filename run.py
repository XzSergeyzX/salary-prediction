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

# # Avoid target leakage
# X = df.drop([config.target_column, 'salary'], axis=1, errors='ignore')
# y = df[config.target_column]

# categorical = X.select_dtypes(include=['object']).columns.tolist()
# numerical = X.select_dtypes(exclude=['object']).columns.tolist()

# print("=== Configuring preprocessor ===")
# preprocessor = ColumnTransformer([
#     ("num", StandardScaler(), numerical),
#     ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
# ])

# print("=== Building pipeline ===")
# model = make_pipeline(preprocessor, LinearRegression())

# print("=== Splitting data ===")
# X_train, X_val, y_train, y_val = train_test_split(
#     X, y, test_size=config.test_size, random_state=config.random_state
# )

# print("Train shape:", X_train.shape)
# print("Val shape:", X_val.shape)

# print("=== Training model ===")
# start = time.time()
# for _ in tqdm(range(1), desc="Training"):
#     model.fit(X_train, y_train)
# training_time = round(time.time() - start, 2)
# print("Training time:", training_time, "seconds")

# print("=== Making predictions ===")
# preds = model.predict(X_val)

# print("=== Evaluating performance ===")
# mse = mean_squared_error(y_val, preds)
# r2 = r2_score(y_val, preds)
# print("MSE:", mse)
# print("R²:", r2)

# # Log performance
# logging.info(f"MSE: {mse}")
# logging.info(f"R²: {r2}")
# logging.info(f"Training time: {training_time} seconds")

# # Plot predictions vs actual values
# plt.figure(figsize=(10, 6))
# plt.scatter(y_val, preds, alpha=0.5)
# plt.xlabel("Actual Salary")
# plt.ylabel("Predicted Salary")
# plt.title("Actual vs Predicted Salary")
# plt.grid(True)
# plt.tight_layout()
# plot_path = os.path.join("logs", "salary_prediction_plot.png")
# plt.savefig(plot_path)
# print(f"Plot saved to {plot_path}")

# # Cross-validation
# print("=== Cross-validation ===")
# cross_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
# print("Cross-validated R² scores:", cross_scores)
# print("Average CV R²:", cross_scores.mean())
# logging.info(f"Cross-validated R² scores: {cross_scores.tolist()}")
# logging.info(f"Average CV R²: {cross_scores.mean()}")
