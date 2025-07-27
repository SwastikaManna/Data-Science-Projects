# UNEMPLOYMENT ANALYSIS WITH PYTHON

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- 1. DATA LOADING ---

df = pd.read_csv("Unemployment in India.csv")
df.dropna(how='all', inplace=True)  # Drop completely empty rows

# --- 2. DATA CLEANING & PREPROCESSING ---

# Rename for convenience
df.columns = [
    "Region", "Date", "Frequency", "Unemployment_Rate", "Employed", "Labour_Participation_Rate", "Area"
]

# Parse date
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date", "Unemployment_Rate", "Employed", "Labour_Participation_Rate", "Area", "Region"])

# Convert to numeric
df["Unemployment_Rate"] = pd.to_numeric(df["Unemployment_Rate"], errors="coerce")
df["Employed"] = pd.to_numeric(df["Employed"], errors="coerce")
df["Labour_Participation_Rate"] = pd.to_numeric(df["Labour_Participation_Rate"], errors="coerce")
df = df.dropna()

# --- 3. FEATURE ENGINEERING ---

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Days_Since_Start"] = (df["Date"] - df["Date"].min()).dt.days
df["COVID_Period"] = (df["Date"] >= "2020-03-01").astype(int)
df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

# Encode categoricals
le_region = LabelEncoder()
le_area = LabelEncoder()
df["Region_Encoded"] = le_region.fit_transform(df["Region"])
df["Area_Encoded"] = le_area.fit_transform(df["Area"])

# --- 4. EXPLORATORY DATA ANALYSIS ---

plt.figure(figsize=(16,5))
sns.lineplot(data=df, x="Date", y="Unemployment_Rate", hue="Area", marker="o")
plt.title("Unemployment Rate Trend (Rural vs Urban)")
plt.ylabel("Unemployment Rate (%)")
plt.xlabel("Date")
plt.legend(title="Area")
plt.show()

# Monthly average unemployment
monthly_avg = df.groupby(["Date", "Area"])["Unemployment_Rate"].mean().reset_index()
pivot = monthly_avg.pivot(index="Date", columns="Area", values="Unemployment_Rate")
pivot.plot(figsize=(14,6), marker="o")
plt.title("Monthly Average Unemployment Rates by Area Type")
plt.ylabel("Unemployment Rate (%)")
plt.xlabel("Month")
plt.show()

# --- 5. MODELING ---

# Model features and target
features = [
    "Region_Encoded", "Area_Encoded", "Year", "Month", "Month_sin", "Month_cos",
    "Days_Since_Start", "Labour_Participation_Rate", "Employed", "COVID_Period"
]
target = "Unemployment_Rate"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# --- 6. EVALUATION ---

def report(model_name, y_true, y_pred):
    print(f"{model_name}")
    print(f" - R2 Score: {r2_score(y_true, y_pred):.4f}")
    print(f" - MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f" - MSE: {mean_squared_error(y_true, y_pred):.4f}\n")

report("Random Forest", y_test, rf_pred)
report("Linear Regression", y_test, lr_pred)

# Feature Importance (Random Forest only)
feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(data=feature_importance, y="Feature", x="Importance", palette="crest")
plt.title("Feature Importance: Random Forest")
plt.show()

# --- 7. PREDICT NEW CASE (Optional) ---

def predict_unemployment(region, area, year, month, lpr, employed, covid_period):
    encoded_region = le_region.transform([region])[0]
    encoded_area = le_area.transform([area])[0]
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    days_since_start = (pd.Timestamp(year=year, month=month, day=1) - df["Date"].min()).days

    features_new = [[
        encoded_region, encoded_area, year, month, month_sin, month_cos, days_since_start,
        lpr, employed, covid_period
    ]]
    rf_prediction = rf.predict(features_new)[0]
    lr_prediction = lr.predict(features_new)[0]

    print(f"Random Forest Prediction: {rf_prediction:.2f}%")
    print(f"Linear Regression Prediction: {lr_prediction:.2f}%")

# Example:
# predict_unemployment("Tamil Nadu", "Urban", 2020, 5, 40, 3000000, 1)

# --- 8. INSIGHTS (Optional Printout) ---

pre_covid = df[df["COVID_Period"] == 0]["Unemployment_Rate"]
post_covid = df[df["COVID_Period"] == 1]["Unemployment_Rate"]
print(f"Pre-COVID average unemployment rate: {pre_covid.mean():.2f}%")
print(f"COVID period average unemployment rate: {post_covid.mean():.2f}%")
print(f"Increase during COVID: {((post_covid.mean() - pre_covid.mean())/pre_covid.mean())*100:.1f}%")
