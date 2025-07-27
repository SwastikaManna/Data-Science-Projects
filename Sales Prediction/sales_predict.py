# ==========================================
# Sales Prediction with SVR + Full EDA
# Generates all charts & tables automatically
# ==========================================
#
# REQUIRED LIBRARIES
# ------------------
# !pip install pandas numpy matplotlib seaborn scikit-learn joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
)
import joblib

# -------- 1. Load & inspect the data -----------------------------------------
df = pd.read_csv("Advertising.csv").drop(columns=["Unnamed: 0"], errors="ignore")

print("\n=== HEAD ===")
print(df.head(), "\n")

print("=== DESCRIPTIVE STATISTICS ===")
print(df.describe().T, "\n")

# -------- 2. Exploratory Data Analysis (EDA) ---------------------------------
plt.figure(figsize=(6, 4))
sns.histplot(df["Sales"], kde=True, bins=20, color="steelblue")
plt.title("Sales Distribution")
plt.xlabel("Sales (k units)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.pairplot(df, diag_kind="kde")
plt.suptitle("Pairplot: TV, Radio, Newspaper vs Sales", y=1.02)
plt.show()

plt.figure(figsize=(5, 4))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# -------- 3. Prepare data -----------------------------------------------------
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# -------- 4. Hyper-parameter search for SVR ----------------------------------
param_grid = {
    "C": [10, 100, 1000],
    "gamma": [0.01, 0.1, 1],
    "kernel": ["rbf"]
}
svr = SVR()
grid = GridSearchCV(
    estimator=svr,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=0
)
grid.fit(X_train_scaled, y_train)

best_svr = grid.best_estimator_
print("=== BEST PARAMETERS ===", grid.best_params_, "\n")

# -------- 5. Train best model & evaluate -------------------------------------
best_svr.fit(X_train_scaled, y_train)
y_pred = best_svr.predict(X_test_scaled)

metrics = {
    "R2"  : r2_score(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    "MAE" : mean_absolute_error(y_test, y_pred),
    "MAPE": mean_absolute_percentage_error(y_test, y_pred)
}
print("=== TEST-SET PERFORMANCE ===")
for k, v in metrics.items():
    if k == "R2":
        print(f"{k:>4}: {v:0.4f}  ({v:0.2%} variance explained)")
    else:
        print(f"{k:>4}: {v:0.4f}")

# -------- 6. Diagnostic plots -------------------------------------------------
# Actual vs Predicted
plt.figure(figsize=(5, 5))
sns.scatterplot(x=y_test, y=y_pred, color="darkorange", s=60, edgecolor="k")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=1)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("SVR: Actual vs Predicted")
plt.tight_layout()
plt.show()

# Residuals
residuals = y_test - y_pred
plt.figure(figsize=(5, 4))
sns.histplot(residuals, kde=True, color="forestgreen", bins=20)
plt.title("Residual Distribution")
plt.xlabel("Residual (Actual − Predicted)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 4))
sns.scatterplot(x=y_pred, y=residuals, color="purple", edgecolor="k")
plt.axhline(0, ls="--", lw=1, color="k")
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.tight_layout()
plt.show()

# -------- 7. Simple feature importance via permutation -----------------------
from sklearn.inspection import permutation_importance
result = permutation_importance(
    best_svr, X_test_scaled, y_test, n_repeats=30, random_state=42, n_jobs=-1
)
importance = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(5, 4))
sns.barplot(x=importance.values, y=importance.index, palette="viridis")
plt.title("Permutation Feature Importance")
plt.xlabel("Decrease in R2 score")
plt.tight_layout()
plt.show()

print("\n=== FEATURE IMPORTANCE (Permutation) ===")
print(importance.to_frame("Importance").round(4), "\n")

# -------- 8. Business-scenario table -----------------------------------------
scenarios = pd.DataFrame({
    "Scenario": ["Conservative", "TV Focus", "Radio Emphasis", "Balanced", "Digital Focus"],
    "TV":       [160, 180, 120, 150, 170],
    "Radio":    [30, 15, 60, 35, 30],
    "Newspaper":[10, 5, 20, 15,  0]
})
scenarios["Predicted_Sales"] = best_svr.predict(scaler.transform(scenarios[["TV", "Radio", "Newspaper"]]))
print("=== BUSINESS SCENARIOS ===")
print(scenarios.to_string(index=False), "\n")

# -------- 9. Save model for production ---------------------------------------
joblib.dump({"model": best_svr, "scaler": scaler}, "svr_advertising_model.joblib")
print("Model and scaler saved to svr_advertising_model.joblib")
a