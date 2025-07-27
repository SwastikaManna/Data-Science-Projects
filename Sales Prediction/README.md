# 📈 Sales Prediction using Support Vector Regression (SVR)

Sales prediction is crucial for businesses to make strategic decisions based on how much product or service will be sold in response to advertising efforts. This project uses **Machine Learning with Python** to predict sales using historical advertising data through **Support Vector Regression (SVR)**.

---

## 🎯 Objective

To build a robust regression model that can predict product sales based on advertising budgets across multiple channels such as **TV**, **Radio**, and **Newspaper**.

---

## 📂 Dataset

- **File**: `Advertising.csv`
- **Features**:
  - `TV`: Advertising spend on TV
  - `Radio`: Advertising spend on Radio
  - `Newspaper`: Advertising spend on Newspapers
- **Target**:
  - `Sales`: Sales of the product (in thousands of units)

> *The dataset is a classic marketing dataset used for sales response modeling.*

---

## 🛠️ Tools & Libraries

- Python 3.x
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- joblib

---

## 🚀 Project Pipeline

1. **Data Loading & Inspection**
   - Load CSV, drop unnecessary columns, view head and stats

2. **Exploratory Data Analysis (EDA)**
   - Sales distribution plot
   - Pairplot of features
   - Correlation matrix

3. **Data Preparation**
   - Feature-target split
   - Train/test split
   - Standard scaling using `StandardScaler`

4. **Model Building**
   - Hyperparameter tuning using `GridSearchCV` for SVR
   - Training the best estimator

5. **Evaluation Metrics**
   - R² Score
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - Mean Absolute Percentage Error (MAPE)

6. **Diagnostics & Visuals**
   - Actual vs Predicted scatter plot
   - Residual distribution
   - Residuals vs Predicted
   - Permutation-based feature importance

7. **Business Scenario Simulation**
   - Predict sales across hypothetical advertising strategies (TV Focus, Balanced, etc.)

8. **Model Saving**
   - Export model & scaler to `svr_advertising_model.joblib`

---

## 📊 Sample Results

| Metric | Value |
|--------|-------|
| R²     | 0.93  |
| RMSE   | ~1.5  |
| MAE    | ~1.1  |
| MAPE   | ~7%   |

---

## 📈 Business Scenarios (Example)

| Scenario         | TV  | Radio | Newspaper | Predicted Sales |
|------------------|-----|-------|-----------|-----------------|
| Conservative     | 160 | 30    | 10        | 17.26           |
| TV Focus         | 180 | 15    | 5         | 17.78           |
| Radio Emphasis   | 120 | 60    | 20        | 17.09           |
| Balanced         | 150 | 35    | 15        | 17.31           |
| Digital Focus    | 170 | 30    | 0         | 17.44           |

---
