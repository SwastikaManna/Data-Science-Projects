# 📧 Email Spam Detection with Machine Learning

Spam emails are more than just annoying—they can be *dangerous*, tricking users into scams or phishing attacks. This project uses **Python** and **Machine Learning** to build an intelligent email spam detector that can classify emails as **spam** or **ham** with high accuracy.

---

## 🎯 Objective

To develop a complete machine learning pipeline that:
- Cleans and preprocesses email text
- Visualizes key insights
- Trains multiple models (Naive Bayes, Logistic Regression, SVM)
- Selects and tunes the best one
- Outputs predictions with clear evaluation

---

## 🧠 Dataset

- Composed of multiple `.xlsx` files (e.g. `spam-1.xlsx` to `spam-6.xlsx`)
- Columns:
  - `v1`: Label (ham/spam)
  - `v2`: Email content

> ⚠️ Ensure all `spam-#.xlsx` files are placed in the project directory before running.

---

## 🛠️ Tech Stack

- **Language**: Python 3.x  
- **Libraries**:  
  - `pandas`, `numpy`, `matplotlib`, `seaborn`  
  - `scikit-learn`, `joblib`  
  - `wordcloud`  
  - `re`, `string`, `warnings`, `Pathlib`  

---

## 🚀 Project Pipeline

1. **Data Loading**  
   Load and combine all Excel files into a unified dataframe.

2. **Text Cleaning**  
   Strip URLs, numbers, punctuation, and stopwords.

3. **Exploratory Data Analysis (EDA)**  
   - Class distribution  
   - Word clouds  
   - Message length histograms

4. **Vectorization**  
   Use `TfidfVectorizer` to convert text into numerical features.

5. **Model Training**  
   Train and evaluate:
   - Naive Bayes (MultinomialNB)
   - Logistic Regression
   - Linear SVM

6. **Model Evaluation**  
   - Accuracy, Precision, Recall, F1-score  
   - Confusion Matrix  
   - Permutation Feature Importance

7. **Hyperparameter Tuning**  
   Use GridSearchCV for the best SVM settings.

8. **Model Saving**  
   Save the trained pipeline using `joblib`.

9. **Live Prediction Demo**  
   Function to test predictions on custom messages.

---

## 📊 Visuals

- 🔹 Class distribution bar plot  
- 🔹 Spam vs Ham word clouds  
- 🔹 Message length histogram  
- 🔹 Top-20 predictive tokens bar chart  
- 🔹 Confusion matrix visualization  

---
