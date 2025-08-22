# Iris Flower Classification – End-to-End Python Project

This project demonstrates a complete machine learning workflow for classifying Iris flower species using the Random Forest algorithm in Python. It includes data loading, preprocessing, stratified train/test split, hyperparameter tuning with GridSearchCV, model evaluation, and helper utilities for single-sample predictions.

---

## 🚀 Project Overview

- **Objective:** Classify Iris flower species (Setosa, Versicolor, Virginica) using scikit-learn's Iris dataset and Random Forests.
- **Pipeline:** Loads data, splits with stratification, tunes hyperparameters, evaluates model, visualizes results, and provides single-sample prediction functionality.

---

## 🗂️ Dataset

- **Source:** scikit-learn's built-in Iris dataset
- **Features:**
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- **Target Classes:** Setosa, Versicolor, Virginica

---

## ⚙️ Key Features

- Loads and preprocesses the Iris dataset using Pandas DataFrames
- Performs stratified 80/20 train-test split
- Tunes Random Forest hyperparameters with GridSearchCV and StratifiedKFold
- Evaluates with test accuracy, classification report, confusion matrix
- Visualizes confusion matrix and feature importances using Matplotlib
- Helper function for single-sample species prediction with correct feature names
---


## 📑 Usage

- Execute the script directly. Upon completion, it will print model performance metrics and example predictions for new flower measurements.
- Use the `predict_iris_species(model, [sepal_len, sepal_wid, petal_len, petal_wid])` function for classifying new samples.

---

## 🖼️ Outputs & Visualizations

- Plots: Confusion matrix and feature importances
- Terminal: Best hyperparameters, accuracy, classification report, example predictions

---

## 🧩 Dependencies

- numpy
- pandas
- scikit-learn
- matplotlib

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Please open a pull request or issue.

---

## 📄 License

This project is licensed under the MIT License.

---

**Happy Learning and Coding!**
