"""
Iris Flower Classification – Final End-to-End Python Script
----------------------------------------------------------
• Loads Iris data directly from scikit-learn
• Keeps data in Pandas DataFrames ⇒ “valid feature names” warnings
• Performs stratified train/test split (80 / 20)
• Tunes a Random-Forest with GridSearchCV (5-fold CV)
• Evaluates on the held-out test set (accuracy, classification report, confusion matrix)
• Prints feature importances
• Provides a helper function for clean single-sample predictions
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from typing import Sequence

RANDOM_STATE = 42  # reproducibility


# --------------------------------------------------------------------
# 1. Load and prepare the dataset
# --------------------------------------------------------------------
def load_and_prepare_data() -> tuple[pd.DataFrame, pd.Series]:
    iris = load_iris(as_frame=True)
    X = iris.data                         # DataFrame (150 × 4)
    y = iris.target                       # Series   (150,)
    y.name = "Species"
    return X, y


# --------------------------------------------------------------------
# 2. Train-test split (stratified)
# --------------------------------------------------------------------
def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )


# --------------------------------------------------------------------
# 3. Hyper-parameter tuning with GridSearchCV
# --------------------------------------------------------------------
def tune_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series
) -> RandomForestClassifier:
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 4, 6],
        "min_samples_split": [2, 4, 6],
        "max_features": ["sqrt", "log2", 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rf_base = RandomForestClassifier(random_state=RANDOM_STATE)

    grid = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    ).fit(X_train, y_train)

    print("Best hyper-parameters:", grid.best_params_)
    return grid.best_estimator_  # already refit on the full training set


# --------------------------------------------------------------------
# 4. Model evaluation utilities
# --------------------------------------------------------------------
def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_names: Sequence[str],
) -> None:
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=target_names
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.show()

    # Feature importance bar plot
    importances = pd.Series(model.feature_importances_, index=X_test.columns)
    importances.sort_values(ascending=True).plot.barh(color="teal")
    plt.title("Random-Forest – Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------
# 5. Single-sample prediction helper (avoids warning)
# --------------------------------------------------------------------
def predict_iris_species(
    model: RandomForestClassifier,
    measurements: Sequence[float]
) -> str:
    """
    Parameters
    ----------
    model : fitted RandomForestClassifier
    measurements : (sepal_len, sepal_wid, petal_len, petal_wid)

    Returns
    -------
    Predicted species name (str)
    """
    iris = load_iris()
    sample = pd.DataFrame(
        [measurements],
        columns=iris.feature_names  # matches training column names exactly
    )
    pred_idx = model.predict(sample)[0]
    return iris.target_names[pred_idx]


# --------------------------------------------------------------------
# 6. Main pipeline
# --------------------------------------------------------------------
def main() -> None:
    print("Loading and preparing data …")
    X, y = load_and_prepare_data()
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\nTuning and training Random-Forest model …")
    model = tune_random_forest(X_train, y_train)
    print("Model training completed!")

    print("\nEvaluating model performance …")
    evaluate_model(model, X_test, y_test, load_iris().target_names)

    # Demonstrate example predictions
    examples = [
        (5.1, 3.5, 1.4, 0.2),  # setosa
        (6.0, 2.7, 5.1, 1.6),  # versicolor
        (6.3, 3.3, 6.0, 2.5),  # virginica
    ]
    print("\nExample Predictions:")
    for m in examples:
        species = predict_iris_species(model, m)
        print(f"Input: {m}  →  {species}")


if __name__ == "__main__":
    main()
