# ==========================================
# EMAIL SPAM DETECTION – FULL PIPELINE
# ==========================================
#
# HOW TO RUN
# ----------
# 1. Place this notebook (or .py file) in the same folder as
#    spam-1.xlsx … spam-6.xlsx (or adjust the paths below).
# 2. Run in Jupyter, Colab or any IDE that supports inline plotting.
#
# REQUIRED LIBRARIES
# ------------------
# !pip install pandas numpy matplotlib seaborn scikit-learn wordcloud joblib

import re, string, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# 1. Load & inspect the data
files = sorted(Path(".").glob("spam-*.xlsx"))
if not files:
    raise FileNotFoundError("Missing spam-#.xlsx files — place them next to this script.")

df = pd.concat([pd.read_excel(f) for f in files], ignore_index=True)
df = (
    df.rename(columns={"v1": "label", "v2": "text"})
      .loc[:, ["label", "text"]]
      .dropna(subset=["text"])
      .reset_index(drop=True)
)
print(f"Loaded {len(df):,} rows — class counts:\n{df.label.value_counts()}\n")

# 2. Robust text-cleaning helper
def clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s{2,}", " ", text).strip()

df["clean"] = df.text.map(clean)
df["length"] = df.clean.str.len()

# 3. Quick EDA
plt.figure(figsize=(4,3))
sns.countplot(x="label", data=df)
plt.title("Class Balance"); plt.tight_layout(); plt.show()

for tgt, cmap in [("ham", "Blues"), ("spam", "Reds")]:
    words = " ".join(df.loc[df.label==tgt, "clean"])
    WordCloud(width=600, height=300, stopwords=STOPWORDS,
              background_color="white", colormap=cmap)\
              .generate(words).to_image().show()

plt.figure(figsize=(4,3))
sns.histplot(data=df, x="length", hue="label", bins=40, kde=True)
plt.title("Message Length Distribution"); plt.tight_layout(); plt.show()

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df.clean, df.label, test_size=0.20, random_state=42, stratify=df.label
)

# 5. Three fast baselines
pipelines = {
    "NB":  Pipeline([("tfidf", TfidfVectorizer(stop_words="english")),
                     ("clf", MultinomialNB())]),
    "LR":  Pipeline([("tfidf", TfidfVectorizer(stop_words="english")),
                     ("clf", LogisticRegression(max_iter=500, n_jobs=-1))]),
    "SVM": Pipeline([("tfidf", TfidfVectorizer(stop_words="english")),
                     ("clf", LinearSVC())])
}

def evaluate(model, name):
    y_pred = model.predict(X_test)
    return pd.Series({
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label="spam"),
        "Recall":    recall_score(y_test, y_pred, pos_label="spam"),
        "F1":        f1_score(y_test, y_pred, pos_label="spam")
    }, name=name)

results = pd.concat([
    evaluate(pipe.fit(X_train, y_train), name)
    for name, pipe in pipelines.items()
], axis=1).T
print("Baseline comparison:\n", results.round(4), "\n")

best_name = results.F1.idxmax()
base_pipeline = pipelines[best_name]
print(f"-> Tuning {best_name} …\n")

# 6. Hyper-parameter search (linear SVM)
param_grid = {
    "tfidf__ngram_range": [(1,1), (1,2)],
    "tfidf__min_df":      [1,3,5],
    "clf__C":             [0.5,1.0,2.0]
}
grid = GridSearchCV(base_pipeline, param_grid, cv=5,
                    scoring="f1", n_jobs=-1).fit(X_train, y_train)
best_pipeline = grid.best_estimator_
print("Best params:", grid.best_params_, "\n")

# 7. Final evaluation
y_pred = best_pipeline.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="PuBu")
plt.show()
print(classification_report(y_test, y_pred, digits=4))

# 8. Top predictive tokens (permutation importance)
tfidf = best_pipeline.named_steps["tfidf"]
clf   = best_pipeline.named_steps["clf"]
X_test_tfidf = tfidf.transform(X_test)

# FIX: convert sparse matrix to dense before calling permutation_importance
X_test_dense = X_test_tfidf.toarray()

perm = permutation_importance(
    clf, X_test_dense, y_test, n_repeats=20,
    random_state=42, n_jobs=-1
)
feature_names = tfidf.get_feature_names_out()
importances = pd.Series(perm.importances_mean, index=feature_names)\
                 .sort_values(ascending=False)[:20]

plt.figure(figsize=(6,4))
sns.barplot(x=importances.values, y=importances.index, palette="mako")
plt.title("Top-20 Predictive Tokens")
plt.xlabel("Decrease in F1 when permuted")
plt.tight_layout(); plt.show()

# 9. Persist full pipeline
joblib.dump(best_pipeline, "email_spam_detector.joblib")
print("Model saved to email_spam_detector.joblib")

# 10. Quick inference helper
def predict_spam(texts):
    """Return ham/spam predictions and decision scores."""
    if isinstance(texts, str):
        texts = [texts]
    preds  = best_pipeline.predict(texts)
    scores = best_pipeline.decision_function(texts)
    return pd.DataFrame({
        "Message": texts,
        "Prediction": preds,
        "DecisionScore": scores
    })

# Live demo
print("\nLIVE DEMO:")
print(predict_spam(
    "Congratulations! You have WON a guaranteed £1000 cash prize. Call now!"
).to_string(index=False))
