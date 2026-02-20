import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    r2_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc
)
from sklearn.preprocessing import StandardScaler

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("heart_disease_uci(1).csv")

# ===============================
# DATA CLEANING
# ===============================

# Drop columns with too many missing values
df = df.drop(columns=["ca", "thal"])

# Drop remaining missing rows
df = df.dropna()

# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# ===============================
# ===============================
# PART A — REGRESSION (ElasticNet)
# ===============================
# ===============================

X = df.drop("num", axis=1)
y = df["num"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

alphas = [0.01, 0.1, 1, 10]
l1_ratios = [0.1, 0.5, 0.9]

results = []

for a in alphas:
    for l in l1_ratios:
        model = ElasticNet(alpha=a, l1_ratio=l, max_iter=10000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        results.append([a, l, r2])

results_df = pd.DataFrame(results, columns=["alpha", "l1_ratio", "R2"])
pivot = results_df.pivot(index="l1_ratio", columns="alpha", values="R2")

plt.figure(figsize=(8,6))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
plt.title("ElasticNet R2 Heatmap")
plt.xlabel("Alpha")
plt.ylabel("L1 Ratio")
plt.tight_layout()
plt.show()

# ===============================
# ===============================
# PART B — CLASSIFICATION
# ===============================
# ===============================

# Convert to binary
df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

X = df.drop("num", axis=1)
y = df["num"]

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# Logistic Regression
# -------------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_probs = log_model.predict_proba(X_test)[:, 1]

log_auroc = roc_auc_score(y_test, log_probs)

precision, recall, _ = precision_recall_curve(y_test, log_probs)
log_auprc = auc(recall, precision)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, log_probs)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"Logistic (AUC = {log_auroc:.2f})")
plt.plot([0,1],[0,1], linestyle="--", label="Random Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()

# PR Curve
baseline = sum(y_test)/len(y_test)

plt.figure(figsize=(6,5))
plt.plot(recall, precision, label=f"Logistic (AUPRC = {log_auprc:.2f})")
plt.hlines(baseline, 0, 1, linestyles="--", label="Random Chance")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Logistic Regression PR Curve")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# kNN
# -------------------------------
neighbors = [3,5,7,9]
best_k = None
best_score = 0

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    probs = knn.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, probs)

    if score > best_score:
        best_score = score
        best_k = k

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
knn_probs = knn.predict_proba(X_test)[:, 1]

knn_auroc = roc_auc_score(y_test, knn_probs)
precision, recall, _ = precision_recall_curve(y_test, knn_probs)
knn_auprc = auc(recall, precision)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, knn_probs)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"kNN (AUC = {knn_auroc:.2f})")
plt.plot([0,1],[0,1], linestyle="--", label="Random Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("kNN ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()

# PR Curve
plt.figure(figsize=(6,5))
plt.plot(recall, precision, label=f"kNN (AUPRC = {knn_auprc:.2f})")
plt.hlines(baseline, 0, 1, linestyles="--", label="Random Chance")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("kNN PR Curve")
plt.legend()
plt.tight_layout()
plt.show()

print("\nFinal Metrics:")
print("Logistic AUROC:", log_auroc)
print("Logistic AUPRC:", log_auprc)
print("kNN AUROC:", knn_auroc)
print("kNN AUPRC:", knn_auprc)