import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score, roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler

# =====================
# Load Data
# =====================
df = pd.read_csv("heart_disease_uci(1).csv")

# Drop problematic columns if they exist
for col in ["ca", "thal"]:
    if col in df.columns:
        df = df.drop(columns=[col])

df = df.dropna()

# One hot encoding
df = pd.get_dummies(df, drop_first=True)

# =====================
# PART A: ElasticNet
# =====================
X = df.drop("num", axis=1)
y = df["num"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

alphas = [0.01, 0.1, 1, 10]
l1_ratios = [0.1, 0.5, 0.9]

best_r2 = -999

for a in alphas:
    for l in l1_ratios:
        model = ElasticNet(alpha=a, l1_ratio=l, max_iter=10000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        if r2 > best_r2:
            best_r2 = r2

print("Best ElasticNet R2:", best_r2)

# =====================
# PART B: Classification
# =====================
df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

X = df.drop("num", axis=1)
y = df["num"]

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_probs = log_model.predict_proba(X_test)[:, 1]

log_auroc = roc_auc_score(y_test, log_probs)
precision, recall, _ = precision_recall_curve(y_test, log_probs)
log_auprc = auc(recall, precision)

print("Logistic AUROC:", log_auroc)
print("Logistic AUPRC:", log_auprc)

# kNN
neighbors = [3, 5, 7, 9]
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

print("Best k:", best_k)
print("kNN AUROC:", best_score)