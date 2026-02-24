import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from students.data_processing import (
    load_heart_disease_data,
    prepare_regression_data,
    prepare_classification_data,
    split_and_scale
)

from students.regression import train_elasticnet_grid
from students.classification import train_logistic_regression_grid, train_knn_grid
from students.evaluation import calculate_auroc_score, calculate_auprc_score

from sklearn.metrics import roc_curve, precision_recall_curve



# LOAD DATA

df = load_heart_disease_data("data/heart.csv")


# ELASTICNET REGRESSION 

X_reg, y_reg = prepare_regression_data(df, target_column="chol")

l1_ratios = [0.3, 0.5, 0.7]
alphas = [0.01, 0.1, 1.0]

results = train_elasticnet_grid(X_reg, y_reg, l1_ratios, alphas)

# Heatmap
heatmap_data = results.pivot(index="l1_ratio", columns="alpha", values="r2_score")

plt.figure(figsize=(8, 6))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".3f",
    cmap="coolwarm"
)
plt.title("ElasticNet R² Heatmap (Predicting Cholesterol)")
plt.xlabel("Alpha")
plt.ylabel("L1 Ratio")
plt.tight_layout()
plt.savefig("elasticnet_heatmap.png")
plt.show()

# Get best regression parameters
best_row = results.loc[results["r2_score"].idxmax()]
print("Best ElasticNet l1_ratio:", best_row["l1_ratio"])
print("Best ElasticNet alpha:", best_row["alpha"])
print("Best ElasticNet R2:", best_row["r2_score"])


# CLASSIFICATION


X_clf, y_clf = prepare_classification_data(df, target_column="num")
X_train, X_test, y_train, y_test, scaler = split_and_scale(X_clf, y_clf)


# Logistic Regression

log_grid = train_logistic_regression_grid(X_train, y_train)
log_model = log_grid.best_estimator_

log_probs = log_model.predict_proba(X_test)[:, 1]

log_auc = calculate_auroc_score(y_test, log_probs)
log_ap = calculate_auprc_score(y_test, log_probs)

print("Best Logistic C:", log_grid.best_params_)
print("Logistic AUROC:", log_auc)
print("Logistic AUPRC:", log_ap)


# kNN

knn_grid = train_knn_grid(X_train, y_train)
knn_model = knn_grid.best_estimator_

knn_probs = knn_model.predict_proba(X_test)[:, 1]

knn_auc = calculate_auroc_score(y_test, knn_probs)
knn_ap = calculate_auprc_score(y_test, knn_probs)

print("Best kNN Params:", knn_grid.best_params_)
print("kNN AUROC:", knn_auc)
print("kNN AUPRC:", knn_ap)


# ROC CURVE (Comparison)


fpr_log, tpr_log, _ = roc_curve(y_test, log_probs)
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_probs)

plt.figure()
plt.plot(fpr_log, tpr_log, label=f"Logistic (AUC = {log_auc:.3f})")
plt.plot(fpr_knn, tpr_knn, label=f"kNN (AUC = {knn_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()



# PR CURVE (Comparison)


prec_log, rec_log, _ = precision_recall_curve(y_test, log_probs)
prec_knn, rec_knn, _ = precision_recall_curve(y_test, knn_probs)

plt.figure()
plt.plot(rec_log, prec_log, label=f"Logistic (AUPRC = {log_ap:.3f})")
plt.plot(rec_knn, prec_knn, label=f"kNN (AUPRC = {knn_ap:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("pr_curve.png")
plt.show()