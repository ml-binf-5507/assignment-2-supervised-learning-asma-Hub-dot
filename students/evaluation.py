from sklearn.metrics import (
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


def calculate_r2_score(y_true, y_pred):
    return r2_score(y_true, y_pred)


def calculate_classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }


def calculate_auroc_score(y_true, y_proba):
    return roc_auc_score(y_true, y_proba)


def calculate_auprc_score(y_true, y_proba):
    return average_precision_score(y_true, y_proba)