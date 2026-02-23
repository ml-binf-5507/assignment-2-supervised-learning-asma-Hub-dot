from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def train_logistic_regression_grid(X, y):
    param_grid = {
        "C": [0.01, 0.1, 1, 10]
    }

    model = LogisticRegression(max_iter=1000)

    grid = GridSearchCV(model, param_grid, scoring="roc_auc")
    grid.fit(X, y)

    return grid


def train_knn_grid(X, y):
    param_grid = {
        "n_neighbors": [3, 5, 7, 9]
    }

    model = KNeighborsClassifier()

    grid = GridSearchCV(model, param_grid, scoring="roc_auc")
    grid.fit(X, y)

    return grid