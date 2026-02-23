from students.data_processing import (
    load_data,
    prepare_regression_data,
    prepare_classification_data
)
from students.regression import run_elasticnet
from students.classification import train_logistic, train_knn
from students.evaluation import evaluate_model


def main():

    df = load_data()

    # Regression
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = prepare_regression_data(df)
    run_elasticnet(X_train_reg, X_test_reg, y_train_reg, y_test_reg)

    # Classification
    X_train, X_test, y_train, y_test = prepare_classification_data(df)

    log_model = train_logistic(X_train, y_train)
    evaluate_model(log_model, X_test, y_test, "logistic")

    knn_model = train_knn(X_train, y_train, n_neighbors=5)
    evaluate_model(knn_model, X_test, y_test, "knn")


if __name__ == "__main__":
    main()