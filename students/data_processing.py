import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    """
    Load dataset from CSV file.
    """
    return pd.read_csv(filepath)


def preprocess_data(df):
    X = df.drop("num", axis=1)
    y = df["num"]

    X = pd.get_dummies(X, drop_first=True)

    return X, y


# Regression
def prepare_regression_data(df, target_column):
    """
    Prepare X and y for regression.
    """

    if target_column not in df.columns:
        raise ValueError("Target column not found.")

    y = df[target_column]

    # Drop only the target column
    X = df.drop(columns=[target_column])

    return X, y


# Classification
def prepare_classification_data(df, target_column):
    """
    Prepare X and y for classification.

    - Converts target to binary (0 = no disease, 1 = disease)
    - Drops the 'chol' column as required by tests
    """

    if target_column not in df.columns:
        if "num" in df.columns:
            target_column = "num"
        else:
            raise ValueError("No valid target column found.")

    y = df[target_column].apply(lambda x: 1 if x > 0 else 0)

    X = df.drop(columns=[target_column, "chol"], errors="ignore")

    return X, y


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Split data into train/test and apply scaling.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(set(y)) > 1 else None
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler