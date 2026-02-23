import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def train_elasticnet_grid(X, y, l1_ratios, alphas):
    """
    Train ElasticNet over a grid of l1_ratios and alphas.

    Splits data internally (80/20 split).

    Returns DataFrame with columns:
    ['l1_ratio', 'alpha', 'r2_score', 'model']
    """

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    results = []

    for l1_ratio in l1_ratios:
        for alpha in alphas:

            model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                random_state=42,
                max_iter=10000
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)

            results.append({
                "l1_ratio": l1_ratio,
                "alpha": alpha,
                "r2_score": r2,
                "model": model   # ⭐ THIS WAS MISSING
            })

    return pd.DataFrame(results)