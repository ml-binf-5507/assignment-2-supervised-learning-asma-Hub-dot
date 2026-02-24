import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def train_elasticnet_grid(X, y, l1_ratios, alphas):

    from sklearn.preprocessing import StandardScaler

    # Scale first
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
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
                "model": model
            })

    return pd.DataFrame(results)