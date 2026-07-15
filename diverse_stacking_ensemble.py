import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from focal_xgboost import FocalLossXGBClassifier
from minirocket_lite import MiniRocketLiteClassifier
from train import train_eval
from data_prep import split_data, pca_transform_signals, get_signals


class ColumnSubsetWrapper(BaseEstimator, ClassifierMixin):
    """Restricts an inner estimator to a fixed subset of feature columns.

    Lets heterogeneous base learners -- e.g. one that wants PCA'd tabular
    features and one that wants the raw signal -- share a single
    StackingClassifier, which otherwise passes the exact same X to every
    base estimator.
    """

    def __init__(self, estimator, columns):
        self.estimator = estimator
        self.columns = columns

    def fit(self, X, y, sample_weight=None):
        X_sub = np.asarray(X)[:, self.columns]
        self.estimator_ = clone(self.estimator)
        if sample_weight is not None:
            self.estimator_.fit(X_sub, y, sample_weight=sample_weight)
        else:
            self.estimator_.fit(X_sub, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        X_sub = np.asarray(X)[:, self.columns]
        return self.estimator_.predict_proba(X_sub)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


if __name__ == "__main__":
    data = pd.read_csv("./data/Ketamine_icp_knn_imputed_k_20_missing.csv")

    data_pca = pca_transform_signals(data, n_components=1)
    raw_signals = get_signals(data)

    # Combine PCA'd tabular features and the raw signal into one wide frame;
    # each base learner below selects only the columns it actually wants.
    combined = pd.concat(
        [data_pca.drop(columns=["label"]), raw_signals, data_pca["label"]], axis=1
    )

    feature_cols = combined.drop(columns=["label"]).columns
    raw_idx = [i for i, c in enumerate(feature_cols) if c.startswith("x") and c[1:].isdigit()]
    tabular_idx = [i for i in range(len(feature_cols)) if i not in raw_idx]

    train_data, test_data = split_data(combined, test_size=0.2, random_state=42, stratify=True)

    base_estimators = [
        ("lr", ColumnSubsetWrapper(
            LogisticRegression(max_iter=1000, penalty="l1", solver="liblinear"), tabular_idx)),
        ("rf", ColumnSubsetWrapper(
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1), tabular_idx)),
        ("focal_xgb", ColumnSubsetWrapper(
            FocalLossXGBClassifier(calibrate=False, n_estimators=100, n_jobs=1), tabular_idx)),
        ("minirocket", ColumnSubsetWrapper(
            MiniRocketLiteClassifier(), raw_idx)),
    ]

    model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=2,
        stack_method="predict_proba",
        n_jobs=1,
    )

    diverse_stacking_results = train_eval(model, train_data, test_data, kfolds=10, threshold=0.5)

    print(diverse_stacking_results)
