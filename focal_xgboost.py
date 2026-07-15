import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV

from train import train_eval
from data_prep import split_data, pca_transform_signals


class FocalLossXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, focal_alpha=0.25, focal_gamma=2.0, focal_epsilon=1e-9,
                 n_estimators=300, max_depth=4, learning_rate=0.05, n_jobs=None,
                 calibrate=True, calibration_cv=3, random_state=42):
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.focal_epsilon = focal_epsilon
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs
        self.calibrate = calibrate
        self.calibration_cv = calibration_cv
        self.random_state = random_state

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-x))

    def _focal_binary_objective(self, y_true, y_pred, sample_weight=None):
        # xgboost's sklearn API calls custom objectives as (y_true, y_pred),
        # where y_pred is the raw margin (pre-sigmoid) score.
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        alpha, gamma, eps = self.focal_alpha, self.focal_gamma, self.focal_epsilon

        p = self._sigmoid(y_pred)
        p = np.clip(p, eps, 1.0 - eps)

        dldp = np.zeros_like(p)
        d2ldp2 = np.zeros_like(p)

        pos = y_true == 1
        neg = ~pos

        if np.any(pos):
            p_pos = p[pos]
            dldp[pos] = alpha * (
                gamma * np.power(1.0 - p_pos, gamma - 1.0) * np.log(p_pos)
                - np.power(1.0 - p_pos, gamma) / p_pos
            )
            d2ldp2[pos] = alpha * (
                -gamma * (gamma - 1.0) * np.power(1.0 - p_pos, gamma - 2.0) * np.log(p_pos)
                + 2.0 * gamma * np.power(1.0 - p_pos, gamma - 1.0) / p_pos
                + np.power(1.0 - p_pos, gamma) / np.square(p_pos)
            )

        if np.any(neg):
            p_neg = p[neg]
            dldp[neg] = (1.0 - alpha) * (
                -gamma * np.power(p_neg, gamma - 1.0) * np.log(1.0 - p_neg)
                + np.power(p_neg, gamma) / (1.0 - p_neg)
            )
            d2ldp2[neg] = (1.0 - alpha) * (
                -gamma * (gamma - 1.0) * np.power(p_neg, gamma - 2.0) * np.log(1.0 - p_neg)
                + 2.0 * gamma * np.power(p_neg, gamma - 1.0) / (1.0 - p_neg)
                + np.power(p_neg, gamma) / np.square(1.0 - p_neg)
            )

        dpdz = p * (1.0 - p)
        d2pdz2 = dpdz * (1.0 - 2.0 * p)

        grad = dldp * dpdz
        hess = d2ldp2 * np.square(dpdz) + dldp * d2pdz2
        hess = np.maximum(hess, eps)

        if sample_weight is not None:
            grad = grad * sample_weight
            hess = hess * sample_weight

        return grad, hess

    def fit(self, X, y, sample_weight=None):
        base = XGBClassifier(
            objective=self._focal_binary_objective,
            eval_metric="logloss",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

        if self.calibrate:
            # Focal loss distorts predicted probabilities (it's not a proper
            # loss), so calibrate them post-hoc. CalibratedClassifierCV fits
            # `base` on cv folds and calibrates on each held-out fold, so this
            # doesn't calibrate on data the model has already memorized.
            self.model_ = CalibratedClassifierCV(base, method="isotonic", cv=self.calibration_cv)
            self.model_.fit(X, y, sample_weight=sample_weight)
        else:
            base.fit(X, y, sample_weight=sample_weight)
            self.model_ = base

        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


if __name__ == "__main__":
    data = pd.read_csv("./data/Ketamine_icp_knn_imputed_k_20_missing.csv")

    data_pca = pca_transform_signals(data, n_components=6)

    train_data, test_data = split_data(data_pca, test_size=0.2, random_state=42, stratify=True)

    model = FocalLossXGBClassifier()

    focal_xgb_results = train_eval(model, train_data, test_data, kfolds=10, threshold=0.5)

    print(focal_xgb_results)
