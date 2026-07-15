import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from train import train_eval
from data_prep import split_data


class MiniRocketLiteClassifier(BaseEstimator, ClassifierMixin):
    """Hand-rolled, simplified MiniRocket: random dilated convolutional
    kernels -> proportion-of-positive-values (PPV) features -> logistic
    regression. Not the exact MiniRocket algorithm (which uses a fixed
    combinatorial {-1, 2} kernel-weight set for determinism/speed) -- this
    uses random Gaussian kernels instead -- but the same core idea: many
    cheap random convolutions summarized by PPV, fed to a linear classifier.
    Avoids the aeon/numba dependency chain entirely.
    """

    def __init__(self, num_kernels=84, num_biases=8, kernel_lengths=(7, 9, 11),
                 bias_fit_subsample=2000, C=1.0, random_state=42):
        self.num_kernels = num_kernels
        self.num_biases = num_biases
        self.kernel_lengths = kernel_lengths
        self.bias_fit_subsample = bias_fit_subsample
        self.C = C
        self.random_state = random_state

    def _make_kernels(self, n_features, rng):
        kernels = []
        for _ in range(self.num_kernels):
            length = min(int(rng.choice(self.kernel_lengths)), n_features)
            weights = rng.normal(size=length).astype(np.float32)
            weights -= weights.mean()

            max_dilation_exp = int(np.log2(max(1.0, (n_features - 1) / max(1, length - 1))))
            dilation = 2 ** rng.randint(0, max_dilation_exp + 1)
            kernels.append((length, weights, dilation))
        return kernels

    @staticmethod
    def _convolve(X, length, weights, dilation):
        n_features = X.shape[1]
        pad_total = (length - 1) * dilation
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        X_padded = np.pad(X, ((0, 0), (pad_left, pad_right)))

        conv_out = np.zeros_like(X, dtype=np.float32)
        for k in range(length):
            offset = k * dilation
            conv_out += weights[k] * X_padded[:, offset:offset + n_features]
        return conv_out

    def _transform(self, X):
        features = np.empty((X.shape[0], self.num_kernels * self.num_biases), dtype=np.float32)
        col = 0
        for (length, weights, dilation), biases in zip(self.kernels_, self.biases_):
            conv_out = self._convolve(X, length, weights, dilation)
            for bias in biases:
                features[:, col] = (conv_out > bias).mean(axis=1)
                col += 1
        return features

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        rng = np.random.RandomState(self.random_state)

        self.kernels_ = self._make_kernels(X.shape[1], rng)

        # Derive each kernel's biases from quantiles of its own convolution
        # output on a subsample of training data (MiniRocket's data-driven
        # bias scheme), rather than fixed/arbitrary bias values.
        sub_idx = rng.choice(len(X), size=min(self.bias_fit_subsample, len(X)), replace=False)
        X_sub = X[sub_idx]

        self.biases_ = []
        quantiles = np.linspace(0.1, 0.9, self.num_biases)
        for length, weights, dilation in self.kernels_:
            conv_out = self._convolve(X_sub, length, weights, dilation)
            self.biases_.append(np.quantile(conv_out, quantiles).astype(np.float32))

        Xt = self._transform(X)
        self.scaler_ = StandardScaler()
        Xt = self.scaler_.fit_transform(Xt)

        self.clf_ = LogisticRegression(max_iter=2000, C=self.C)
        self.clf_.fit(Xt, y, sample_weight=sample_weight)
        self.classes_ = self.clf_.classes_
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xt = self._transform(X)
        Xt = self.scaler_.transform(Xt)
        return self.clf_.predict_proba(Xt)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


if __name__ == "__main__":
    data = pd.read_csv("./data/Ketamine_icp_knn_imputed_k_20_missing.csv")

    train_data, test_data = split_data(data, test_size=0.2, random_state=42, stratify=True)

    model = MiniRocketLiteClassifier()

    minirocket_results = train_eval(model, train_data, test_data, kfolds=10, threshold=0.5)

    print(minirocket_results)
