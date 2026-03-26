import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(
        self,
        file_path,
        impute=False,
        apply_pca=False,
        signals_variance_threshold=0.95,
        stats_variance_threshold=0.95,
        split_size=0.15,
        random_state=42,
        stratify=True,
        **kwargs
    ):
        self.file_path = file_path
        self.impute = impute
        self.apply_pca_flag = apply_pca
        self.signals_variance_threshold = signals_variance_threshold
        self.stats_variance_threshold = stats_variance_threshold
        self.split_size = split_size
        self.random_state = random_state
        self.stratify = stratify
        self.kwargs = kwargs

        self.data = self.clean_data(impute=impute, **kwargs)

        self.label = self.data["label"]
        self.signals = self.data.drop("label", axis=1).iloc[:, :1000]
        self.stats = self.data.drop("label", axis=1).iloc[:, 1000:]

        self.signals_pca = None
        self.stats_pca = None
        self.df_signals_pca = None
        self.df_stats_pca = None

        if self.apply_pca_flag:
            if self.signals.shape[1] > 0:
                self.signals_pca = self.apply_pca(
                    self.signals,
                    variance_threshold=self.signals_variance_threshold
                )
                self.df_signals_pca = pd.DataFrame(
                    self.signals_pca,
                    index=self.signals.index
                )
                self.df_signals_pca.columns = self.df_signals_pca.columns.astype(str)

            if self.stats.shape[1] > 0:
                self.stats_pca = self.apply_pca(
                    self.stats,
                    variance_threshold=self.stats_variance_threshold
                )
                self.df_stats_pca = pd.DataFrame(
                    self.stats_pca,
                    index=self.stats.index
                )
                self.df_stats_pca.columns = self.df_stats_pca.columns.astype(str)

        self.final_data = self._build_final_data()
        self.X = self.final_data.drop("label", axis=1).astype(np.float32)
        self.y = self.final_data["label"]

        self.X_train, self.X_test, self.y_train, self.y_test = self.get_split_data()

    def clean_data(self, impute=False, **kwargs):
        print(f"Loading data from {self.file_path}...")
        data = pd.read_csv(self.file_path).drop("id", axis=1)
        data["label"] = data["label"].apply(lambda x: 1 if x == 0 else 0)

        if impute:
            k = kwargs.get("k", 5)
            print(f"Imputing missing values using KNN Imputer with k = {k} ...")
            from sklearn.impute import KNNImputer
            data["has_missing"] = data.isnull().any(axis=1).astype(int)
            imputer = KNNImputer(n_neighbors=k)
            data_imputed = imputer.fit_transform(data)
            data = pd.DataFrame(data_imputed, columns=data.columns)
        else:
            print("Dropping rows with missing values...")
            data = data.dropna()

        return data

    def apply_pca(self, features, variance_threshold=0.95):
        from sklearn.decomposition import PCA
        print(f"Applying PCA to retain {variance_threshold * 100}% variance...")
        pca = PCA(n_components=variance_threshold)
        features_pca = pca.fit_transform(features)
        print(
            f"Original number of features: {features.shape[1]}, "
            f"Reduced number of features: {features_pca.shape[1]}"
        )
        return features_pca

    def _build_final_data(self):
        parts = []

        if self.apply_pca_flag:
            if self.df_signals_pca is not None:
                parts.append(self.df_signals_pca)
            if self.df_stats_pca is not None:
                parts.append(self.df_stats_pca)
        else:
            if self.signals.shape[1] > 0:
                parts.append(self.signals)
            if self.stats.shape[1] > 0:
                parts.append(self.stats)

        parts.append(self.label)

        data = pd.concat(parts, axis=1)
        data.columns = data.columns.astype(str)
        return data

    def get_split_data(
        self,
        X=None,
        y=None,
        test_size=None,
        random_state=None,
        stratify=None
    ):
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        if test_size is None:
            test_size = self.split_size
        if random_state is None:
            random_state = self.random_state
        if stratify is None:
            stratify = self.stratify

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify else None
        )
        return X_train, X_test, y_train, y_test

    def split_data(
        self,
        data=None,
        test_size=None,
        random_state=None,
        stratify_col="label"
    ):
        if data is None:
            X = self.X
            y = self.y
        else:
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]

        if test_size is None:
            test_size = self.split_size
        if random_state is None:
            random_state = self.random_state

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify_col else None
        )
        return X_train, X_test, y_train, y_test