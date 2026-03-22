import pandas as pd


class Dataset:
    def __init__(self, file_path, impute = False, **kwargs):

        self.file_path = file_path
        self.data = self.clean_data(impute, **kwargs)


    def clean_data(self, impute=False, **kwargs):
        print(f"Loading data from {self.file_path}...")
        data = pd.read_csv(self.file_path).drop('id', axis=1)
        data['label'] = data['label'].apply(lambda x: 1 if x == 0 else 0)

        if impute:
            k = kwargs.get('k', 5)
            print(f"Imputing missing values using KNN Imputer with k = {k} ...")
            from sklearn.impute import KNNImputer
            data['has_missing'] = data.isnull().any(axis=1).astype(int)
            imputer = KNNImputer(n_neighbors=k)
            data_imputed = imputer.fit_transform(data)
            data = pd.DataFrame(data_imputed, columns=data.columns)
            
        else:
            print("Dropping rows with missing values...")
            data = data.dropna()

        return data


    def split_data(self, data=None, test_size=0.15, random_state=42, stratify_col='label'):
        if data is None:
            data = self.data
        from sklearn.model_selection import train_test_split
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if stratify_col else None
        )
        return X_train, X_test, y_train, y_test

    def apply_pca(self, features, variance_threshold=0.95):
        from sklearn.decomposition import PCA
        print(f"Applying PCA to retain {variance_threshold*100}% variance...")
        pca = PCA(n_components=variance_threshold)
        features_pca = pca.fit_transform(features)
        print(f"Original number of features: {features.shape[1]}, Reduced number of features: {features_pca.shape[1]}")
        return features_pca