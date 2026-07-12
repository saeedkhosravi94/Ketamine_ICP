import pandas as pd

def read_data(filename):
    data = pd.read_csv(filename)

    data = data.drop(columns=["id"])
    data["label"] = data["label"].replace({0: 1, 1: 0})
    return data

def get_signals(data):
    # get features x0 to x999
    signals = data.filter(regex="^x[0-9]+$")
    return signals

def get_labels(data):
    labels = data["label"]
    return labels

def get_stats(data):
    stats = data.filter(regex="^(?!x[0-9]+$).*")
    stats = stats.drop(columns=["label"])

    return stats

def get_features(data):
    features = data.drop(columns=["label"])
    return features

def split_features_labels(data):
    signals = get_signals(data)
    stats = get_stats(data)
    labels = get_labels(data)
    return signals, stats, labels

def concat_features_labels(signals, stats, labels):
    data = pd.concat([signals, stats, labels], axis=1)
    return data

def fill_missing_values(data, method=None, missing=False, k=5):

    
    if method == "knn_impute":
        from sklearn.impute import KNNImputer
        if missing:
            # add a missing value indicator column
            data["missing"] = data.isnull().any(axis=1).astype(int)
            
        imputer = KNNImputer(n_neighbors=k)
        data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        
    else:
        data = data.fillna(data.mean())

    return data

def save_data(data, filename):
    data.to_csv(filename, index=False)


def split_data(data, test_size=0.2, random_state=42, stratify=True):
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state, stratify=data["label"] if stratify else None)
    return train_data, test_data



def pca_transform_signals(data, n_components=10):
    signals, stats, labels = split_features_labels(data)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    signals_pca = pca.fit_transform(signals)
    signals_pca = pd.DataFrame(signals_pca, columns=[f"pca_{i+1}" for i in range(n_components)])
    data = concat_features_labels(signals_pca, stats, labels)
    return data