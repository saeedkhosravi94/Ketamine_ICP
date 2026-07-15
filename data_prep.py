import numpy as np
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



def normalize_signals(data, eps=1e-8):
    # Per-sample z-score: each signal's own 1000 points are centered/scaled
    # using that row's own mean and std, independent of every other sample
    # (unlike StandardScaler, which normalizes per-column across samples).
    # Removes baseline offset/amplitude differences between recordings that
    # aren't related to risk, before PCA/MiniRocket/etc. ever see the signal.
    # Unsupervised and row-independent, so safe to call before or after
    # split_data, and before pca_transform_signals and friends.
    signals, stats, labels = split_features_labels(data)
    row_mean = signals.mean(axis=1)
    row_std = signals.std(axis=1)
    normalized = signals.sub(row_mean, axis=0).div(row_std + eps, axis=0)
    data = concat_features_labels(normalized, stats, labels)
    return data


def denoise_signals(data, window_length=11, polyorder=3):
    # Savitzky-Golay smoothing, applied along each signal's own 1000 points
    # independently. Fits a local polynomial per window rather than just
    # averaging, so it smooths sample-level noise with less peak-shape
    # distortion than a plain moving average -- intended to help
    # convolution-based feature extraction (MiniRocket) find stabler
    # patterns instead of fitting noise. Unsupervised and row-independent,
    # so safe to call before or after split_data.
    from scipy.signal import savgol_filter
    signals, stats, labels = split_features_labels(data)
    smoothed = savgol_filter(signals.values, window_length=window_length, polyorder=polyorder, axis=1)
    smoothed = pd.DataFrame(smoothed, columns=signals.columns, index=signals.index)
    data = concat_features_labels(smoothed, stats, labels)
    return data


def scale_stats(data):
    # Standardize the engineered stats features (per-column z-score across
    # samples) -- unlike normalize_signals, which is row-wise on the raw
    # signal, this is standard per-feature scaling. The ~33 stats span a
    # ~60,000x range in std (e.g. acf_lag_dom vs acf_integral_time), which
    # distorts L1-penalized linear models (a uniform penalty strength
    # effectively punishes small-scale features harder); irrelevant for
    # tree-based models (RF/XGB), which split on raw values regardless of
    # scale. Unsupervised, so safe to call before or after split_data, same
    # convention as pca_transform_signals.
    from sklearn.preprocessing import StandardScaler
    signals, stats, labels = split_features_labels(data)
    scaler = StandardScaler()
    stats_scaled = pd.DataFrame(
        scaler.fit_transform(stats), columns=stats.columns, index=stats.index
    )
    data = concat_features_labels(signals, stats_scaled, labels)
    return data


def catch22_transform_signals(data):
    # Extract the catch22 feature set (Lubba et al. 2019) from the raw
    # signal: 22 canonical time-series characteristics (nonlinear
    # autocorrelation, value-distribution shape, fluctuation scaling, etc.)
    # selected from ~4791 candidate features specifically for cross-domain
    # classification power. Mathematically distinct from both the PCA/stats
    # view (linear/summary-statistic) and MiniRocket (random convolution) --
    # a third, differently-shaped view of the same signal. Unsupervised, so
    # safe to call before or after split_data, same convention as
    # pca_transform_signals.
    import pycatch22
    signals, stats, labels = split_features_labels(data)

    rows = signals.values.tolist()
    feature_names = None
    catch22_values = []
    for row in rows:
        result = pycatch22.catch22_all(row)
        if feature_names is None:
            feature_names = result["names"]
        catch22_values.append(result["values"])

    catch22_df = pd.DataFrame(
        catch22_values, columns=[f"catch22_{n}" for n in feature_names], index=signals.index
    )
    data = concat_features_labels(catch22_df, stats, labels)
    return data


def remove_majority_outliers(train_data, contamination=0.05, random_state=42):
    # Drop outliers within the majority (normal, label=1) class ONLY, using
    # IsolationForest on the non-signal (stats/PCA) features. Deliberately
    # never touches the minority (risky, label=0) class: a risky signal is
    # expected to look anomalous relative to normal ones, so an unsupervised
    # outlier detector applied there would preferentially flag exactly the
    # signal we're trying to classify, shrinking an already-scarce class.
    # This changes which SAMPLES are trained on (not just how features are
    # represented), so -- unlike the transform functions above -- only ever
    # call this on the training split, never on test data: dropping "hard"
    # rows from the test set would make evaluation easier in a way that
    # doesn't reflect real deployment, not just clean up training noise.
    from sklearn.ensemble import IsolationForest
    signals, stats, labels = split_features_labels(train_data)

    majority_mask = (labels == 1).values
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    is_inlier = iso.fit_predict(stats[majority_mask]) == 1

    keep = np.ones(len(train_data), dtype=bool)
    keep[majority_mask] = is_inlier
    return train_data.iloc[keep].reset_index(drop=True)


def clip_majority_stats_outliers(train_data, lower_quantile=0.01, upper_quantile=0.99):
    # Winsorize extreme per-feature values in the stats columns, within the
    # majority (normal, label=1) class ONLY -- caps implausible values (e.g.
    # time_range_ratio ranging to +4304, likely a divide-by-near-zero
    # artifact) at a percentile bound computed from that class, without
    # dropping the row entirely (unlike remove_majority_outliers) and
    # without touching minority (risky) rows' values at all, since an
    # extreme value there may be the actual risk signal rather than noise.
    # Train-only, same reasoning as remove_majority_outliers.
    signals, stats, labels = split_features_labels(train_data)
    majority_mask = (labels == 1).values

    majority_stats = stats.iloc[majority_mask]
    lower = majority_stats.quantile(lower_quantile)
    upper = majority_stats.quantile(upper_quantile)

    stats_clipped = stats.copy()
    stats_clipped.iloc[majority_mask] = majority_stats.clip(lower=lower, upper=upper, axis=1)

    return concat_features_labels(signals, stats_clipped, labels)


def pca_transform_signals(data, n_components=10):
    # n_components may be a float in (0, 1) -- e.g. 0.95 -- in which case
    # sklearn picks however many components are needed to explain that
    # fraction of variance, and the actual count isn't known until after
    # fitting; hence naming columns off signals_pca.shape[1], not the raw
    # n_components argument.
    signals, stats, labels = split_features_labels(data)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    signals_pca = pca.fit_transform(signals)
    signals_pca = pd.DataFrame(signals_pca, columns=[f"pca_{i+1}" for i in range(signals_pca.shape[1])])
    data = concat_features_labels(signals_pca, stats, labels)
    return data


def spca_transform_signals(data, n_components=10, alpha=1.0, random_state=42):
    # Sparse PCA: like PCA, unsupervised (doesn't use the label), so it's
    # safe to call before split_data the same way pca_transform_signals is.
    # Each component loads on fewer of the original signal points (sparser,
    # more interpretable loadings), at the cost of being slower to fit.
    signals, stats, labels = split_features_labels(data)
    from sklearn.decomposition import SparsePCA
    spca = SparsePCA(n_components=n_components, alpha=alpha, random_state=random_state)
    signals_spca = spca.fit_transform(signals)
    signals_spca = pd.DataFrame(signals_spca, columns=[f"spca_{i+1}" for i in range(n_components)])
    data = concat_features_labels(signals_spca, stats, labels)
    return data


def pls_transform_signals(train_data, test_data, n_components=10):
    # Partial Least Squares: unlike PCA/SparsePCA, this is supervised -- it
    # uses the label to find signal directions that covary with class, which
    # PCA has no mechanism to prefer. Because it uses the label during
    # fitting, it must be fit on the training split ONLY and then applied to
    # the test split, or test labels leak into the transformed features.
    # Call this AFTER split_data, not before, unlike pca_transform_signals:
    #   train_data, test_data = split_data(data, ...)
    #   train_data, test_data = pls_transform_signals(train_data, test_data)
    from sklearn.cross_decomposition import PLSRegression

    train_signals, train_stats, train_labels = split_features_labels(train_data)
    test_signals, test_stats, test_labels = split_features_labels(test_data)

    pls = PLSRegression(n_components=n_components)
    pls.fit(train_signals, train_labels)

    cols = [f"pls_{i+1}" for i in range(n_components)]
    train_signals_pls = pd.DataFrame(
        pls.transform(train_signals), columns=cols, index=train_signals.index
    )
    test_signals_pls = pd.DataFrame(
        pls.transform(test_signals), columns=cols, index=test_signals.index
    )

    train_data = concat_features_labels(train_signals_pls, train_stats, train_labels)
    test_data = concat_features_labels(test_signals_pls, test_stats, test_labels)
    return train_data, test_data