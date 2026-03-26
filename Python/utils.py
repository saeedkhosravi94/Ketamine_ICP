import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight


def set_device(device='GPU'):
    if device == 'GPU':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                device_name = '/GPU:0'
                print(f"Using GPU (MPS): {gpus}")
            except Exception as e:
                device_name = '/CPU:0'
                print(f"GPU found but could not configure it: {e}")
                print("Falling back to CPU")
        else:
            device_name = '/CPU:0'
            print("No GPU found. Using CPU")
        tf.config.optimizer.set_jit(False)
        return device_name

    print("Using CPU")
    return '/CPU:0'


def get_scaler(name='standard'):
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler(),
        None: None
    }
    if name not in scalers:
        raise ValueError(f"Unknown scaler: {name}")
    return scalers[name]


def scale_train_test(X_train, X_test, scaler_name='standard'):
    scaler = get_scaler(scaler_name)

        
    if scaler is None:
        return X_train.astype(np.float32), X_test.astype(np.float32), None
    

    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    return X_train_s, X_test_s, scaler


def reshape_for_dl(X_train, X_test):
    X_train_dl = X_train.reshape(-1, X_train.shape[1], 1).astype(np.float32)
    X_test_dl = X_test.reshape(-1, X_test.shape[1], 1).astype(np.float32)
    return X_train_dl, X_test_dl


def get_classification_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    cw_dict = {cls: w for cls, w in zip(classes, weights)}

    class_counts = {cls: (y == cls).sum() for cls in classes}
    minority_class = min(class_counts, key=class_counts.get)
    majority_class = max(class_counts, key=class_counts.get)

    minority_weight = weights[np.where(classes == minority_class)[0][0]]
    majority_weight = weights[np.where(classes == majority_class)[0][0]]
    w_ratio = minority_weight / majority_weight

    return {
        "classes": classes,
        "weights": weights,
        "class_weight_dict": cw_dict,
        "minority_class": minority_class,
        "majority_class": majority_class,
        "w_ratio": w_ratio
    }


def plot_dolan_more(perf_dict, title='Performance Profile (Dolan-Moré Plot)'):
    df_perf = 1 - pd.DataFrame(perf_dict)
    r_ps = df_perf.div(df_perf.min(axis=1), axis=0)
    taus = np.linspace(1, r_ps.values.max(), 100)

    plt.figure(figsize=(12, 7))
    for col in r_ps.columns:
        rho = [(r_ps[col] <= t).mean() for t in taus]
        plt.step(taus, rho, label=col, where='post')

    plt.title(title)
    plt.xlabel('tau')
    plt.ylabel('rho')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_roc_curves(roc_entries, title='ROC Curve Comparison'):
    plt.figure(figsize=(12, 8))

    for entry in roc_entries:
        plt.plot(
            1 - entry['fpr'],
            entry['tpr'],
            label=f"{entry['name']} (AUC={entry['auc']:.2f})",
            linestyle=entry.get('linestyle', '-')
        )

    plt.plot([1, 0], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('Specificity')
    plt.ylabel('Sensitivity')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.show()


def save_results_to_csv(results_table, filepath='results.csv'):
    results_df = pd.DataFrame(results_table).sort_values('Test_AUC', ascending=False)
    results_df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")
    return results_df