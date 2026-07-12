import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

def compute_weights(y):
    counts = pd.Series(y).value_counts()
    n, k = len(y), len(counts)
    return {c: n / (k * cnt) for c, cnt in counts.items()}

def confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    return {"TN": int(cm[0, 0]), "FP": int(cm[0, 1]), "FN": int(cm[1, 0]), "TP": int(cm[1, 1])}

def train_eval(model, train_data, test_data, kfolds=10):
    X = train_data.drop(columns=["label"]).values
    y = train_data["label"].values
    skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=42)

    def process_fold(train_idx, val_idx):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        weights = compute_weights(y_tr)
        sw = np.array([weights[c] for c in y_tr])
        fold_model = clone(model)
        fold_model.fit(X_tr, y_tr, sample_weight=sw)
        return roc_auc_score(y_val, fold_model.predict_proba(X_val)[:, 1])

    aucs = Parallel(n_jobs=-1)(delayed(process_fold)(tr, va) for tr, va in skf.split(X, y))

    weights = compute_weights(y)
    sw = np.array([weights[c] for c in y])
    final_model = clone(model)
    final_model.fit(X, y, sample_weight=sw)

    X_test = test_data.drop(columns=["label"]).values
    y_test = test_data["label"].values
    pred = final_model.predict(X_test)
    proba = final_model.predict_proba(X_test)[:, 1]

    return {
        "aucs": np.round(aucs, 3).tolist(),
        "mean_auc": float(np.mean(aucs)),
        "test_auc": float(round(roc_auc_score(y_test, proba), 3)),
        "test_confusion_matrix": confusion_matrix(y_test, pred),
    }
