import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score

def compute_weights(y):
    counts = pd.Series(y).value_counts()
    n, k = len(y), len(counts)
    return {c: n / (k * cnt) for c, cnt in counts.items()}


def confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    # labels=[0, 1] pinned explicitly: without it, sklearn infers the label
    # set from what's actually present, so a degenerate all-one-class
    # prediction returns a 1x1 matrix instead of 2x2 and the indexing
    # below crashes.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {"TN": int(cm[0, 0]), "FP": int(cm[0, 1]), "FN": int(cm[1, 0]), "TP": int(cm[1, 1])}

def train_eval(model, train_data, test_data, kfolds=10, threshold=0.5):
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
    train_start_time = pd.Timestamp.now()
    final_model.fit(X, y, sample_weight=sw)
    train_end_time = pd.Timestamp.now()

    X_test = test_data.drop(columns=["label"]).values
    y_test = test_data["label"].values
    test_start_time = pd.Timestamp.now()
    proba = final_model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    test_end_time = pd.Timestamp.now()

    return {
        "aucs": np.round(aucs, 3).tolist(),
        "mean_auc": float(np.mean(aucs)),
        "train_time(sec)": (train_end_time - train_start_time).total_seconds(),
        "test_auc": float(round(roc_auc_score(y_test, proba), 3)),
        "test_confusion_matrix": confusion_matrix(y_test, pred),
        "f1_score": f1_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "specificity": confusion_matrix(y_test, pred)["TN"] / (confusion_matrix(y_test, pred)["TN"] + confusion_matrix(y_test, pred)["FP"]),
        "sensitivity": confusion_matrix(y_test, pred)["TP"] / (confusion_matrix(y_test, pred)["TP"] + confusion_matrix(y_test, pred)["FN"]),
        "accuracy": (confusion_matrix(y_test, pred)["TP"] + confusion_matrix(y_test, pred)["TN"]) / len(y_test),
        "pred_time(sec)": (test_end_time - test_start_time).total_seconds()
    }
