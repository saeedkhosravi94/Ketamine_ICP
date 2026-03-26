from copy import deepcopy
import time
import numpy as np
import tensorflow as tf

from tqdm.auto import tqdm
from tensorflow.keras.callbacks import Callback

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout,
    Add, LSTM, Bidirectional, GlobalAveragePooling1D,
    MultiHeadAttention, LayerNormalization
)


class TqdmKerasCallback(Callback):
    def __init__(self, epochs, desc, colour="cyan"):
        super().__init__()
        self.epochs = epochs
        self.desc = desc
        self.colour = colour
        self.pbar = None

    def on_train_begin(self, logs=None):
        self.pbar = tqdm(
            total=self.epochs,
            desc=self.desc,
            colour=self.colour,
            leave=False,
            dynamic_ncols=True
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        postfix = {}
        if "loss" in logs:
            postfix["loss"] = f"{logs['loss']:.4f}"
        if "val_loss" in logs:
            postfix["val_loss"] = f"{logs['val_loss']:.4f}"
        self.pbar.set_postfix(postfix)
        self.pbar.update(1)

    def on_train_end(self, logs=None):
        if self.pbar is not None:
            self.pbar.close()


class ModelFactory:
    def __init__(self, random_state=42, show_progress=True):
        self.random_state = random_state
        self.show_progress = show_progress

    def _make_tqdm(self, iterable=None, total=None, desc="", colour="blue", leave=True):
        if not self.show_progress:
            return iterable if iterable is not None else range(total)
        return tqdm(
            iterable=iterable,
            total=total,
            desc=desc,
            colour=colour,
            leave=leave,
            dynamic_ncols=True
        )

    def build_simple_cnn(self, shape, conv_filters=32, kernel_size=3, dense_units=64, dropout=0.2):
        return Sequential([
            Input(shape=shape),
            Conv1D(conv_filters, kernel_size, activation='relu'),
            MaxPooling1D(2),
            Flatten(),
            Dense(dense_units, activation='relu'),
            Dropout(dropout),
            Dense(1, activation='sigmoid')
        ])

    def build_resnet(self, shape, filters=32, kernel_size=3):
        i = Input(shape=shape)
        x = Conv1D(filters, kernel_size, padding='same', activation='relu')(i)
        s = x
        x = Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
        x = Conv1D(filters, kernel_size, padding='same')(x)
        x = Add()([x, s])
        x = tf.keras.layers.Activation('relu')(x)
        x = GlobalAveragePooling1D()(x)
        o = Dense(1, activation='sigmoid')(x)
        return Model(i, o)

    def build_cnn_lstm(self, shape, conv_filters=64, kernel_size=3, lstm_units=32, dropout=0.3):
        i = Input(shape=shape)
        x = Conv1D(conv_filters, kernel_size, activation='relu')(i)
        x = MaxPooling1D(2)(x)
        x = Bidirectional(LSTM(lstm_units))(x)
        x = Dropout(dropout)(x)
        o = Dense(1, activation='sigmoid')(x)
        return Model(i, o)

    def build_dilated_cnn(self, shape, filters=32, kernel_size=3):
        i = Input(shape=shape)
        x = Conv1D(filters, kernel_size, padding='same', dilation_rate=1, activation='relu')(i)
        x = Conv1D(filters, kernel_size, padding='same', dilation_rate=2, activation='relu')(x)
        x = GlobalAveragePooling1D()(x)
        o = Dense(1, activation='sigmoid')(x)
        return Model(i, o)

    def build_attn_cnn(self, shape, filters=64, kernel_size=3, num_heads=2, key_dim=64):
        i = Input(shape=shape)
        x = Conv1D(filters, kernel_size, padding='same', activation='relu')(i)
        a = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
        x = LayerNormalization()(Add()([x, a]))
        x = GlobalAveragePooling1D()(x)
        o = Dense(1, activation='sigmoid')(x)
        return Model(i, o)

    def get_sklearn_registry(self, w_ratio=1.0):
        return {
            "logreg": {
                "display_name": "Logistic Regression",
                "estimator": LogisticRegression(
                    class_weight='balanced',
                    max_iter=5000,
                    random_state=self.random_state
                )
            },
            "rf": {
                "display_name": "Random Forest",
                "estimator": RandomForestClassifier(
                    class_weight='balanced',
                    random_state=self.random_state,
                    n_jobs=-1
                )
            },
            "xgb": {
                "display_name": "XGBoost",
                "estimator": XGBClassifier(
                    scale_pos_weight=w_ratio,
                    eval_metric='logloss',
                    random_state=self.random_state
                )
            },
            "lgbm": {
                "display_name": "LightGBM",
                "estimator": LGBMClassifier(
                    class_weight='balanced',
                    verbosity=-1,
                    random_state=self.random_state
                )
            },
            "catboost": {
                "display_name": "CatBoost",
                "estimator": CatBoostClassifier(
                    auto_class_weights='Balanced',
                    verbose=0,
                    random_state=self.random_state
                )
            },
            "lda": {
                "display_name": "LDA",
                "estimator": LinearDiscriminantAnalysis()
            },
            "mlp": {
                "display_name": "MLP",
                "estimator": MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=1000,
                    random_state=self.random_state
                )
            },
            "bayes_lr": {
                "display_name": "Bayesian LR",
                "estimator": LogisticRegression(
                    solver='liblinear',
                    penalty='l1',
                    C=0.1,
                    class_weight='balanced',
                    random_state=self.random_state
                )
            },
            "extra_trees": {
                "display_name": "Extra Trees",
                "estimator": ExtraTreesClassifier(
                    n_estimators=300,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            },
        }

    def get_dl_registry(self):
        return {
            "cnn_base": {
                "display_name": "1D-CNN (Base)",
                "builder": self.build_simple_cnn,
                "build_params": {}
            },
            "resnet1d": {
                "display_name": "ResNet-1D",
                "builder": self.build_resnet,
                "build_params": {}
            },
            "cnn_lstm": {
                "display_name": "CNN-LSTM",
                "builder": self.build_cnn_lstm,
                "build_params": {}
            },
            "dilated_cnn": {
                "display_name": "Dilated-CNN",
                "builder": self.build_dilated_cnn,
                "build_params": {}
            },
            "attn_cnn": {
                "display_name": "Attention-CNN",
                "builder": self.build_attn_cnn,
                "build_params": {}
            },
        }

    def get_selected_sklearn_models(self, selected_models, params_by_model=None, w_ratio=1.0):
        registry = self.get_sklearn_registry(w_ratio=w_ratio)
        params_by_model = params_by_model or {}
        selected = {}

        for model_key in selected_models:
            if model_key not in registry:
                raise ValueError(f"Unknown sklearn model key: {model_key}")
            item = deepcopy(registry[model_key])
            estimator = clone(item["estimator"])
            if model_key in params_by_model:
                estimator.set_params(**params_by_model[model_key])
            selected[model_key] = {
                "display_name": item["display_name"],
                "estimator": estimator
            }

        return selected

    def get_selected_dl_models(self, selected_models, params_by_model=None):
        registry = self.get_dl_registry()
        params_by_model = params_by_model or {}
        selected = {}

        for model_key in selected_models:
            if model_key not in registry:
                raise ValueError(f"Unknown DL model key: {model_key}")
            item = deepcopy(registry[model_key])
            build_params = item.get("build_params", {}).copy()
            if model_key in params_by_model:
                build_params.update(params_by_model[model_key])
            selected[model_key] = {
                "display_name": item["display_name"],
                "builder": item["builder"],
                "build_params": build_params
            }

        return selected

    @staticmethod
    def _safe_metrics(y_true, y_prob, y_pred):
        auc = roc_auc_score(y_true, y_prob)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return auc, tn, fp, fn, tp, sensitivity, specificity

    def train_eval_sklearn_models(
        self,
        models,
        X_train,
        y_train,
        X_test,
        y_test,
        cv=None
    ):
        if cv is None:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        results_table = []
        perf_profile_data = {}
        roc_entries = []

        model_bar = self._make_tqdm(
            iterable=models.items(),
            desc="Sklearn models",
            colour="green",
            leave=True
        )

        for model_key, model_info in model_bar:
            name = model_info["display_name"]
            model = model_info["estimator"]

            cv_aucs = []
            cv_times = []
            splits = list(cv.split(X_train, y_train))

            fold_bar = self._make_tqdm(
                iterable=enumerate(splits, start=1),
                total=len(splits),
                desc=f"{name} CV",
                colour="cyan",
                leave=False
            )

            for fold_idx, (tr_idx, val_idx) in fold_bar:
                X_tr, X_val = X_train[tr_idx], X_train[val_idx]
                y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

                fold_model = clone(model)

                fold_start = time.time()
                fold_model.fit(X_tr, y_tr)
                if hasattr(fold_model, "predict_proba"):
                    p = fold_model.predict_proba(X_val)[:, 1]
                else:
                    p = fold_model.decision_function(X_val)
                fold_end = time.time()

                fold_auc = roc_auc_score(y_val, p)
                cv_aucs.append(fold_auc)
                cv_times.append(fold_end - fold_start)

                if self.show_progress:
                    fold_bar.set_postfix({
                        "fold": fold_idx,
                        "auc": f"{fold_auc:.4f}",
                        "mean_auc": f"{np.mean(cv_aucs):.4f}",
                        "sec": f"{cv_times[-1]:.1f}"
                    })

            perf_profile_data[name] = cv_aucs

            final_model = clone(model)

            test_bar = self._make_tqdm(
                total=1,
                desc=f"{name} Final fit",
                colour="yellow",
                leave=False
            )

            test_start = time.time()
            final_model.fit(X_train, y_train)
            if hasattr(final_model, "predict_proba"):
                y_prob = final_model.predict_proba(X_test)[:, 1]
            else:
                y_prob = final_model.decision_function(X_test)
            y_pred = final_model.predict(X_test)
            test_end = time.time()
            test_bar.update(1)
            test_bar.close()

            auc, tn, fp, fn, tp, sensitivity, specificity = self._safe_metrics(y_test, y_prob, y_pred)

            results_table.append({
                "Model": name,
                "Model_Key": model_key,
                "Family": "sklearn",
                "Train_mean_AUC": np.mean(cv_aucs),
                "Train_std_AUC": np.std(cv_aucs),
                "Train_mean_time": np.mean(cv_times),
                "Test_AUC": auc,
                "Test_Sensitivity": sensitivity,
                "Test_Specificity": specificity,
                "Test_TP": tp,
                "Test_TN": tn,
                "Test_FP": fp,
                "Test_FN": fn,
                "Test_time": test_end - test_start
            })

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_entries.append({
                "name": name,
                "auc": auc,
                "fpr": fpr,
                "tpr": tpr,
                "linestyle": "-"
            })

            if self.show_progress:
                model_bar.set_postfix({
                    "current": name,
                    "cv_auc": f"{np.mean(cv_aucs):.4f}",
                    "test_auc": f"{auc:.4f}",
                    "sens": f"{sensitivity:.4f}",
                    "spec": f"{specificity:.4f}"
                })

        return results_table, perf_profile_data, roc_entries

    def train_eval_dl_models(
        self,
        models,
        X_train,
        y_train,
        X_test,
        y_test,
        class_weight=None,
        device="/CPU:0",
        cv=None,
        cv_epochs=15,
        final_epochs=20,
        batch_size=64,
        optimizer="adam",
        loss="binary_crossentropy",
        threshold=0.5
    ):
        if cv is None:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        results_table = []
        perf_profile_data = {}
        roc_entries = []

        input_shape = (X_train.shape[1], X_train.shape[2])

        model_bar = self._make_tqdm(
            iterable=models.items(),
            desc="DL models",
            colour="magenta",
            leave=True
        )

        for model_key, model_info in model_bar:
            name = model_info["display_name"]
            builder = model_info["builder"]
            build_params = model_info.get("build_params", {})

            cv_aucs = []
            cv_times = []
            splits = list(cv.split(X_train, y_train))

            fold_bar = self._make_tqdm(
                iterable=enumerate(splits, start=1),
                total=len(splits),
                desc=f"{name} CV",
                colour="blue",
                leave=False
            )

            for fold_idx, (tr_idx, val_idx) in fold_bar:
                X_tr, X_val = X_train[tr_idx], X_train[val_idx]
                y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

                with tf.device(device):
                    m = builder(input_shape, **build_params)
                    m.compile(optimizer=optimizer, loss=loss)

                    callbacks = []
                    if self.show_progress:
                        callbacks.append(
                            TqdmKerasCallback(
                                epochs=cv_epochs,
                                desc=f"{name} Fold {fold_idx} Epochs",
                                colour="cyan"
                            )
                        )

                    fold_start = time.time()
                    m.fit(
                        X_tr,
                        y_tr,
                        epochs=cv_epochs,
                        batch_size=batch_size,
                        verbose=0,
                        class_weight=class_weight,
                        callbacks=callbacks
                    )
                    y_val_prob = m.predict(X_val, verbose=0).flatten()
                    fold_end = time.time()

                fold_auc = roc_auc_score(y_val, y_val_prob)
                cv_aucs.append(fold_auc)
                cv_times.append(fold_end - fold_start)

                if self.show_progress:
                    fold_bar.set_postfix({
                        "fold": fold_idx,
                        "auc": f"{fold_auc:.4f}",
                        "mean_auc": f"{np.mean(cv_aucs):.4f}",
                        "sec": f"{cv_times[-1]:.1f}"
                    })

                tf.keras.backend.clear_session()

            perf_profile_data[name] = cv_aucs

            with tf.device(device):
                final_m = builder(input_shape, **build_params)
                final_m.compile(optimizer=optimizer, loss=loss)

                callbacks = []
                if self.show_progress:
                    callbacks.append(
                        TqdmKerasCallback(
                            epochs=final_epochs,
                            desc=f"{name} Final Epochs",
                            colour="yellow"
                        )
                    )

                test_start = time.time()
                final_m.fit(
                    X_train,
                    y_train,
                    epochs=final_epochs,
                    batch_size=batch_size,
                    verbose=0,
                    class_weight=class_weight,
                    callbacks=callbacks
                )
                y_prob = final_m.predict(X_test, verbose=0).flatten()
                y_pred = (y_prob > threshold).astype(int)
                test_end = time.time()

            auc, tn, fp, fn, tp, sensitivity, specificity = self._safe_metrics(y_test, y_prob, y_pred)

            results_table.append({
                "Model": name,
                "Model_Key": model_key,
                "Family": "deep_learning",
                "Train_mean_AUC": np.mean(cv_aucs),
                "Train_std_AUC": np.std(cv_aucs),
                "Train_mean_time": np.mean(cv_times),
                "Test_AUC": auc,
                "Test_Sensitivity": sensitivity,
                "Test_Specificity": specificity,
                "Test_TP": tp,
                "Test_TN": tn,
                "Test_FP": fp,
                "Test_FN": fn,
                "Test_time": test_end - test_start
            })

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_entries.append({
                "name": name,
                "auc": auc,
                "fpr": fpr,
                "tpr": tpr,
                "linestyle": "--"
            })

            if self.show_progress:
                model_bar.set_postfix({
                    "current": name,
                    "cv_auc": f"{np.mean(cv_aucs):.4f}",
                    "test_auc": f"{auc:.4f}",
                    "sens": f"{sensitivity:.4f}",
                    "spec": f"{specificity:.4f}"
                })

            tf.keras.backend.clear_session()

        return results_table, perf_profile_data, roc_entries