# experiment_runner.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from data import Dataset
from smote import SASMOTE
from models import ModelFactory
from utils import (
    set_device,
    scale_train_test,
    reshape_for_dl,
    get_classification_weights,
)

from experiment_configs import EXPERIMENTS


class ExperimentRunner:
    def __init__(self, data_path, device_type='GPU'):
        self.data_path = data_path
        self.device = set_device(device_type)

    def run(self, experiment_name):
        if experiment_name not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment: {experiment_name}")

        cfg = EXPERIMENTS[experiment_name]
        random_state = cfg["dataset"].get("random_state", 42)

        dataset = Dataset(
            self.data_path,
            **cfg["dataset"]
        )

        X_train = dataset.X_train.copy()
        y_train = dataset.y_train.copy()
        X_test = dataset.X_test.copy()
        y_test = dataset.y_test.copy()

        if cfg["resampling"].get("use_sasmote", False):
            print("Applying SASMOTE resampling to training data...")
            sas = SASMOTE(**cfg["resampling"]["params"])
            X_train_res, y_train_res = sas.fit_resample(X_train.values, y_train.values)
            X_train = pd.DataFrame(X_train_res, columns=X_train.columns)
            y_train = pd.Series(y_train_res, name='label')

        X_train_s, X_test_s, _ = scale_train_test(
            X_train,
            X_test,
            scaler_name=cfg["preprocessing"].get("scaler", "standard")
        )

        class_info = get_classification_weights(y_train)

        cv = StratifiedKFold(
            n_splits=cfg["cv"].get("n_splits", 5),
            shuffle=cfg["cv"].get("shuffle", True),
            random_state=cfg["cv"].get("random_state", random_state)
        )

        factory = ModelFactory(random_state=random_state)

        sk_models = factory.get_selected_sklearn_models(
            selected_models=cfg["models"].get("sklearn", []),
            params_by_model=cfg["model_params"].get("sklearn", {}),
            w_ratio=class_info["w_ratio"]
        )

        dl_models = factory.get_selected_dl_models(
            selected_models=cfg["models"].get("deep_learning", []),
            params_by_model=cfg["model_params"].get("deep_learning", {})
        )

        results_table = []
        perf_profile_data = {}
        roc_entries = []

        if sk_models:
            sk_results, sk_perf, sk_roc = factory.train_eval_sklearn_models(
                models=sk_models,
                X_train=X_train_s,
                y_train=y_train,
                X_test=X_test_s,
                y_test=y_test,
                cv=cv
            )
            results_table += sk_results
            perf_profile_data.update(sk_perf)
            roc_entries += sk_roc

        if dl_models:
            X_train_dl, X_test_dl = reshape_for_dl(X_train_s, X_test_s)

            dl_cfg = cfg.get("dl_train", {})
            dl_results, dl_perf, dl_roc = factory.train_eval_dl_models(
                models=dl_models,
                X_train=X_train_dl,
                y_train=y_train,
                X_test=X_test_dl,
                y_test=y_test,
                class_weight=class_info["class_weight_dict"],
                device=self.device,
                cv=cv,
                cv_epochs=dl_cfg.get("cv_epochs", 15),
                final_epochs=dl_cfg.get("final_epochs", 20),
                batch_size=dl_cfg.get("batch_size", 64),
                optimizer=dl_cfg.get("optimizer", "adam"),
                loss_name=dl_cfg.get("loss_name", "binary_crossentropy"),
                loss_params=dl_cfg.get("loss_params", {}),
                threshold=dl_cfg.get("threshold", 0.5)
            )
            results_table += dl_results
            perf_profile_data.update(dl_perf)
            roc_entries += dl_roc

        return {
            "experiment_name": experiment_name,
            "config": cfg,
            "results_table": results_table,
            "perf_profile_data": perf_profile_data,
            "roc_entries": roc_entries
        }