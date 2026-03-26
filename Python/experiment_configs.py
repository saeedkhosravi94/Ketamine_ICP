EXPERIMENTS = {



    "baseline_raw": {
        "dataset": {
            "impute": True,
            "k": 5,
            "apply_pca": False,
            "split_size": 0.15,
            "random_state": 42
        },
        "resampling": {
            "use_sasmote": False
        },
        "preprocessing": {
            "scaler": "standard"
        },
        "cv": {
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42
        },
        "models": {
            "sklearn": ["logreg", "rf", "xgb", "lgbm", "catboost"],
            "deep_learning": []
        },
        "model_params": {
            "sklearn": {
                "rf": {"n_estimators": 300, "max_depth": 12},
                "xgb": {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
                "lgbm": {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05},
            },
            "deep_learning": {}
        },
        "dl_train": {
            "cv_epochs": 10,
            "final_epochs": 15,
            "batch_size": 64,
            "threshold": 0.5
        }
    },

    "pca_sasmote_mixed": {
        "dataset": {
            "impute": True,
            "k": 5,
            "apply_pca": True,
            "signals_variance_threshold": 0.98,
            "stats_variance_threshold": 0.98,
            "split_size": 0.15,
            "random_state": 42
        },
        "resampling": {
            "use_sasmote": True,
            "params": {
                "k_neighbors": 20,
                "rf_n_estimators": 100,
                "rf_max_depth": None,
                "random_state": 42
            }
        },
        "preprocessing": {
            "scaler": "standard"
        },
        "cv": {
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42
        },
        "models": {
            "sklearn": ["logreg", "rf", "xgb", "lgbm", "catboost", "mlp", "bayes_lr"],
            "deep_learning": ["cnn_base", "resnet1d", "dilated_cnn"]
        },
        "model_params": {
            "sklearn": {
                "rf": {"n_estimators": 500, "max_depth": 16},
                "xgb": {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.9},
                "lgbm": {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 63},
                "mlp": {"hidden_layer_sizes": (256, 128), "alpha": 1e-4}
            },
            "deep_learning": {
                "cnn_base": {"conv_filters": 64, "dense_units": 128, "dropout": 0.3},
                "resnet1d": {"filters": 64},
                "dilated_cnn": {"filters": 64}
            }
        },
        "dl_train": {
            "cv_epochs": 15,
            "final_epochs": 20,
            "batch_size": 64,
            "threshold": 0.5
        }
    },

    "impute_5_sig_pca_0.95_stats_pca_0.95_SASMOTE": {
        "dataset": {
            "impute": True,
            "k": 5,
            "apply_pca": True,
            "signals_variance_threshold": 0.95,
            "stats_variance_threshold": 0.95,
            "split_size": 0.15,
            "random_state": 42
        },
        "resampling": {
            "use_sasmote": True,
            "params": {
                "k_neighbors": 25,
                "rf_n_estimators": 150,
                "rf_max_depth": 12,
                "random_state": 42
            }
        },
        "preprocessing": {
            "scaler": "standard"
        },
        "cv": {
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42
        },
        "models": {
            "sklearn": ["rf", "xgb", "lgbm", "catboost", "mlp", "extra_trees"],
            "deep_learning": ["cnn_base", "resnet1d", "cnn_lstm", "dilated_cnn", "attn_cnn"]
        },
        "model_params": {
            "sklearn": {
                "extra_trees": { "n_estimators": 500, "max_depth": 20},
                "rf": {"n_estimators": 800, "max_depth": 20, "min_samples_leaf": 2},
                "xgb": {
                    "n_estimators": 600,
                    "max_depth": 4,
                    "learning_rate": 0.03,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "reg_alpha": 0.5,
                    "reg_lambda": 2.0
                },
                "lgbm": {
                    "n_estimators": 600,
                    "learning_rate": 0.03,
                    "num_leaves": 63,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9
                },
                "catboost": {
                    "iterations": 500,
                    "depth": 6,
                    "learning_rate": 0.03
                },
                "mlp": {
                    "hidden_layer_sizes": (512, 256, 64),
                    "alpha": 1e-4,
                    "early_stopping": True
                }
            },
            "deep_learning": {
                "cnn_base": {"conv_filters": 64, "dense_units": 128, "dropout": 0.3},
                "resnet1d": {"filters": 64},
                "cnn_lstm": {"conv_filters": 64, "lstm_units": 64, "dropout": 0.35},
                "dilated_cnn": {"filters": 64},
                "attn_cnn": {"filters": 64, "num_heads": 4, "key_dim": 32}
            }
        },
        "dl_train": {
            "cv_epochs": 20,
            "final_epochs": 30,
            "batch_size": 64,
            "threshold": 0.5
        }
    },


    "impute_5_sig_pca_0.95_stats_pca_0.95_no_SASMOTE": {
        "dataset": {
            "impute": True,
            "k": 5,
            "apply_pca": True,
            "signals_variance_threshold": 0.95,
            "stats_variance_threshold": 0.95,
            "split_size": 0.15,
            "random_state": 42
        },
        "resampling": {
            "use_sasmote": False,
            "params": {
                "k_neighbors": 25,
                "rf_n_estimators": 150,
                "rf_max_depth": 12,
                "random_state": 42
            }
        },
        "preprocessing": {
            "scaler": "standard"
        },
        "cv": {
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42
        },
        "models": {
            "sklearn": ["rf", "xgb", "lgbm", "catboost", "mlp", "extra_trees"],
            "deep_learning": ["cnn_base", "resnet1d", "cnn_lstm", "dilated_cnn", "attn_cnn"]
        },
        "model_params": {
            "sklearn": {
                "extra_trees": { "n_estimators": 500, "max_depth": 20},
                "rf": {"n_estimators": 800, "max_depth": 20, "min_samples_leaf": 2},
                "xgb": {
                    "n_estimators": 600,
                    "max_depth": 4,
                    "learning_rate": 0.03,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "reg_alpha": 0.5,
                    "reg_lambda": 2.0
                },
                "lgbm": {
                    "n_estimators": 600,
                    "learning_rate": 0.03,
                    "num_leaves": 63,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9
                },
                "catboost": {
                    "iterations": 500,
                    "depth": 6,
                    "learning_rate": 0.03
                },
                "mlp": {
                    "hidden_layer_sizes": (512, 256, 64),
                    "alpha": 1e-4,
                    "early_stopping": True
                }
            },
            "deep_learning": {
                "cnn_base": {"conv_filters": 64, "dense_units": 128, "dropout": 0.3},
                "resnet1d": {"filters": 64},
                "cnn_lstm": {"conv_filters": 64, "lstm_units": 64, "dropout": 0.35},
                "dilated_cnn": {"filters": 64},
                "attn_cnn": {"filters": 64, "num_heads": 4, "key_dim": 32}
            }
        },
        "dl_train": {
            "cv_epochs": 20,
            "final_epochs": 30,
            "batch_size": 64,
            "threshold": 0.5
        }
    }
}