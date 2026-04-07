BILSTM_EXPERIMENTS = {
    "bilstm_base_smote_12000": {
        "dataset": {
            "impute": True,
            "k": 5,
            "apply_pca": False,
            "split_size": 0.15,
            "random_state": 42
        },
        "resampling": {
            "use_sasmote": True,
            "params": {
                "k_neighbors": 50,
                "rf_n_estimators": 150,
                "rf_max_depth": 12,
                "random_state": 42,
                "n_to_generate": 12000
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
            "deep_learning": ["cnn_bilstm_attention"]
        },
        "model_params": {
            "deep_learning": {
                "cnn_bilstm_attention": {
                    "conv1_filters": 64,
                    "conv2_filters": 128,
                    "kernel_size": 64,
                    "pool_size": 4,
                    "bilstm_units": 128,
                    "dense_units": 128,
                    "dropout": 0.30
                }
            }
        },
        "dl_train": {
            "cv_epochs": 20,
            "final_epochs": 30,
            "batch_size": 64,
            "optimizer": "adam",
            "loss_name": "focal",
            "loss_params": {
                "alpha": 0.25,
                "gamma": 2.0,
                "apply_class_balancing": False,
                "from_logits": False,
                "label_smoothing": 0.0
            },
            "threshold": 0.5
        }
    },

    "bilstm_base_smote_full": {
        "dataset": {
            "impute": True,
            "k": 5,
            "apply_pca": False,
            "split_size": 0.15,
            "random_state": 42
        },
        "resampling": {
            "use_sasmote": True,
            "params": {
                "k_neighbors": 50,
                "rf_n_estimators": 150,
                "rf_max_depth": 12,
                "random_state": 42,
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
            "deep_learning": ["cnn_bilstm_attention"]
        },
        "model_params": {
            "deep_learning": {
                "cnn_bilstm_attention": {
                    "conv1_filters": 64,
                    "conv2_filters": 128,
                    "kernel_size": 64,
                    "pool_size": 4,
                    "bilstm_units": 128,
                    "dense_units": 128,
                    "dropout": 0.30
                }
            }
        },
        "dl_train": {
            "cv_epochs": 20,
            "final_epochs": 30,
            "batch_size": 64,
            "optimizer": "adam",
            "loss_name": "focal",
            "loss_params": {
                "alpha": 0.25,
                "gamma": 2.0,
                "apply_class_balancing": False,
                "from_logits": False,
                "label_smoothing": 0.0
            },
            "threshold": 0.5
        }
    },
}