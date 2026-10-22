import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import Settings, get_settings

optuna.logging.set_verbosity(optuna.logging.WARNING)


class LightGBMOptimizer:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def optimize(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        n_trials: int | None = None,
    ) -> dict[str, int | float | str]:
        trials = n_trials if n_trials is not None else self.settings.optuna_trials
        sampler = TPESampler(seed=self.settings.random_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial: optuna.Trial) -> float:
            params: dict[str, int | float | str] = {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "n_estimators": trial.suggest_int("n_estimators", 120, 450),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 16, 96),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "subsample_freq": 1,
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "random_state": self.settings.random_seed,
                "n_jobs": -1,
                "verbose": -1,
            }

            skf = StratifiedKFold(
                n_splits=self.settings.cv_folds,
                shuffle=True,
                random_state=self.settings.random_seed,
            )
            fold_scores: list[float] = []

            for train_idx, val_idx in skf.split(x_train, y_train):
                x_tr = x_train.iloc[train_idx]
                y_tr = y_train.iloc[train_idx]
                x_va = x_train.iloc[val_idx]
                y_va = y_train.iloc[val_idx]

                model = lgb.LGBMClassifier(**params)
                model.fit(
                    x_tr,
                    y_tr,
                    eval_X=x_va,
                    eval_y=y_va,
                    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
                )
                val_probs = model.predict_proba(x_va)[:, 1]
                fold_scores.append(float(roc_auc_score(y_va, val_probs)))

            return float(np.mean(fold_scores))

        study.optimize(objective, n_trials=trials)

        best_params: dict[str, int | float | str] = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "subsample_freq": 1,
            "random_state": self.settings.random_seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        best_params.update(study.best_params)
        return best_params
