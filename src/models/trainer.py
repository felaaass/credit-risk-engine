import json

import joblib
import lightgbm as lgb
import mlflow

from src.config import Settings, get_settings
from src.data.loader import CreditDataLoader
from src.data.preprocessor import CreditDataPreprocessor
from src.domain.schemas import ModelMetricsSummary
from src.features.feature_engineering import CreditFeatureEngineer
from src.models.evaluator import CreditModelEvaluator
from src.models.explainer import CreditShapExplainer
from src.models.optimizer import LightGBMOptimizer


class CreditModelTrainer:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.loader = CreditDataLoader(self.settings)
        self.preprocessor = CreditDataPreprocessor(self.settings)
        self.feature_engineer = CreditFeatureEngineer()
        self.optimizer = LightGBMOptimizer(self.settings)
        self.evaluator = CreditModelEvaluator(self.settings)

    def run_training_pipeline(
        self, n_trials: int | None = None
    ) -> tuple[lgb.LGBMClassifier, ModelMetricsSummary, dict[str, float]]:
        raw_df = self.loader.load_raw_data()
        cleaned_df = self.preprocessor.clean(raw_df)
        enriched_df = self.feature_engineer.transform_polars(cleaned_df)

        self.settings.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        enriched_df.write_parquet(self.settings.processed_data_path)

        x_train, x_test, y_train, y_test = self.preprocessor.split_train_test(enriched_df)

        best_params = self.optimizer.optimize(x_train, y_train, n_trials=n_trials)
        model = lgb.LGBMClassifier(**best_params)
        model.fit(x_train, y_train)

        test_probs = model.predict_proba(x_test)[:, 1]
        metrics = self.evaluator.evaluate(y_test, test_probs)

        explainer = CreditShapExplainer(model)
        shap_sample = x_test.sample(
            n=min(1000, len(x_test)), random_state=self.settings.random_seed
        )
        global_shap = explainer.compute_global_importance(shap_sample)

        self._log_to_mlflow(best_params, metrics, global_shap)
        self._save_artifacts(model, best_params, metrics, global_shap, list(x_train.columns))

        return model, metrics, global_shap

    def _log_to_mlflow(
        self,
        params: dict[str, int | float | str],
        metrics: ModelMetricsSummary,
        global_shap: dict[str, float],
    ) -> None:
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        mlflow.set_experiment(self.settings.mlflow_experiment_name)

        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(
                {
                    "roc_auc": metrics.roc_auc,
                    "gini_coefficient": metrics.gini_coefficient,
                    "ks_statistic": metrics.ks_statistic,
                    "pr_auc": metrics.pr_auc,
                    "brier_score": metrics.brier_score,
                    "f1_score": metrics.f1_score,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "optimal_threshold": metrics.optimal_threshold,
                    "expected_cost_per_applicant": metrics.expected_cost_per_applicant,
                }
            )
            mlflow.log_dict(global_shap, "shap_global_importance.json")

    def _save_artifacts(
        self,
        model: lgb.LGBMClassifier,
        params: dict[str, int | float | str],
        metrics: ModelMetricsSummary,
        global_shap: dict[str, float],
        feature_columns: list[str],
    ) -> None:
        self.settings.model_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.settings.model_artifact_path)

        metadata = {
            "app_version": self.settings.app_version,
            "optimal_threshold": metrics.optimal_threshold,
            "feature_columns": feature_columns,
            "best_hyperparameters": params,
        }
        self.settings.metadata_artifact_path.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        report_payload = {
            "dataset": {
                "source": "UCI Machine Learning Repository (Dataset ID 350 / OpenML 42477)",
                "name": "Default of Credit Card Clients",
                "total_records": 30000,
                "test_records": int(30000 * self.settings.test_size),
            },
            "metrics": metrics.model_dump(),
            "top_shap_features": dict(list(global_shap.items())[:15]),
            "hyperparameters": params,
        }
        self.settings.metrics_report_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings.metrics_report_path.write_text(
            json.dumps(report_payload, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    trainer = CreditModelTrainer()
    trainer.run_training_pipeline()
