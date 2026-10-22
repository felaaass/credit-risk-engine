import json
from typing import Literal

import joblib
import lightgbm as lgb
import polars as pl

from src.config import Settings, get_settings
from src.data.preprocessor import CreditDataPreprocessor
from src.domain.schemas import (
    BatchCreditPredictionResponse,
    CreditApplicantInput,
    CreditPredictionResponse,
)
from src.exceptions import InferenceExecutionError, ModelNotLoadedError
from src.features.feature_engineering import CreditFeatureEngineer
from src.models.explainer import CreditShapExplainer


class CreditScoringService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.preprocessor = CreditDataPreprocessor(self.settings)
        self.feature_engineer = CreditFeatureEngineer()
        self.model: lgb.LGBMClassifier | None = None
        self.explainer: CreditShapExplainer | None = None
        self.optimal_threshold: float = 0.25
        self.feature_columns: list[str] = []

    def load_artifacts(self) -> None:
        if not self.settings.model_artifact_path.exists():
            return
        self.model = joblib.load(self.settings.model_artifact_path)
        self.explainer = CreditShapExplainer(self.model)

        if self.settings.metadata_artifact_path.exists():
            raw_meta = json.loads(
                self.settings.metadata_artifact_path.read_text(encoding="utf-8")
            )
            self.optimal_threshold = float(raw_meta.get("optimal_threshold", 0.25))
            self.feature_columns = list(raw_meta.get("feature_columns", []))

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.explainer is not None

    def _classify_risk(
        self, probability: float
    ) -> tuple[
        Literal["LOW", "MODERATE", "HIGH", "CRITICAL"],
        Literal["APPROVED", "MANUAL_REVIEW", "REJECTED"],
    ]:
        t = self.optimal_threshold
        if probability < t * 0.65:
            return "LOW", "APPROVED"
        if probability < t:
            return "MODERATE", "APPROVED"
        if probability < min(t * 1.6, 0.75):
            return "HIGH", "MANUAL_REVIEW"
        return "CRITICAL", "REJECTED"

    def predict_single(
        self, applicant: CreditApplicantInput
    ) -> CreditPredictionResponse:
        batch_result = self.predict_batch([applicant])
        return batch_result.predictions[0]

    def predict_batch(
        self, applicants: list[CreditApplicantInput]
    ) -> BatchCreditPredictionResponse:
        if not self.is_ready or self.model is None or self.explainer is None:
            raise ModelNotLoadedError(
                "Trained model artifacts are not loaded. Run the training pipeline first."
            )

        try:
            records = [item.model_dump() for item in applicants]
            raw_df = pl.DataFrame(records)
            cleaned_df = self.preprocessor.clean(raw_df)
            enriched_df = self.feature_engineer.transform_polars(cleaned_df)
            pdf = enriched_df.to_pandas()

            if self.feature_columns:
                pdf = pdf[self.feature_columns]

            probs = self.model.predict_proba(pdf)[:, 1]
            explanations = self.explainer.explain_instances(pdf, top_k=5)
        except Exception as exc:
            raise InferenceExecutionError(f"Inference pipeline failed: {exc}") from exc

        responses: list[CreditPredictionResponse] = []
        for idx, prob in enumerate(probs):
            prob_val = round(float(prob), 5)
            risk_tier, decision = self._classify_risk(prob_val)
            responses.append(
                CreditPredictionResponse(
                    probability_of_default=prob_val,
                    risk_tier=risk_tier,
                    decision=decision,
                    optimal_threshold=self.optimal_threshold,
                    top_risk_factors=explanations[idx],
                )
            )

        return BatchCreditPredictionResponse(
            predictions=responses,
            total_processed=len(responses),
        )
