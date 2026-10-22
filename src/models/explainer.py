import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from src.domain.schemas import FeatureContribution


class CreditShapExplainer:
    def __init__(self, model: lgb.LGBMClassifier) -> None:
        self.model = model
        self.explainer = shap.TreeExplainer(model.booster_)

    def _extract_positive_class_shap(self, features: pd.DataFrame) -> np.ndarray:
        raw_shap = self.explainer.shap_values(features)
        if isinstance(raw_shap, list):
            return np.asarray(raw_shap[1], dtype=float)
        arr = np.asarray(raw_shap, dtype=float)
        if arr.ndim == 3:
            return arr[:, :, 1]
        return arr

    def explain_instances(
        self, features: pd.DataFrame, top_k: int = 5
    ) -> list[list[FeatureContribution]]:
        shap_matrix = self._extract_positive_class_shap(features)
        feature_names = list(features.columns)
        explanations: list[list[FeatureContribution]] = []

        for row_idx in range(len(features)):
            row_shap = shap_matrix[row_idx]
            row_values = features.iloc[row_idx]
            top_indices = np.argsort(np.abs(row_shap))[::-1][:top_k]

            contributions: list[FeatureContribution] = [
                FeatureContribution(
                    feature=feature_names[idx],
                    value=round(float(row_values.iloc[idx]), 4),
                    shap_impact=round(float(row_shap[idx]), 5),
                )
                for idx in top_indices
            ]
            explanations.append(contributions)

        return explanations

    def compute_global_importance(self, features: pd.DataFrame) -> dict[str, float]:
        shap_matrix = self._extract_positive_class_shap(features)
        mean_abs = np.mean(np.abs(shap_matrix), axis=0)
        importance_Pairs = sorted(
            zip(features.columns, mean_abs, strict=False),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return {name: round(float(val), 5) for name, val in importance_Pairs}
