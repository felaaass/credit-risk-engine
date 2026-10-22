import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import Settings, get_settings
from src.domain.schemas import ModelMetricsSummary


class CreditModelEvaluator:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def compute_ks_statistic(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        pos_probs = y_prob[y_true == 1]
        neg_probs = y_prob[y_true == 0]
        if len(pos_probs) == 0 or len(neg_probs) == 0:
            return 0.0
        ks_result = ks_2samp(pos_probs, neg_probs)
        return float(ks_result.statistic)

    def find_optimal_threshold(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> tuple[float, float]:
        thresholds = np.linspace(0.05, 0.85, 81)
        best_threshold = 0.5
        lowest_cost = float("inf")
        n_samples = len(y_true)

        for threshold in thresholds:
            preds = (y_prob >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
            total_cost = (
                fn * self.settings.cost_false_negative
                + fp * self.settings.cost_false_positive
            )
            avg_cost = float(total_cost) / max(n_samples, 1)
            if avg_cost < lowest_cost:
                lowest_cost = avg_cost
                best_threshold = float(threshold)

        return round(best_threshold, 4), round(lowest_cost, 6)

    def evaluate(
        self,
        y_true: pd.Series | np.ndarray,
        y_prob: np.ndarray,
        threshold: float | None = None,
    ) -> ModelMetricsSummary:
        y_arr = np.asarray(y_true, dtype=int)
        prob_arr = np.asarray(y_prob, dtype=float)

        roc_auc = float(roc_auc_score(y_arr, prob_arr))
        gini = 2.0 * roc_auc - 1.0
        ks_stat = self.compute_ks_statistic(y_arr, prob_arr)
        pr_auc = float(average_precision_score(y_arr, prob_arr))
        brier = float(brier_score_loss(y_arr, prob_arr))

        opt_threshold, min_cost = self.find_optimal_threshold(y_arr, prob_arr)
        applied_threshold = threshold if threshold is not None else opt_threshold

        preds = (prob_arr >= applied_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_arr, preds, labels=[0, 1]).ravel()

        if threshold is not None:
            total_cost = (
                fn * self.settings.cost_false_negative
                + fp * self.settings.cost_false_positive
            )
            min_cost = float(total_cost) / max(len(y_arr), 1)

        return ModelMetricsSummary(
            roc_auc=round(roc_auc, 5),
            gini_coefficient=round(gini, 5),
            ks_statistic=round(ks_stat, 5),
            pr_auc=round(pr_auc, 5),
            brier_score=round(brier, 5),
            f1_score=round(float(f1_score(y_arr, preds, zero_division=0)), 5),
            precision=round(float(precision_score(y_arr, preds, zero_division=0)), 5),
            recall=round(float(recall_score(y_arr, preds, zero_division=0)), 5),
            optimal_threshold=round(applied_threshold, 4),
            expected_cost_per_applicant=round(min_cost, 5),
            confusion_matrix={
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
        )
