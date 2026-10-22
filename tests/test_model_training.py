import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.evaluator import CreditModelEvaluator
from src.models.explainer import CreditShapExplainer


def test_evaluator_calculates_banking_metrics() -> None:
    rng = np.random.default_rng(42)
    y_true = np.array([0] * 160 + [1] * 40)
    pos_scores = rng.uniform(0.35, 0.92, size=40)
    neg_scores = rng.uniform(0.02, 0.55, size=160)
    y_prob = np.concatenate([neg_scores, pos_scores])

    evaluator = CreditModelEvaluator()
    summary = evaluator.evaluate(y_true, y_prob)

    assert 0.70 <= summary.roc_auc <= 1.0
    assert summary.gini_coefficient == round(2.0 * summary.roc_auc - 1.0, 5)
    assert 0.40 <= summary.ks_statistic <= 1.0
    assert 0.0 <= summary.brier_score <= 0.35
    assert 0.05 <= summary.optimal_threshold <= 0.85
    assert sum(summary.confusion_matrix.values()) == 200


def test_shap_explainer_returns_ranked_contributions() -> None:
    rng = np.random.default_rng(42)
    n_rows = 120
    features = pd.DataFrame(
        {
            "limit_bal": rng.uniform(20000, 300000, size=n_rows),
            "pay_0": rng.integers(-1, 4, size=n_rows),
            "utilization_ratio_m1": rng.uniform(0.0, 1.1, size=n_rows),
            "months_past_due_count": rng.integers(0, 6, size=n_rows),
            "payment_to_bill_ratio_6m": rng.uniform(0.0, 0.8, size=n_rows),
            "age": rng.integers(21, 68, size=n_rows),
        }
    )
    target = (
        (features["pay_0"] >= 2) | (features["utilization_ratio_m1"] > 0.85)
    ).astype(int)

    model = lgb.LGBMClassifier(
        n_estimators=25,
        max_depth=4,
        num_leaves=15,
        random_state=42,
        verbose=-1,
    )
    model.fit(features, target)

    explainer = CreditShapExplainer(model)
    local_explanations = explainer.explain_instances(features.iloc[:3], top_k=4)

    assert len(local_explanations) == 3
    assert len(local_explanations[0]) == 4
    impacts = [abs(item.shap_impact) for item in local_explanations[0]]
    assert impacts == sorted(impacts, reverse=True)

    global_importance = explainer.compute_global_importance(features)
    assert set(global_importance.keys()) == set(features.columns)
