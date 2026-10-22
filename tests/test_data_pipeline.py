import polars as pl
import pytest

from src.data.preprocessor import CreditDataPreprocessor
from src.domain.schemas import CreditApplicantInput
from src.features.feature_engineering import CreditFeatureEngineer


def _sample_credit_dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "limit_bal": [120000.0, 50000.0, 200000.0, 80000.0],
            "sex": [2, 1, 2, 1],
            "education": [5, 2, 1, 0],
            "marriage": [1, 0, 2, 1],
            "age": [29, 41, 35, 26],
            "pay_0": [0, 2, -1, 1],
            "pay_2": [0, 2, -1, 2],
            "pay_3": [0, 0, -1, 2],
            "pay_4": [0, 0, 0, 0],
            "pay_5": [0, 0, 0, 0],
            "pay_6": [0, 0, 0, 0],
            "bill_amt1": [15400.0, 48200.0, 4200.0, 61000.0],
            "bill_amt2": [14200.0, 46900.0, 3800.0, 59500.0],
            "bill_amt3": [13800.0, 45100.0, 4100.0, 58000.0],
            "bill_amt4": [12900.0, 29000.0, 3500.0, 42000.0],
            "bill_amt5": [11500.0, 27500.0, 3100.0, 39000.0],
            "bill_amt6": [10900.0, 26000.0, 2900.0, 37500.0],
            "pay_amt1": [2000.0, 0.0, 3800.0, 1500.0],
            "pay_amt2": [2000.0, 1800.0, 4100.0, 0.0],
            "pay_amt3": [1500.0, 1500.0, 3500.0, 1200.0],
            "pay_amt4": [1500.0, 1200.0, 3100.0, 1000.0],
            "pay_amt5": [1200.0, 1000.0, 2900.0, 1000.0],
            "pay_amt6": [1200.0, 1000.0, 3000.0, 1000.0],
            "target": [0, 1, 0, 1],
        }
    )


def test_preprocessor_normalizes_undocumented_categories() -> None:
    raw_df = _sample_credit_dataframe()
    preprocessor = CreditDataPreprocessor()
    cleaned = preprocessor.clean(raw_df)

    assert cleaned["education"].to_list() == [4, 2, 1, 4]
    assert cleaned["marriage"].to_list() == [1, 3, 2, 1]


def test_feature_engineering_adds_financial_ratios() -> None:
    raw_df = _sample_credit_dataframe()
    preprocessor = CreditDataPreprocessor()
    engineer = CreditFeatureEngineer()

    enriched = engineer.transform_polars(preprocessor.clean(raw_df))

    expected_new_cols = {
        "total_bill_6m",
        "total_pay_6m",
        "utilization_ratio_m1",
        "avg_utilization_6m",
        "max_utilization_6m",
        "payment_to_bill_ratio_6m",
        "recent_payment_gap_m1",
        "bill_growth_6m",
        "max_delinquency_months",
        "months_past_due_count",
        "severe_delinquency_count",
        "recent_delinquency_trend",
    }
    assert expected_new_cols.issubset(set(enriched.columns))
    assert enriched["months_past_due_count"].to_list() == [0, 2, 0, 3]
    assert enriched["severe_delinquency_count"].to_list() == [0, 2, 0, 2]
    assert pytest.approx(enriched["utilization_ratio_m1"][0], rel=1e-4) == 15400.0 / 120000.0


def test_applicant_schema_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError):
        CreditApplicantInput(
            limit_bal=-5000.0,
            sex=1,
            education=2,
            marriage=1,
            age=34,
            pay_0=0,
            pay_2=0,
            pay_3=0,
            pay_4=0,
            pay_5=0,
            pay_6=0,
            bill_amt1=10000.0,
            bill_amt2=9500.0,
            bill_amt3=9000.0,
            bill_amt4=8500.0,
            bill_amt5=8000.0,
            bill_amt6=7500.0,
            pay_amt1=1500.0,
            pay_amt2=1500.0,
            pay_amt3=1500.0,
            pay_amt4=1500.0,
            pay_amt5=1500.0,
            pay_amt6=1500.0,
        )
