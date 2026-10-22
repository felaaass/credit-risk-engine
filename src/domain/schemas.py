from typing import Literal

from pydantic import BaseModel, Field


class CreditApplicantInput(BaseModel):
    limit_bal: float = Field(..., gt=0, description="Credit limit in NT dollars")
    sex: int = Field(..., ge=1, le=2)
    education: int = Field(..., ge=0, le=6)
    marriage: int = Field(..., ge=0, le=3)
    age: int = Field(..., ge=18, le=100)
    pay_0: int = Field(..., ge=-2, le=9)
    pay_2: int = Field(..., ge=-2, le=9)
    pay_3: int = Field(..., ge=-2, le=9)
    pay_4: int = Field(..., ge=-2, le=9)
    pay_5: int = Field(..., ge=-2, le=9)
    pay_6: int = Field(..., ge=-2, le=9)
    bill_amt1: float = Field(...)
    bill_amt2: float = Field(...)
    bill_amt3: float = Field(...)
    bill_amt4: float = Field(...)
    bill_amt5: float = Field(...)
    bill_amt6: float = Field(...)
    pay_amt1: float = Field(..., ge=0)
    pay_amt2: float = Field(..., ge=0)
    pay_amt3: float = Field(..., ge=0)
    pay_amt4: float = Field(..., ge=0)
    pay_amt5: float = Field(..., ge=0)
    pay_amt6: float = Field(..., ge=0)


class FeatureContribution(BaseModel):
    feature: str
    value: float
    shap_impact: float


class CreditPredictionResponse(BaseModel):
    probability_of_default: float = Field(..., ge=0.0, le=1.0)
    risk_tier: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]
    decision: Literal["APPROVED", "MANUAL_REVIEW", "REJECTED"]
    optimal_threshold: float
    top_risk_factors: list[FeatureContribution]


class BatchCreditPredictionRequest(BaseModel):
    applicants: list[CreditApplicantInput] = Field(..., min_length=1, max_length=1000)


class BatchCreditPredictionResponse(BaseModel):
    predictions: list[CreditPredictionResponse]
    total_processed: int


class ModelMetricsSummary(BaseModel):
    roc_auc: float
    gini_coefficient: float
    ks_statistic: float
    pr_auc: float
    brier_score: float
    f1_score: float
    precision: float
    recall: float
    optimal_threshold: float
    expected_cost_per_applicant: float
    confusion_matrix: dict[str, int]


class HealthCheckResponse(BaseModel):
    status: Literal["ok", "degraded"]
    app_name: str
    version: str
    model_loaded: bool
    optimal_threshold: float | None = None
