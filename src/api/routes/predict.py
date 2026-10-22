from fastapi import APIRouter, Request

from src.api.service import CreditScoringService
from src.domain.schemas import (
    BatchCreditPredictionRequest,
    BatchCreditPredictionResponse,
    CreditApplicantInput,
    CreditPredictionResponse,
)

router = APIRouter(prefix="/v1", tags=["Credit Scoring"])


@router.post("/predict", response_model=CreditPredictionResponse)
def predict_credit_risk(
    payload: CreditApplicantInput, request: Request
) -> CreditPredictionResponse:
    service: CreditScoringService = request.app.state.scoring_service
    return service.predict_single(payload)


@router.post("/predict/batch", response_model=BatchCreditPredictionResponse)
def predict_credit_risk_batch(
    payload: BatchCreditPredictionRequest, request: Request
) -> BatchCreditPredictionResponse:
    service: CreditScoringService = request.app.state.scoring_service
    return service.predict_batch(payload.applicants)
