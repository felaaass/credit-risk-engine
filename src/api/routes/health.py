from fastapi import APIRouter, Request

from src.api.service import CreditScoringService
from src.config import get_settings
from src.domain.schemas import HealthCheckResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthCheckResponse)
def get_health(request: Request) -> HealthCheckResponse:
    settings = get_settings()
    service: CreditScoringService = request.app.state.scoring_service
    return HealthCheckResponse(
        status="ok" if service.is_ready else "degraded",
        app_name=settings.app_name,
        version=settings.app_version,
        model_loaded=service.is_ready,
        optimal_threshold=service.optimal_threshold if service.is_ready else None,
    )
