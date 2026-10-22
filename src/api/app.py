from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.routes.health import router as health_router
from src.api.routes.predict import router as predict_router
from src.api.service import CreditScoringService
from src.config import Settings, get_settings
from src.exceptions import CreditRiskError


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        service = CreditScoringService(cfg)
        service.load_artifacts()
        app.state.scoring_service = service
        yield

    app = FastAPI(
        title=cfg.app_name,
        version=cfg.app_version,
        lifespan=lifespan,
    )

    @app.exception_handler(CreditRiskError)
    async def credit_risk_exception_handler(
        request: Request, exc: CreditRiskError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.__class__.__name__,
                "detail": exc.message,
                "path": str(request.url.path),
            },
        )

    app.include_router(health_router)
    app.include_router(predict_router)
    return app


app = create_app()
