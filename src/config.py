from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "credit-risk-engine"
    app_version: str = "1.0.0"
    environment: str = "development"
    log_level: str = "INFO"

    base_dir: Path = Path(__file__).resolve().parent.parent
    raw_data_path: Path = base_dir / "data" / "raw" / "uci_credit_card_default.parquet"
    processed_data_path: Path = base_dir / "data" / "processed" / "credit_features.parquet"
    model_artifact_path: Path = base_dir / "artifacts" / "lightgbm_credit_model.joblib"
    metadata_artifact_path: Path = base_dir / "artifacts" / "model_metadata.json"
    metrics_report_path: Path = base_dir / "reports" / "metrics_summary.json"

    uci_dataset_url: str = (
        "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
    )
    random_seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    optuna_trials: int = 20

    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "credit_default_scoring"

    cost_false_negative: float = 5.0
    cost_false_positive: float = 1.0

    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
