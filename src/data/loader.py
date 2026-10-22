import io
import zipfile
from pathlib import Path

import pandas as pd
import polars as pl
import requests

from src.config import Settings, get_settings
from src.exceptions import DataIngestionError, DataValidationError

REQUIRED_COLUMNS: tuple[str, ...] = (
    "limit_bal",
    "sex",
    "education",
    "marriage",
    "age",
    "pay_0",
    "pay_2",
    "pay_3",
    "pay_4",
    "pay_5",
    "pay_6",
    "bill_amt1",
    "bill_amt2",
    "bill_amt3",
    "bill_amt4",
    "bill_amt5",
    "bill_amt6",
    "pay_amt1",
    "pay_amt2",
    "pay_amt3",
    "pay_amt4",
    "pay_amt5",
    "pay_amt6",
    "target",
)


class CreditDataLoader:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def load_raw_data(self, force_download: bool = False) -> pl.DataFrame:
        raw_path: Path = self.settings.raw_data_path
        if raw_path.exists() and not force_download:
            df = pl.read_parquet(raw_path)
            self._validate_dataframe(df)
            return df

        df = self._download_from_uci()
        self._validate_dataframe(df)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(raw_path)
        return df

    def _download_from_uci(self) -> pl.DataFrame:
        try:
            response = requests.get(self.settings.uci_dataset_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise DataIngestionError(
                f"Failed to download UCI dataset from {self.settings.uci_dataset_url}: {exc}"
            ) from exc

        try:
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                excel_files = [name for name in archive.namelist() if name.endswith(".xls")]
                if not excel_files:
                    raise DataIngestionError("No Excel file found inside UCI dataset archive.")
                with archive.open(excel_files[0]) as excel_stream:
                    pdf = pd.read_excel(excel_stream, header=1, engine="calamine")
        except Exception as exc:
            if isinstance(exc, DataIngestionError):
                raise
            raise DataIngestionError(f"Failed to parse UCI dataset archive: {exc}") from exc

        rename_map = {col: col.strip().lower() for col in pdf.columns}
        rename_map["default payment next month"] = "target"
        pdf = pdf.rename(columns=rename_map)

        if "id" in pdf.columns:
            pdf = pdf.drop(columns=["id"])

        return pl.from_pandas(pdf)

    def _validate_dataframe(self, df: pl.DataFrame) -> None:
        if df.is_empty():
            raise DataValidationError("Loaded dataset is empty.")

        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise DataValidationError(f"Dataset is missing required columns: {missing_cols}")

        null_count = sum(df.null_count().row(0))
        if null_count > 0:
            raise DataValidationError(f"Dataset contains {null_count} unexpected null values.")
