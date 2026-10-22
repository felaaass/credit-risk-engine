import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split

from src.config import Settings, get_settings


class CreditDataPreprocessor:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def clean(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.when(pl.col("education").is_in([0, 5, 6]))
            .then(pl.lit(4))
            .otherwise(pl.col("education"))
            .cast(pl.Int64)
            .alias("education"),
            pl.when(pl.col("marriage") == 0)
            .then(pl.lit(3))
            .otherwise(pl.col("marriage"))
            .cast(pl.Int64)
            .alias("marriage"),
            pl.col("sex").cast(pl.Int64),
            pl.col("age").cast(pl.Int64),
            pl.col("limit_bal").cast(pl.Float64),
        )

    def split_train_test(
        self, df: pl.DataFrame, target_col: str = "target"
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        pdf = df.to_pandas()
        features = pdf.drop(columns=[target_col])
        target = pdf[target_col].astype(int)

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=self.settings.test_size,
            random_state=self.settings.random_seed,
            stratify=target,
        )
        return x_train, x_test, y_train, y_test
