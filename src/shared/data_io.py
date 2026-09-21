"""Data loading helpers shared by every experiment (Gaussian HMM, GMM)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


class MarketDataConfig(Protocol):
    """Structural type for the config attributes these helpers need."""

    data_path: Path
    feature_columns: list[str]
    date_column: str
    split_column: str
    fold_column: str
    price_column: str
    return_column: str
    volatility_column: str
    volume_column: str


def load_market_data(config: MarketDataConfig) -> pd.DataFrame:
    """Load the shared processed dataset without modifying it."""

    if not config.data_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {config.data_path}")

    df = pd.read_csv(config.data_path)
    required_columns = {
        config.date_column,
        config.split_column,
        config.fold_column,
        config.price_column,
        config.return_column,
        config.volatility_column,
        config.volume_column,
        *config.feature_columns,
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Processed data is missing required columns: {missing}")

    df = df.copy()
    df[config.date_column] = pd.to_datetime(df[config.date_column])
    df[config.fold_column] = pd.to_numeric(df[config.fold_column], errors="coerce")
    df = df.sort_values(config.date_column).reset_index(drop=True)
    return df


def feature_matrix(df: pd.DataFrame, feature_columns: Iterable[str]) -> np.ndarray:
    """Convert a DataFrame to a numpy feature matrix, validating it is well-formed."""

    X = df.loc[:, list(feature_columns)].astype(float).to_numpy()
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("Feature matrix must have at least one row and one column.")
    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains NaN or infinite values.")
    return X


def split_frame(df: pd.DataFrame, split_column: str, split_names: Iterable[str]) -> pd.DataFrame:
    """Select the rows belonging to the given named splits."""

    split_names = set(split_names)
    return df.loc[df[split_column].isin(split_names)].copy()
