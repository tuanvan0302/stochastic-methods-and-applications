"""Build inference output tables shared by every experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.shared.data_io import MarketDataConfig


def build_inference_tables(
    df: pd.DataFrame,
    states: np.ndarray,
    posterior: np.ndarray,
    config: MarketDataConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble the state-sequence table and the per-state posterior table."""

    states = np.asarray(states, dtype=int)
    posterior = np.asarray(posterior, dtype=float)
    if len(df) != len(states) or posterior.shape[0] != len(df):
        raise ValueError("Inference outputs must have the same number of rows as input data.")

    base_columns = [
        config.date_column,
        config.split_column,
        config.fold_column,
        config.price_column,
        config.return_column,
        config.volatility_column,
        config.volume_column,
    ]

    state_table = df.loc[:, base_columns].copy()
    state_table["state"] = states
    state_table["state_probability"] = posterior[np.arange(len(states)), states]

    posterior_table = df.loc[:, [config.date_column, config.split_column, config.fold_column]].copy()
    for state in range(posterior.shape[1]):
        posterior_table[f"posterior_state_{state}"] = posterior[:, state]

    return state_table, posterior_table
