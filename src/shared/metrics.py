"""Metrics and tabular summaries shared by every experiment.

These are intentionally model-agnostic: they operate on a state/cluster
assignment plus (optional) posterior probabilities, so the same functions
score the Gaussian HMM's states and the GMM baseline's clusters.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd


def information_criteria(
    log_likelihood: float,
    n_samples: int,
    n_parameters: int,
) -> tuple[float, float]:
    """Compute AIC and BIC from a log-likelihood and a parameter count."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    aic = -2.0 * log_likelihood + 2.0 * n_parameters
    bic = -2.0 * log_likelihood + math.log(n_samples) * n_parameters
    return aic, bic


def expected_durations(transition_matrix: np.ndarray) -> np.ndarray:
    """Expected state duration: E[D_i] = 1 / (1 - p_ii)."""

    self_probs = np.diag(np.asarray(transition_matrix, dtype=float))
    durations = np.empty_like(self_probs)
    for index, probability in enumerate(self_probs):
        durations[index] = math.inf if probability >= 1.0 else 1.0 / (1.0 - probability)
    return durations


def transition_table(transition_matrix: np.ndarray) -> pd.DataFrame:
    """Tabulate transition probabilities and expected durations per state."""

    matrix = np.asarray(transition_matrix, dtype=float)
    durations = expected_durations(matrix)
    rows: list[dict[str, float | int]] = []
    for from_state in range(matrix.shape[0]):
        row: dict[str, float | int] = {
            "from_state": from_state,
            "self_transition_probability": matrix[from_state, from_state],
            "expected_duration": durations[from_state],
        }
        for to_state in range(matrix.shape[1]):
            row[f"to_state_{to_state}"] = matrix[from_state, to_state]
        rows.append(row)
    return pd.DataFrame(rows)


def empirical_transition_matrix(states: Iterable[int], n_components: int) -> np.ndarray:
    """Transition matrix estimated from an observed state/cluster sequence."""

    states = np.asarray(list(states), dtype=int)
    counts = np.zeros((n_components, n_components), dtype=float)
    for from_state, to_state in zip(states[:-1], states[1:]):
        if 0 <= from_state < n_components and 0 <= to_state < n_components:
            counts[from_state, to_state] += 1.0

    row_sums = counts.sum(axis=1, keepdims=True)
    probabilities = np.zeros_like(counts)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(counts, row_sums, out=probabilities, where=row_sums > 0)
    probabilities[row_sums[:, 0] == 0] = 0.0
    return probabilities


def state_summary(
    df: pd.DataFrame,
    states: Iterable[int],
    posterior: np.ndarray | None,
    *,
    return_column: str,
    volatility_column: str,
    volume_column: str,
) -> pd.DataFrame:
    """Descriptive statistics (return, volatility, volume) per state/cluster."""

    work = df.copy()
    work["state"] = np.asarray(list(states), dtype=int)

    if posterior is not None:
        posterior = np.asarray(posterior, dtype=float)
        assigned_probability = posterior[np.arange(len(work)), work["state"].to_numpy()]
        work["assigned_state_probability"] = assigned_probability
    else:
        work["assigned_state_probability"] = np.nan

    total_rows = len(work)
    summaries: list[dict[str, float | int | str]] = []
    for state, group in work.groupby("state", sort=True):
        return_values = group[return_column].astype(float)
        volatility_values = group[volatility_column].astype(float)
        volume_values = group[volume_column].astype(float)
        summaries.append(
            {
                "state": int(state),
                "observations": int(len(group)),
                "share": float(len(group) / total_rows),
                "mean_log_return": float(return_values.mean()),
                "median_log_return": float(return_values.median()),
                "std_log_return": float(return_values.std(ddof=1)) if len(group) > 1 else 0.0,
                "mean_feature_volatility": float(volatility_values.mean()),
                "mean_volume": float(volume_values.mean()),
                "mean_assigned_probability": float(group["assigned_state_probability"].mean()),
            }
        )

    table = pd.DataFrame(summaries)
    if table.empty:
        return table

    table["financial_profile"] = _label_state_profiles(table)
    return table


def _label_state_profiles(table: pd.DataFrame) -> list[str]:
    mean_return = table["mean_log_return"]
    volatility = table["std_log_return"]
    high_return = mean_return.quantile(0.67)
    low_return = mean_return.quantile(0.33)
    high_vol = volatility.quantile(0.67)

    labels: list[str] = []
    for _, row in table.iterrows():
        if row["mean_log_return"] >= high_return and row["std_log_return"] < high_vol:
            labels.append("positive_return")
        elif row["mean_log_return"] <= low_return and row["std_log_return"] >= high_vol:
            labels.append("negative_high_risk")
        elif row["std_log_return"] >= high_vol:
            labels.append("volatile")
        else:
            labels.append("sideways_or_mixed")
    return labels
