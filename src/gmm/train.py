"""Train and select GMM baseline market-regime models."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.gmm.config import DEFAULT_CONFIG_PATH, GMMConfig, ensure_output_dirs, load_gmm_config
from src.shared.data_io import feature_matrix, load_market_data, split_frame
from src.shared.model_io import save_model_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GMM baseline market-regime models.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to configs/gmm.yaml.")
    return parser.parse_args()


def create_gmm(config: GMMConfig, *, n_components: int, covariance_type: str, seed: int):
    from sklearn.mixture import GaussianMixture

    return GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=config.max_iter,
        tol=config.tol,
        n_init=config.n_init,
        random_state=seed,
    )


def run_training(config: GMMConfig) -> pd.DataFrame:
    ensure_output_dirs(config)
    df = load_market_data(config)
    train_df = split_frame(df, config.split_column, config.train_splits)
    validation_df = split_frame(df, config.split_column, config.validation_splits)

    X_train = feature_matrix(train_df, config.feature_columns)
    X_validation = feature_matrix(validation_df, config.feature_columns)

    results = fit_model_grid(config, X_train, X_validation)
    results.to_csv(config.model_selection_path, index=False)

    best_result = choose_best_result(results, config)
    final_model, final_log_likelihood = refit_best_model(config, df, best_result)

    artifact = {
        "model": final_model,
        "feature_columns": config.feature_columns,
        "best_result": _json_safe(best_result),
        "final_fit_splits": config.final_fit_splits,
        "final_fit_log_likelihood": final_log_likelihood,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_model_artifact(config.best_model_path, artifact)

    write_training_summary(
        config=config,
        data=df,
        results=results,
        best_result=best_result,
        final_log_likelihood=final_log_likelihood,
    )

    print(f"Wrote model selection table to {config.model_selection_path}")
    print(f"Wrote best GMM artifact to {config.best_model_path}")
    return results


def fit_model_grid(config: GMMConfig, X_train: np.ndarray, X_validation: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for n_components in config.n_components:
        for covariance_type in config.covariance_types:
            for seed in config.seeds:
                row = {
                    "n_components": n_components,
                    "covariance_type": covariance_type,
                    "seed": seed,
                }
                try:
                    model = create_gmm(
                        config,
                        n_components=n_components,
                        covariance_type=covariance_type,
                        seed=seed,
                    )
                    model.fit(X_train)
                    train_log_likelihood = float(model.score(X_train)) * X_train.shape[0]
                    validation_log_likelihood = float(model.score(X_validation)) * X_validation.shape[0]
                    aic = float(model.aic(X_train))
                    bic = float(model.bic(X_train))
                    train_labels = model.predict(X_train)
                    state_counts = np.bincount(train_labels, minlength=n_components)
                    state_shares = state_counts / state_counts.sum()

                    row.update(
                        {
                            "status": "ok",
                            "train_log_likelihood": train_log_likelihood,
                            "train_avg_log_likelihood": train_log_likelihood / X_train.shape[0],
                            "validation_log_likelihood": validation_log_likelihood,
                            "validation_avg_log_likelihood": validation_log_likelihood
                            / X_validation.shape[0],
                            "aic": aic,
                            "bic": bic,
                            "converged": bool(model.converged_),
                            "iterations": int(model.n_iter_),
                            "min_state_share": float(state_shares.min()),
                            "max_state_share": float(state_shares.max()),
                            "state_counts": json.dumps(state_counts.tolist()),
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - grid search should keep running
                    row.update(
                        {
                            "status": "failed",
                            "error": str(exc),
                            "train_log_likelihood": math.nan,
                            "train_avg_log_likelihood": math.nan,
                            "validation_log_likelihood": math.nan,
                            "validation_avg_log_likelihood": math.nan,
                            "aic": math.nan,
                            "bic": math.nan,
                            "converged": False,
                            "iterations": 0,
                            "min_state_share": math.nan,
                            "max_state_share": math.nan,
                            "state_counts": "[]",
                        }
                    )
                rows.append(row)

    return pd.DataFrame(rows)


def choose_best_result(results: pd.DataFrame, config: GMMConfig) -> dict[str, Any]:
    ok = results.loc[results["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError("All GMM training runs failed.")

    eligible = ok.loc[
        (ok["converged"].astype(bool))
        & (ok["min_state_share"].astype(float) >= config.minimum_state_share)
    ].copy()
    if eligible.empty:
        eligible = ok

    eligible = eligible.sort_values(
        by=[config.primary_metric, "validation_log_likelihood"],
        ascending=[True, False],
    )
    return eligible.iloc[0].to_dict()


def refit_best_model(config: GMMConfig, df: pd.DataFrame, best_result: dict[str, Any]):
    final_df = split_frame(df, config.split_column, config.final_fit_splits)
    if final_df.empty:
        final_df = split_frame(df, config.split_column, config.train_splits)
    X_final = feature_matrix(final_df, config.feature_columns)

    model = create_gmm(
        config,
        n_components=int(best_result["n_components"]),
        covariance_type=str(best_result["covariance_type"]),
        seed=int(best_result["seed"]),
    )
    model.fit(X_final)
    return model, float(model.score(X_final)) * X_final.shape[0]


def write_training_summary(
    *,
    config: GMMConfig,
    data: pd.DataFrame,
    results: pd.DataFrame,
    best_result: dict[str, Any],
    final_log_likelihood: float,
) -> None:
    split_counts = data[config.split_column].value_counts().to_dict()
    summary = {
        "data_path": str(config.data_path),
        "rows": int(len(data)),
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
        "feature_columns": config.feature_columns,
        "grid_size": int(len(results)),
        "primary_metric": config.primary_metric,
        "minimum_state_share": config.minimum_state_share,
        "best_result": _json_safe(best_result),
        "final_fit_splits": config.final_fit_splits,
        "final_fit_log_likelihood": final_log_likelihood,
        "model_path": str(config.best_model_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    config.training_summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    args = parse_args()
    config = load_gmm_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
