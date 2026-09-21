"""Train and select Gaussian HMM market-regime models."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.gaussian_hmm.config import DEFAULT_CONFIG_PATH, HMMConfig, ensure_output_dirs, load_hmm_config
from src.gaussian_hmm.metrics import count_hmm_parameters
from src.shared.data_io import feature_matrix, load_market_data, split_frame
from src.shared.metrics import information_criteria
from src.shared.model_io import save_model_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Gaussian HMM market-regime models.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to configs/hmm.yaml.")
    return parser.parse_args()

# Create a Gaussian HMM model using the specified configuration, number of components, covariance type, and random seed.
# The function checks for the presence of the hmmlearn library and raises an error if it is not installed.
def create_gaussian_hmm(
    config: HMMConfig,
    *,
    n_components: int,
    covariance_type: str,
    seed: int,
):
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as exc:  # pragma: no cover - exercised before deps install
        raise RuntimeError("hmmlearn is required to train Gaussian HMM models.") from exc

    return GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=config.n_iter,
        tol=config.tol,
        min_covar=config.min_covar,
        algorithm=config.algorithm,
        random_state=seed,
        init_params=config.init_params,
        params=config.params,
    )

# Training and model selection for Gaussian HMM market-regime models.
# The function loads the market data, splits it into training and validation sets, fits a grid of models with different hyperparameters, evaluates their performance, and selects the best model based on the specified primary metric.
# The best model is then refitted on the final dataset and saved as an artifact.
def run_training(config: HMMConfig) -> pd.DataFrame:
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
    print(f"Wrote best Gaussian HMM artifact to {config.best_model_path}")
    return results

# Fit a grid of Gaussian HMM models with different hyperparameters and evaluate their performance on the training and validation datasets.
def fit_model_grid(
    config: HMMConfig,
    X_train: np.ndarray,
    X_validation: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n_features = X_train.shape[1]

    for n_components in config.n_components:
        for covariance_type in config.covariance_types:
            for seed in config.seeds:
                row = {
                    "n_components": n_components,
                    "covariance_type": covariance_type,
                    "seed": seed,
                }
                try:
                    model = create_gaussian_hmm(
                        config,
                        n_components=n_components,
                        covariance_type=covariance_type,
                        seed=seed,
                    )
                    model.fit(X_train)
                    train_log_likelihood = float(model.score(X_train))
                    validation_log_likelihood = float(model.score(X_validation))
                    n_parameters = count_hmm_parameters(n_components, n_features, covariance_type)
                    aic, bic = information_criteria(
                        train_log_likelihood,
                        n_samples=X_train.shape[0],
                        n_parameters=n_parameters,
                    )
                    train_states = model.predict(X_train)
                    state_counts = np.bincount(train_states, minlength=n_components)
                    state_shares = state_counts / state_counts.sum()

                    row.update(
                        {
                            "status": "ok",
                            "train_log_likelihood": train_log_likelihood,
                            "train_avg_log_likelihood": train_log_likelihood / X_train.shape[0],
                            "validation_log_likelihood": validation_log_likelihood,
                            "validation_avg_log_likelihood": validation_log_likelihood
                            / X_validation.shape[0],
                            "n_parameters": n_parameters,
                            "aic": aic,
                            "bic": bic,
                            "converged": bool(model.monitor_.converged),
                            "iterations": int(model.monitor_.iter),
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
                            "n_parameters": math.nan,
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

# Select the best result from the model grid search based on the specified primary metric and other criteria.
# The function filters the results to include only successful runs, checks for convergence and minimum state share, and sorts the eligible results to find the best one.
def choose_best_result(results: pd.DataFrame, config: HMMConfig) -> dict[str, Any]:
    ok = results.loc[results["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError("All Gaussian HMM training runs failed.")

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

# Refit the best Gaussian HMM model on the final dataset using the specified configuration and the best hyperparameters.
def refit_best_model(
    config: HMMConfig,
    df: pd.DataFrame,
    best_result: dict[str, Any],
):
    final_df = split_frame(df, config.split_column, config.final_fit_splits)
    if final_df.empty:
        final_df = split_frame(df, config.split_column, config.train_splits)
    X_final = feature_matrix(final_df, config.feature_columns)

    model = create_gaussian_hmm(
        config,
        n_components=int(best_result["n_components"]),
        covariance_type=str(best_result["covariance_type"]),
        seed=int(best_result["seed"]),
    )
    model.fit(X_final)
    return model, float(model.score(X_final))

# Write a training summary to a JSON file, including information about the dataset, model selection results, best model parameters, and final fit log-likelihood. The summary is saved to the specified path in the configuration.
def write_training_summary(
    *,
    config: HMMConfig,
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
    config = load_hmm_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
