"""Run inference and evaluate the GMM baseline's walk-forward behavior.

GMM has no learned transition structure between components, so unlike the
Gaussian HMM there is no transition_matrix step here: only the empirical
transition matrix computed post-hoc from the predicted cluster sequence.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.gmm.config import DEFAULT_CONFIG_PATH, GMMConfig, ensure_output_dirs, load_gmm_config
from src.gmm.train import create_gmm
from src.shared.data_io import feature_matrix, load_market_data
from src.shared.inference import build_inference_tables
from src.shared.metrics import empirical_transition_matrix, state_summary, transition_table
from src.shared.model_io import load_model_artifact
from src.shared.splits import iter_walk_forward_folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GMM baseline market regimes.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to configs/gmm.yaml.")
    return parser.parse_args()


def run_inference(config: GMMConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_output_dirs(config)
    artifact = load_model_artifact(config.best_model_path)
    model = artifact["model"]
    feature_columns = artifact.get("feature_columns", config.feature_columns)

    df = load_market_data(config)
    X = feature_matrix(df, feature_columns)
    states = model.predict(X)
    posterior = model.predict_proba(X)

    state_table, posterior_table = build_inference_tables(df, states, posterior, config)
    state_table.to_csv(config.state_sequence_path, index=False)
    posterior_table.to_csv(config.posterior_path, index=False)

    print(f"Wrote cluster assignment sequence to {config.state_sequence_path}")
    print(f"Wrote posterior probabilities to {config.posterior_path}")
    return state_table, posterior_table


def run_evaluation(config: GMMConfig) -> dict[str, pd.DataFrame]:
    ensure_output_dirs(config)
    artifact = load_model_artifact(config.best_model_path)
    model = artifact["model"]

    if config.state_sequence_path.exists() and config.posterior_path.exists():
        state_table = pd.read_csv(config.state_sequence_path)
        posterior_table = pd.read_csv(config.posterior_path)
    else:
        state_table, posterior_table = run_inference(config)

    df = load_market_data(config)
    states = state_table["state"].to_numpy(dtype=int)
    posterior = posterior_table.filter(like="posterior_state_").to_numpy(dtype=float)

    empirical_transition = transition_table(
        empirical_transition_matrix(states, model.n_components)
    )
    summary = state_summary(
        df,
        states,
        posterior,
        return_column=config.return_column,
        volatility_column=config.volatility_column,
        volume_column=config.volume_column,
    )
    walk_forward = walk_forward_evaluation(config, df, artifact)

    empirical_transition.to_csv(config.empirical_transition_matrix_path, index=False)
    summary.to_csv(config.state_summary_path, index=False)
    walk_forward.to_csv(config.walk_forward_path, index=False)

    print(f"Wrote empirical transition matrix to {config.empirical_transition_matrix_path}")
    print(f"Wrote state summary to {config.state_summary_path}")
    print(f"Wrote walk-forward results to {config.walk_forward_path}")

    return {
        "empirical_transition_matrix": empirical_transition,
        "state_summary": summary,
        "walk_forward_results": walk_forward,
    }


def walk_forward_evaluation(config: GMMConfig, df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    best = artifact["best_result"]
    n_components = int(best["n_components"])
    covariance_type = str(best["covariance_type"])
    seed = int(best["seed"])
    rows: list[dict[str, object]] = []

    for fold, train_index, eval_index in iter_walk_forward_folds(df, config.fold_column):
        train_df = df.loc[train_index]
        eval_df = df.loc[eval_index]
        if train_df.empty or eval_df.empty:
            continue

        X_train = feature_matrix(train_df, config.feature_columns)
        X_eval = feature_matrix(eval_df, config.feature_columns)

        model = create_gmm(
            config,
            n_components=n_components,
            covariance_type=covariance_type,
            seed=seed,
        )
        try:
            model.fit(X_train)
            log_likelihood = float(model.score(X_eval)) * X_eval.shape[0]
            fold_states = model.predict(X_eval)
            fold_posterior = model.predict_proba(X_eval)
            fold_summary = state_summary(
                eval_df,
                fold_states,
                fold_posterior,
                return_column=config.return_column,
                volatility_column=config.volatility_column,
                volume_column=config.volume_column,
            )
            for _, state_row in fold_summary.iterrows():
                rows.append(
                    {
                        "fold": fold,
                        "status": "ok",
                        "train_rows": int(len(train_df)),
                        "eval_rows": int(len(eval_df)),
                        "eval_split_values": "+".join(
                            sorted(str(value) for value in eval_df[config.split_column].unique())
                        ),
                        "eval_start": eval_df[config.date_column].iloc[0],
                        "eval_end": eval_df[config.date_column].iloc[-1],
                        "log_likelihood": log_likelihood,
                        "avg_log_likelihood": log_likelihood / len(eval_df),
                        "converged": bool(model.converged_),
                        **state_row.to_dict(),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - later folds can still be useful
            rows.append(
                {
                    "fold": fold,
                    "status": "failed",
                    "train_rows": int(len(train_df)),
                    "eval_rows": int(len(eval_df)),
                    "eval_split_values": "+".join(
                        sorted(str(value) for value in eval_df[config.split_column].unique())
                    ),
                    "eval_start": eval_df[config.date_column].iloc[0],
                    "eval_end": eval_df[config.date_column].iloc[-1],
                    "error": str(exc),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = load_gmm_config(args.config)
    run_evaluation(config)


if __name__ == "__main__":
    main()
