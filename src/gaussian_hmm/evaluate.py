"""Evaluate Gaussian HMM regimes and walk-forward behavior."""

from __future__ import annotations

import argparse

import pandas as pd

from src.gaussian_hmm.config import DEFAULT_CONFIG_PATH, HMMConfig, ensure_output_dirs, load_hmm_config
from src.gaussian_hmm.inference import run_inference
from src.gaussian_hmm.train import create_gaussian_hmm
from src.shared.data_io import feature_matrix, load_market_data
from src.shared.metrics import empirical_transition_matrix, state_summary, transition_table
from src.shared.model_io import load_model_artifact
from src.shared.splits import iter_walk_forward_folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Gaussian HMM market regimes.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to configs/hmm.yaml.")
    return parser.parse_args()


def run_evaluation(config: HMMConfig) -> dict[str, pd.DataFrame]:
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

    transition = transition_table(model.transmat_)
    empirical_transition = transition_table(empirical_transition_matrix(states, model.n_components))
    summary = state_summary(
        df,
        states,
        posterior,
        return_column=config.return_column,
        volatility_column=config.volatility_column,
        volume_column=config.volume_column,
    )
    walk_forward = walk_forward_evaluation(config, df, artifact)

    transition.to_csv(config.transition_matrix_path, index=False)
    empirical_transition.to_csv(config.empirical_transition_matrix_path, index=False)
    summary.to_csv(config.state_summary_path, index=False)
    walk_forward.to_csv(config.walk_forward_path, index=False)

    print(f"Wrote transition matrix to {config.transition_matrix_path}")
    print(f"Wrote state summary to {config.state_summary_path}")
    print(f"Wrote walk-forward results to {config.walk_forward_path}")

    return {
        "transition_matrix": transition,
        "empirical_transition_matrix": empirical_transition,
        "state_summary": summary,
        "walk_forward_results": walk_forward,
    }


def walk_forward_evaluation(
    config: HMMConfig,
    df: pd.DataFrame,
    artifact: dict,
) -> pd.DataFrame:
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

        model = create_gaussian_hmm(
            config,
            n_components=n_components,
            covariance_type=covariance_type,
            seed=seed,
        )
        try:
            model.fit(X_train)
            log_likelihood = float(model.score(X_eval))
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
                        "converged": bool(model.monitor_.converged),
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
    config = load_hmm_config(args.config)
    run_evaluation(config)


if __name__ == "__main__":
    main()
