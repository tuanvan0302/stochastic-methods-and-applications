"""Run Viterbi and Forward-Backward inference with the best Gaussian HMM."""

from __future__ import annotations

import argparse

import pandas as pd

from src.gaussian_hmm.config import DEFAULT_CONFIG_PATH, HMMConfig, ensure_output_dirs, load_hmm_config
from src.shared.data_io import feature_matrix, load_market_data
from src.shared.inference import build_inference_tables
from src.shared.model_io import load_model_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer market states with the best Gaussian HMM.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to configs/hmm.yaml.")
    return parser.parse_args()

# Run inference with the best Gaussian HMM model, generating Viterbi state sequences and posterior probabilities. The function loads the trained model artifact, processes the market data to create a feature matrix, and uses the model to predict states and compute posterior probabilities. It then builds inference tables containing the results and saves them to CSV files in the specified output directory.
def run_inference(config: HMMConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    print(f"Wrote Viterbi state sequence to {config.state_sequence_path}")
    print(f"Wrote posterior probabilities to {config.posterior_path}")
    return state_table, posterior_table


def main() -> None:
    args = parse_args()
    config = load_hmm_config(args.config)
    run_inference(config)


if __name__ == "__main__":
    main()
