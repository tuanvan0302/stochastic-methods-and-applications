"""Create Gaussian HMM figures for model selection and regime analysis."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.gaussian_hmm.config import DEFAULT_CONFIG_PATH, HMMConfig, ensure_output_dirs, load_hmm_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Gaussian HMM outputs.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to configs/hmm.yaml.")
    return parser.parse_args()

# Run the visualization process for Gaussian HMM outputs, generating figures for price states, posterior probabilities, transition matrix, and model selection criteria. The function checks for the existence of required output tables and raises an error if any are missing. It reads the necessary data from CSV files, creates plots using matplotlib and seaborn, saves the figures to the specified output directory, and returns a list of paths to the generated figures.
def run_visualization(config: HMMConfig) -> list[str]:
    ensure_output_dirs(config)
    paths = [
        config.state_sequence_path,
        config.posterior_path,
        config.transition_matrix_path,
        config.model_selection_path,
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Gaussian HMM output tables. Run train.py, inference.py, and evaluate.py first. "
            f"Missing: {missing}"
        )

    state_table = pd.read_csv(config.state_sequence_path, parse_dates=[config.date_column])
    posterior_table = pd.read_csv(config.posterior_path, parse_dates=[config.date_column])
    transition = pd.read_csv(config.transition_matrix_path)
    model_selection = pd.read_csv(config.model_selection_path)

    output_paths = [
        plot_price_states(config, state_table),
        plot_posteriors(config, posterior_table),
        plot_transition_matrix(config, transition),
        plot_aic_bic(config, model_selection),
    ]
    for path in output_paths:
        print(f"Wrote figure to {path}")
    return [str(path) for path in output_paths]

# Plot the close price with Gaussian HMM Viterbi states overlaid. 
# The function creates a scatter plot of the close price, coloring points by their assigned state, and saves the figure to the specified output directory.
def plot_price_states(config: HMMConfig, state_table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        state_table[config.date_column],
        state_table[config.price_column],
        color="0.75",
        linewidth=1.0,
        label=config.price_column,
    )
    palette = sns.color_palette("tab10", n_colors=state_table["state"].nunique())
    for color_index, state in enumerate(sorted(state_table["state"].unique())):
        subset = state_table.loc[state_table["state"] == state]
        ax.scatter(
            subset[config.date_column],
            subset[config.price_column],
            s=8,
            color=palette[color_index],
            label=f"State {state}",
        )
    ax.set_title("Gaussian HMM Viterbi States on Close Price")
    ax.set_xlabel("Date")
    ax.set_ylabel(config.price_column)
    ax.legend(loc="best", ncols=2)
    fig.tight_layout()
    output_path = config.figures_dir / "price_states.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path

# Plot the posterior probabilities of Gaussian HMM states over time.
# The function creates a line plot of the posterior probabilities for each state, with the x-axis representing time and the y-axis representing the probability, and saves the figure to the specified output directory.
def plot_posteriors(config: HMMConfig, posterior_table: pd.DataFrame):
    posterior_columns = [column for column in posterior_table.columns if column.startswith("posterior_state_")]
    fig, ax = plt.subplots(figsize=(14, 6))
    for column in posterior_columns:
        ax.plot(posterior_table[config.date_column], posterior_table[column], linewidth=1.0, label=column)
    ax.set_title("Gaussian HMM Posterior State Probabilities")
    ax.set_xlabel("Date")
    ax.set_ylabel("Posterior probability")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="best", ncols=2)
    fig.tight_layout()
    output_path = config.figures_dir / "posterior_probabilities.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path

# Plot the transition matrix of the Gaussian HMM as a heatmap.
# The function creates a heatmap of the transition probabilities between states, with the x-axis representing the "to" states and the y-axis representing the "from" states, and saves the figure to the specified output directory.
def plot_transition_matrix(config: HMMConfig, transition: pd.DataFrame):
    matrix_columns = [column for column in transition.columns if column.startswith("to_state_")]
    matrix = transition.loc[:, matrix_columns].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax)
    ax.set_title("Gaussian HMM Transition Matrix")
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    fig.tight_layout()
    output_path = config.figures_dir / "transition_matrix.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path

# Plot the AIC and BIC values from the Gaussian HMM model selection results.
# The function creates two line plots, one for AIC and one for BIC, showing how these criteria vary with the number of states and covariance types, and saves the figure to the specified output directory.
def plot_aic_bic(config: HMMConfig, model_selection: pd.DataFrame):
    ok = model_selection.loc[model_selection["status"] == "ok"].copy()
    if ok.empty:
        raise ValueError("No successful model-selection rows to plot.")

    summary = (
        ok.groupby(["n_components", "covariance_type"], as_index=False)
        .agg(aic=("aic", "min"), bic=("bic", "min"))
        .sort_values(["covariance_type", "n_components"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for covariance_type, group in summary.groupby("covariance_type"):
        axes[0].plot(group["n_components"], group["aic"], marker="o", label=covariance_type)
        axes[1].plot(group["n_components"], group["bic"], marker="o", label=covariance_type)
    axes[0].set_title("AIC by State Count")
    axes[1].set_title("BIC by State Count")
    for ax in axes:
        ax.set_xlabel("Number of states")
        ax.set_ylabel("Criterion value")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path = config.figures_dir / "aic_bic.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    config = load_hmm_config(args.config)
    run_visualization(config)


if __name__ == "__main__":
    main()
