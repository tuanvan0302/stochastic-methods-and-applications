"""Create GMM baseline figures for model selection and cluster analysis."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.gmm.config import DEFAULT_CONFIG_PATH, GMMConfig, ensure_output_dirs, load_gmm_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GMM baseline outputs.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to configs/gmm.yaml.")
    return parser.parse_args()


def run_visualization(config: GMMConfig) -> list[str]:
    ensure_output_dirs(config)
    paths = [
        config.state_sequence_path,
        config.posterior_path,
        config.empirical_transition_matrix_path,
        config.model_selection_path,
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing GMM output tables. Run train.py and evaluate.py first. "
            f"Missing: {missing}"
        )

    state_table = pd.read_csv(config.state_sequence_path, parse_dates=[config.date_column])
    posterior_table = pd.read_csv(config.posterior_path, parse_dates=[config.date_column])
    empirical_transition = pd.read_csv(config.empirical_transition_matrix_path)
    model_selection = pd.read_csv(config.model_selection_path)

    output_paths = [
        plot_price_states(config, state_table),
        plot_posteriors(config, posterior_table),
        plot_empirical_transition_matrix(config, empirical_transition),
        plot_aic_bic(config, model_selection),
    ]
    for path in output_paths:
        print(f"Wrote figure to {path}")
    return [str(path) for path in output_paths]


def plot_price_states(config: GMMConfig, state_table: pd.DataFrame):
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
            label=f"Cluster {state}",
        )
    ax.set_title("GMM Clusters on Close Price")
    ax.set_xlabel("Date")
    ax.set_ylabel(config.price_column)
    ax.legend(loc="best", ncols=2)
    fig.tight_layout()
    output_path = config.figures_dir / "price_states.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_posteriors(config: GMMConfig, posterior_table: pd.DataFrame):
    posterior_columns = [column for column in posterior_table.columns if column.startswith("posterior_state_")]
    fig, ax = plt.subplots(figsize=(14, 6))
    for column in posterior_columns:
        ax.plot(posterior_table[config.date_column], posterior_table[column], linewidth=1.0, label=column)
    ax.set_title("GMM Posterior Cluster Probabilities")
    ax.set_xlabel("Date")
    ax.set_ylabel("Posterior probability")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="best", ncols=2)
    fig.tight_layout()
    output_path = config.figures_dir / "posterior_probabilities.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_empirical_transition_matrix(config: GMMConfig, transition: pd.DataFrame):
    matrix_columns = [column for column in transition.columns if column.startswith("to_state_")]
    matrix = transition.loc[:, matrix_columns].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax)
    ax.set_title("GMM Empirical Transition Matrix\n(no transition structure is learned)")
    ax.set_xlabel("To cluster")
    ax.set_ylabel("From cluster")
    fig.tight_layout()
    output_path = config.figures_dir / "empirical_transition_matrix.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_aic_bic(config: GMMConfig, model_selection: pd.DataFrame):
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
    axes[0].set_title("AIC by Component Count")
    axes[1].set_title("BIC by Component Count")
    for ax in axes:
        ax.set_xlabel("Number of components")
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
    config = load_gmm_config(args.config)
    run_visualization(config)


if __name__ == "__main__":
    main()
