"""Configuration handling for the GMM baseline experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.shared.model_io import ensure_output_dirs as ensure_output_dirs

DEFAULT_CONFIG_PATH = Path("configs/gmm.yaml")


@dataclass(frozen=True)
class GMMConfig:
    config_path: Path
    project_root: Path
    data_path: Path
    metadata_path: Path | None
    feature_columns: list[str]
    date_column: str
    split_column: str
    fold_column: str
    price_column: str
    return_column: str
    volatility_column: str
    volume_column: str
    train_splits: list[str]
    validation_splits: list[str]
    final_fit_splits: list[str]
    n_components: list[int]
    covariance_types: list[str]
    seeds: list[int]
    n_init: int
    max_iter: int
    tol: float
    primary_metric: str
    minimum_state_share: float
    model_dir: Path
    tables_dir: Path
    figures_dir: Path
    best_model_file: str
    training_summary_file: str
    model_selection_file: str
    state_sequence_file: str
    posterior_file: str
    empirical_transition_matrix_file: str
    state_summary_file: str
    walk_forward_file: str

    @property
    def best_model_path(self) -> Path:
        return self.model_dir / self.best_model_file

    @property
    def training_summary_path(self) -> Path:
        return self.tables_dir / self.training_summary_file

    @property
    def model_selection_path(self) -> Path:
        return self.tables_dir / self.model_selection_file

    @property
    def state_sequence_path(self) -> Path:
        return self.tables_dir / self.state_sequence_file

    @property
    def posterior_path(self) -> Path:
        return self.tables_dir / self.posterior_file

    @property
    def empirical_transition_matrix_path(self) -> Path:
        return self.tables_dir / self.empirical_transition_matrix_file

    @property
    def state_summary_path(self) -> Path:
        return self.tables_dir / self.state_summary_file

    @property
    def walk_forward_path(self) -> Path:
        return self.tables_dir / self.walk_forward_file


def load_gmm_config(path: str | Path = DEFAULT_CONFIG_PATH) -> GMMConfig:
    """Load the YAML config and resolve all project-relative paths."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised before deps install
        raise RuntimeError("PyYAML is required to read configs/gmm.yaml.") from exc

    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    project_root = _infer_project_root(config_path)
    data = _section(raw, "data")
    features = _section(raw, "features")
    splits = _section(raw, "splits")
    model_grid = _section(raw, "model_grid")
    gmm = _section(raw, "gmm")
    selection = _section(raw, "selection")
    outputs = _section(raw, "outputs")

    feature_columns = _required_list(features, "columns")
    primary_metric = str(selection.get("primary_metric", "bic")).lower()
    if primary_metric not in {"aic", "bic"}:
        raise ValueError("selection.primary_metric must be either 'aic' or 'bic'.")

    return GMMConfig(
        config_path=config_path,
        project_root=project_root,
        data_path=_resolve_path(project_root, data.get("path", "data/processed/market_features.csv")),
        metadata_path=_optional_path(project_root, data.get("metadata_path")),
        feature_columns=feature_columns,
        date_column=str(data.get("date_column", "Date")),
        split_column=str(data.get("split_column", "split")),
        fold_column=str(data.get("fold_column", "walk_forward_fold")),
        price_column=str(data.get("price_column", "Close")),
        return_column=str(data.get("return_column", "log_return")),
        volatility_column=str(data.get("volatility_column", "volatility_20")),
        volume_column=str(data.get("volume_column", "Volume")),
        train_splits=_string_list(splits.get("train", ["train"])),
        validation_splits=_string_list(splits.get("validation", ["validation"])),
        final_fit_splits=_string_list(splits.get("final_fit", ["train", "validation"])),
        n_components=_int_list(model_grid.get("n_components", [2, 3, 4, 5])),
        covariance_types=_string_list(model_grid.get("covariance_types", ["diag", "full"])),
        seeds=_int_list(model_grid.get("seeds", [0, 1, 2, 3, 4])),
        n_init=int(gmm.get("n_init", 5)),
        max_iter=int(gmm.get("max_iter", 300)),
        tol=float(gmm.get("tol", 0.0001)),
        primary_metric=primary_metric,
        minimum_state_share=float(selection.get("minimum_state_share", 0.01)),
        model_dir=_resolve_path(project_root, outputs.get("model_dir", "models/gmm")),
        tables_dir=_resolve_path(project_root, outputs.get("tables_dir", "reports/tables/gmm")),
        figures_dir=_resolve_path(project_root, outputs.get("figures_dir", "reports/figures/gmm")),
        best_model_file=str(outputs.get("best_model_file", "best_gmm.pkl")),
        training_summary_file=str(outputs.get("training_summary_file", "training_summary.json")),
        model_selection_file=str(outputs.get("model_selection_file", "model_selection.csv")),
        state_sequence_file=str(outputs.get("state_sequence_file", "state_sequence.csv")),
        posterior_file=str(outputs.get("posterior_file", "posterior_probabilities.csv")),
        empirical_transition_matrix_file=str(
            outputs.get("empirical_transition_matrix_file", "empirical_transition_matrix.csv")
        ),
        state_summary_file=str(outputs.get("state_summary_file", "state_summary.csv")),
        walk_forward_file=str(outputs.get("walk_forward_file", "walk_forward_results.csv")),
    )


def _infer_project_root(config_path: Path) -> Path:
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return Path.cwd().resolve()


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _optional_path(project_root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return _resolve_path(project_root, str(value))


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    section = raw.get(key, {})
    if not isinstance(section, dict):
        raise ValueError(f"Config section {key!r} must be a mapping.")
    return section


def _required_list(section: dict[str, Any], key: str) -> list[str]:
    if key not in section:
        raise ValueError(f"Missing required config list: {key}.")
    return _string_list(section[key])


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or not value:
        raise ValueError("Expected a non-empty list of strings.")
    return [str(item) for item in value]


def _int_list(value: Any) -> list[int]:
    if isinstance(value, int):
        return [value]
    if not isinstance(value, list) or not value:
        raise ValueError("Expected a non-empty list of integers.")
    return [int(item) for item in value]
