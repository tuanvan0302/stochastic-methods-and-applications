"""Persistence helpers for model artifacts, shared by every experiment."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def ensure_output_dirs(config: Any) -> None:
    config.model_dir.mkdir(parents=True, exist_ok=True)
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)


def save_model_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(artifact, file)


def load_model_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}. Run train.py before inference/evaluation."
        )
    with path.open("rb") as file:
        artifact = pickle.load(file)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(f"Invalid model artifact: {path}")
    return artifact
