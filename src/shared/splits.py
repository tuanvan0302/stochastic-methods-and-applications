"""Walk-forward split logic shared by every experiment.

Both the Gaussian HMM model and the GMM baseline must use the exact same
fold boundaries so their out-of-sample results are comparable.
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


def iter_walk_forward_folds(df: pd.DataFrame, fold_column: str) -> Iterator[tuple[int, pd.Index, pd.Index]]:
    fold_values = sorted(
        int(value)
        for value in df[fold_column].dropna().unique()
        if str(value).strip() != ""
    )
    for fold in fold_values:
        eval_index = df.index[df[fold_column] == fold]
        if eval_index.empty:
            continue
        first_eval_index = int(eval_index.min())
        train_index = df.index[df.index < first_eval_index]
        yield fold, train_index, eval_index
