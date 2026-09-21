"""Preprocess raw stock market data for shared model experiments.

The script intentionally uses only the Python standard library so the shared
processed dataset can be generated before project dependencies are installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable


RAW_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "RSI",
    "MACD",
    "Sentiment",
    "Target",
]

FEATURE_COLUMNS = [
    "log_return",
    "simple_return",
    "intraday_return",
    "intraday_range",
    "volume_log_change",
    "volatility_5",
    "volatility_20",
    "return_mean_5",
    "volume_zscore_20",
    "RSI",
    "MACD",
    "Sentiment",
]

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
WALK_FORWARD_FOLDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create shared preprocessed features from raw stock data."
    )
    parser.add_argument(
        "--input",
        default="data/raw/stock_market_data_large.csv",
        help="Path to the raw CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory where processed files will be written.",
    )
    parser.add_argument(
        "--output-file",
        default="market_features.csv",
        help="Processed CSV filename.",
    )
    return parser.parse_args()


def to_float(value: str, column: str, row_number: int) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse {column}={value!r} as float at raw row {row_number}."
        ) from exc


def load_raw_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != RAW_COLUMNS:
            raise ValueError(
                "Unexpected raw CSV schema. "
                f"Expected {RAW_COLUMNS}, got {reader.fieldnames}."
            )

        for row_number, row in enumerate(reader, start=2):
            date = datetime.strptime(row["Date"], "%Y-%m-%d").date()
            parsed = {
                "Date": date.isoformat(),
                "Open": to_float(row["Open"], "Open", row_number),
                "High": to_float(row["High"], "High", row_number),
                "Low": to_float(row["Low"], "Low", row_number),
                "Close": to_float(row["Close"], "Close", row_number),
                "Volume": to_float(row["Volume"], "Volume", row_number),
                "RSI": to_float(row["RSI"], "RSI", row_number),
                "MACD": to_float(row["MACD"], "MACD", row_number),
                "Sentiment": to_float(row["Sentiment"], "Sentiment", row_number),
                "Target": int(row["Target"]),
            }

            if (
                parsed["Open"] <= 0
                or parsed["High"] <= 0
                or parsed["Low"] <= 0
                or parsed["Close"] <= 0
                or parsed["Volume"] <= 0
            ):
                continue

            parsed["ohlc_consistent"] = int(
                parsed["High"] >= max(parsed["Open"], parsed["Low"], parsed["Close"])
                and parsed["Low"] <= min(parsed["Open"], parsed["High"], parsed["Close"])
            )

            rows.append(parsed)

    rows.sort(key=lambda item: item["Date"])
    return rows


def rolling(values: list[float], end_index: int, window: int) -> list[float] | None:
    start_index = end_index - window + 1
    if start_index < 0:
        return None
    window_values = values[start_index : end_index + 1]
    if any(value is None for value in window_values):
        return None
    return window_values


def safe_stdev(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    return stdev(values)


def build_features(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    log_returns: list[float | None] = [None] * len(rows)
    simple_returns: list[float | None] = [None] * len(rows)
    volume_log_changes: list[float | None] = [None] * len(rows)
    volume_logs: list[float] = [math.log1p(float(row["Volume"])) for row in rows]

    for index in range(1, len(rows)):
        close = float(rows[index]["Close"])
        prev_close = float(rows[index - 1]["Close"])
        volume = float(rows[index]["Volume"])
        prev_volume = float(rows[index - 1]["Volume"])

        log_returns[index] = math.log(close / prev_close)
        simple_returns[index] = (close / prev_close) - 1.0
        volume_log_changes[index] = math.log(volume / prev_volume)

    processed_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        returns_5 = rolling(log_returns, index, 5)
        returns_20 = rolling(log_returns, index, 20)
        volume_window_20 = rolling(volume_logs, index, 20)

        if (
            log_returns[index] is None
            or simple_returns[index] is None
            or volume_log_changes[index] is None
            or returns_5 is None
            or returns_20 is None
            or volume_window_20 is None
        ):
            continue

        volume_std_20 = safe_stdev(volume_window_20)
        volume_zscore_20 = (
            (volume_logs[index] - mean(volume_window_20)) / volume_std_20
            if volume_std_20 > 0
            else 0.0
        )

        price_max = max(
            float(row["Open"]),
            float(row["High"]),
            float(row["Low"]),
            float(row["Close"]),
        )
        price_min = min(
            float(row["Open"]),
            float(row["High"]),
            float(row["Low"]),
            float(row["Close"]),
        )

        processed_rows.append(
            {
                "Date": row["Date"],
                "Open": row["Open"],
                "High": row["High"],
                "Low": row["Low"],
                "Close": row["Close"],
                "Volume": int(row["Volume"]),
                "Target": row["Target"],
                "ohlc_consistent": row["ohlc_consistent"],
                "log_return": log_returns[index],
                "simple_return": simple_returns[index],
                "intraday_return": (float(row["Close"]) / float(row["Open"])) - 1.0,
                "intraday_range": (price_max - price_min) / float(row["Close"]),
                "volume_log_change": volume_log_changes[index],
                "volatility_5": safe_stdev(returns_5),
                "volatility_20": safe_stdev(returns_20),
                "return_mean_5": mean(returns_5),
                "volume_zscore_20": volume_zscore_20,
                "RSI": row["RSI"],
                "MACD": row["MACD"],
                "Sentiment": row["Sentiment"],
            }
        )

    return processed_rows


def assign_splits(rows: list[dict[str, object]]) -> None:
    train_end = int(len(rows) * TRAIN_RATIO)
    validation_end = int(len(rows) * (TRAIN_RATIO + VALIDATION_RATIO))
    out_of_sample_count = len(rows) - train_end
    fold_size = math.ceil(out_of_sample_count / WALK_FORWARD_FOLDS)

    for index, row in enumerate(rows):
        if index < train_end:
            row["split"] = "train"
            row["walk_forward_fold"] = ""
        elif index < validation_end:
            row["split"] = "validation"
            row["walk_forward_fold"] = min(
                ((index - train_end) // fold_size) + 1,
                WALK_FORWARD_FOLDS,
            )
        else:
            row["split"] = "test"
            row["walk_forward_fold"] = min(
                ((index - train_end) // fold_size) + 1,
                WALK_FORWARD_FOLDS,
            )


def standardize_with_train_stats(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    train_rows = [row for row in rows if row["split"] == "train"]
    stats: dict[str, dict[str, float]] = {}

    for column in FEATURE_COLUMNS:
        train_values = [float(row[column]) for row in train_rows]
        column_mean = mean(train_values)
        column_std = safe_stdev(train_values)
        if column_std == 0:
            column_std = 1.0
        stats[column] = {"mean": column_mean, "std": column_std}

        z_column = f"{column}_z"
        for row in rows:
            row[z_column] = (float(row[column]) - column_mean) / column_std

    return stats


def write_processed_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "Date",
        "split",
        "walk_forward_fold",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Target",
        "ohlc_consistent",
        *FEATURE_COLUMNS,
        *(f"{column}_z" for column in FEATURE_COLUMNS),
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(
    raw_rows: list[dict[str, object]],
    processed_rows: list[dict[str, object]],
    stats: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    split_counts = Counter(row["split"] for row in processed_rows)
    metadata = {
        "source": "data/raw/stock_market_data_large.csv",
        "generated_file": "data/processed/market_features.csv",
        "raw_rows_after_basic_validation": len(raw_rows),
        "processed_rows": len(processed_rows),
        "dropped_for_lags_and_rolling_windows": len(raw_rows) - len(processed_rows),
        "date_min": processed_rows[0]["Date"],
        "date_max": processed_rows[-1]["Date"],
        "split_counts": dict(split_counts),
        "feature_columns": FEATURE_COLUMNS,
        "standardized_feature_columns": [
            f"{column}_z" for column in FEATURE_COLUMNS
        ],
        "standardization": "Mean and standard deviation are calculated on train rows only.",
        "train_statistics": stats,
    }

    output_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = load_raw_rows(input_path)
    processed_rows = build_features(raw_rows)
    if not processed_rows:
        raise RuntimeError("No processed rows were created.")

    assign_splits(processed_rows)
    stats = standardize_with_train_stats(processed_rows)

    write_processed_csv(processed_rows, output_dir / args.output_file)
    write_metadata(raw_rows, processed_rows, stats, output_dir / "market_features_metadata.json")

    print(f"Wrote {len(processed_rows)} rows to {output_dir / args.output_file}")
    print(f"Wrote metadata to {output_dir / 'market_features_metadata.json'}")


if __name__ == "__main__":
    main()
