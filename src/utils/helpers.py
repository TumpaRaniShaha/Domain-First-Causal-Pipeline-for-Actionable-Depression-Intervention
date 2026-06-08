"""Shared helpers for the depression causal feature-selection pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "Dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "mental_outputs"


def ensure_dir(path: Path | str) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def write_json(path: Path | str, payload: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_phq9(df: pd.DataFrame) -> pd.DataFrame:
    """Build PHQ-9 total from DPQ010-DPQ090 item columns."""
    dpq_valid_values = {0, 1, 2, 3}
    dpq_cols = [column for column in df.columns if __import__("re").fullmatch(r"DPQ0[1-9]0(?:__.*)?", column)]
    output = df.copy()

    if not dpq_cols:
        print("[WARN] No DPQ item columns found; PHQ9_TOTAL set to missing.")
        output["PHQ9_TOTAL"] = np.nan
        return output

    def coerce_item(series: pd.Series) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        return values.where(values.isin(dpq_valid_values), np.nan)

    matrix = pd.DataFrame({column: coerce_item(output[column]) for column in dpq_cols})
    output["PHQ9_TOTAL"] = matrix.sum(axis=1, min_count=1)
    return output


def base_feature_name(feature_name: str) -> str:
    return feature_name.split("_")[0].split("=")[0]


def make_ohe(min_frequency: float, max_categories: int):
    """Create a scikit-learn OneHotEncoder across old/new sklearn versions."""
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=min_frequency,
            max_categories=max_categories,
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
