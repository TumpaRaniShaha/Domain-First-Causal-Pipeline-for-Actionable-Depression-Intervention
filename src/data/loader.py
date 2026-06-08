"""NHANES XPT loading and person-level merge utilities."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DOMAIN_BY_MODULE: Dict[str, str] = {
    "DEMO": "socio-demographic",
    "INQ": "socio-demographic",
    "HIQ": "socio-demographic",
    "OCQ": "socio-demographic",
    "FSQ": "socio-demographic",
    "ALQ": "behavioral-lifestyle",
    "SMQ": "behavioral-lifestyle",
    "PAQ": "behavioral-lifestyle",
    "SLQ": "behavioral-lifestyle",
    "DBQ": "behavioral-lifestyle",
    "WHQ": "behavioral-lifestyle",
    "HSQ": "clinical-health",
    "HUQ": "clinical-health",
    "MCQ": "clinical-health",
    "BPQ": "clinical-health",
    "OHQ": "clinical-health",
    "RHQ": "clinical-health",
    "HEQ": "clinical-health",
    "RXQ": "clinical-health",
    "ACQ": "clinical-health",
    "AUQ": "clinical-health",
    "DPQ": "mental-health-outcome",
}


def infer_module_from_filename(stem: str) -> str:
    """Infer NHANES module name from a file stem such as ``P_DEMO``."""
    normalized = stem.strip().upper()
    return normalized[2:] if normalized.startswith("P_") else normalized


def coarse_module_key(module: str) -> str:
    """Map module variants such as RXQ_RX or KIQ_U to a coarse module key."""
    module = module.strip().upper()
    if module in DOMAIN_BY_MODULE:
        return module

    parts = module.split("_")
    for candidate in (parts[0], module[:3], module[:4]):
        if candidate in DOMAIN_BY_MODULE:
            return candidate
    return parts[0] if parts else module


def assign_domain_from_module(module: str) -> str:
    """Return expert domain label for a module."""
    return DOMAIN_BY_MODULE.get(coarse_module_key(module), "other")


def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory footprint without changing values materially."""
    for column in df.select_dtypes(include=["float64"]).columns:
        df[column] = pd.to_numeric(df[column], downcast="float")
    for column in df.select_dtypes(include=["int64", "int32"]).columns:
        df[column] = pd.to_numeric(df[column], downcast="integer")
    return df


def _mode_agg(series: pd.Series):
    series = series.dropna()
    if series.empty:
        return np.nan
    mode = series.mode()
    return mode.iloc[0] if not mode.empty else series.iloc[0]


def _coerce_numeric_if_mostly_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_object_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().mean() >= 0.8:
            return numeric
    return series


def aggregate_module_person_level(df: pd.DataFrame, module: str) -> pd.DataFrame:
    """Aggregate long NHANES modules to one row per SEQN."""
    data = df.copy()
    data.columns = [str(column).upper() for column in data.columns]
    for column in data.columns:
        if column != "SEQN":
            data[column] = _coerce_numeric_if_mostly_numeric(data[column])

    row_count = data.groupby("SEQN").size().rename(f"{module}_N").reset_index()
    aggregations = {
        column: ("median" if pd.api.types.is_numeric_dtype(data[column]) else _mode_agg)
        for column in data.columns
        if column != "SEQN"
    }
    aggregated = data.groupby("SEQN", as_index=False).agg(aggregations)
    return downcast_numeric(aggregated.merge(row_count, on="SEQN", how="left"))


def read_all_xpt(input_dir: Path | str) -> Dict[str, pd.DataFrame]:
    """Read every XPT file with a SEQN identifier from ``input_dir``."""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    dataframes: Dict[str, pd.DataFrame] = {}
    for path in sorted(input_path.glob("*.xpt")):
        module = infer_module_from_filename(path.stem)
        try:
            df = pd.read_sas(path, format="xport", encoding="utf-8")
        except Exception as exc:
            print(f"[WARN] Could not read {path.name}: {exc}")
            continue

        df.columns = [str(column).upper() for column in df.columns]
        if "SEQN" not in df.columns:
            print(f"[INFO] {path.name} has no SEQN; skipped.")
            continue

        dataframes[module] = downcast_numeric(df)
        duplicate_count = int(df.duplicated("SEQN").sum())
        long_tag = " (long)" if duplicate_count else ""
        print(f"[OK] {path.name:20s} -> {module:10s} shape={tuple(df.shape)}{long_tag}")

    if not dataframes:
        raise RuntimeError(f"No readable XPT modules found in {input_path}")
    return dataframes


def merge_modules_person_level(
    modules: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Outer-merge modules at person level and return feature-to-domain mapping."""
    if not modules:
        raise RuntimeError("No modules to merge.")

    start_key = "DEMO" if "DEMO" in modules else next(iter(modules.keys()))
    base = modules[start_key]
    if base.duplicated("SEQN").any():
        base = aggregate_module_person_level(base, start_key)

    merged = base.copy()
    column_domain = {
        column: assign_domain_from_module(start_key)
        for column in merged.columns
        if column != "SEQN"
    }

    for module, df in modules.items():
        if module == start_key:
            continue

        df_module = (
            aggregate_module_person_level(df, module)
            if df.duplicated("SEQN").any()
            else downcast_numeric(df.copy())
        )
        for column in list(df_module.columns):
            if column == "SEQN":
                continue
            domain = assign_domain_from_module(module)
            if column in merged.columns:
                new_column = f"{column}__{module}"
                df_module.rename(columns={column: new_column}, inplace=True)
                column_domain[new_column] = domain
            else:
                column_domain[column] = domain

        merged = pd.merge(merged, df_module, on="SEQN", how="outer", validate="one_to_one")
        del df_module
        gc.collect()

    return downcast_numeric(merged), column_domain
