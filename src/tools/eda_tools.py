"""
EDA tools for Explorer agent.
All inputs must be validated (paths under allowed dirs, no path traversal).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.security.validation import validate_path


def load_dataset(path: str, allowed_dirs: list[str]) -> pd.DataFrame:
    """Load CSV; path must be under one of allowed_dirs."""
    path = validate_path(str(path), allowed_dirs, must_exist=True)
    if not path.suffix.lower() == ".csv":
        raise ValueError("Only CSV files are allowed")
    return pd.read_csv(path)


def get_dtypes(df: pd.DataFrame) -> dict[str, str]:
    """Return column dtypes as string mapping."""
    return df.dtypes.astype(str).to_dict()


def describe(df: pd.DataFrame) -> dict[str, Any]:
    """Descriptive statistics (numeric)."""
    return df.describe().to_dict() if len(df) else {}


def missing_report(df: pd.DataFrame) -> dict[str, int]:
    """Missing value counts per column."""
    return df.isnull().sum().to_dict()


def correlation_with_target(df: pd.DataFrame, target_col: str = "target") -> dict[str, float]:
    """Correlation of numeric columns with target."""
    if target_col not in df.columns:
        return {}
    numeric = df.select_dtypes(include=["number"])
    if target_col not in numeric.columns:
        return {}
    return numeric.corr()[target_col].drop(target_col, errors="ignore").to_dict()


def save_eda_report(report: dict | str, output_path: str, allowed_dirs: list[str]) -> str:
    """Save EDA report (JSON or text) to an allowed directory."""
    path = validate_path(str(output_path), allowed_dirs, must_exist=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(report, dict):
        import json
        path.write_text(json.dumps(report, indent=2, default=str))
    else:
        path.write_text(report)
    return str(path)
