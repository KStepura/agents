"""
Инструменты EDA для агента Explorer.
Все входные данные должны проходить валидацию (пути только в разрешённых директориях, без path traversal).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.security.validation import validate_path


def load_dataset(path: str, allowed_dirs: list[str]) -> pd.DataFrame:
    """Загружает CSV; путь должен находиться в одной из allowed_dirs."""
    path = validate_path(str(path), allowed_dirs, must_exist=True)
    if not path.suffix.lower() == ".csv":
        raise ValueError("Only CSV files are allowed")
    return pd.read_csv(path)


def get_dtypes(df: pd.DataFrame) -> dict[str, str]:
    """Возвращает типы данных столбцов в виде отображения строк."""
    return df.dtypes.astype(str).to_dict()


def describe(df: pd.DataFrame) -> dict[str, Any]:
    """Описательная статистика (для числовых признаков)."""
    return df.describe().to_dict() if len(df) else {}


def missing_report(df: pd.DataFrame) -> dict[str, int]:
    """Количество пропущенных значений по каждому столбцу."""
    return df.isnull().sum().to_dict()


def correlation_with_target(df: pd.DataFrame, target_col: str = "target") -> dict[str, float]:
    """Корреляция числовых признаков с целевой переменной."""
    if target_col not in df.columns:
        return {}
    numeric = df.select_dtypes(include=["number"])
    if target_col not in numeric.columns:
        return {}
    return numeric.corr()[target_col].drop(target_col, errors="ignore").to_dict()


def save_eda_report(report: dict | str, output_path: str, allowed_dirs: list[str]) -> str:
    """Сохраняет отчёт EDA (JSON или текст) в разрешённую директорию."""
    path = validate_path(str(output_path), allowed_dirs, must_exist=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(report, dict):
        import json
        path.write_text(json.dumps(report, indent=2, default=str))
    else:
        path.write_text(report)
    return str(path)
