"""
Feature engineering tools for Engineer agent.
Input validation and allowed dirs required for all file paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression


def _resolve_allowed(path: str, allowed_dirs: list[str]) -> Path:
    """Resolve path and check it is under one of allowed_dirs."""
    path = Path(path).resolve()
    for d in allowed_dirs:
        base = Path(d).resolve()
        try:
            path.relative_to(base)
            return path
        except ValueError:
            continue
    raise PermissionError(f"Path not allowed: {path}")


def load_eda_report(report_path: str, allowed_dirs: list[str]) -> dict[str, Any]:
    """Load EDA report from JSON."""
    path = _resolve_allowed(report_path, allowed_dirs)
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text())


# Колонки датасета: name, _id, host_name, location_cluster, location, lat, lon,
# type_house, sum, min_days, amt_reviews, last_dt, avg_reviews, total_host, target
NUMERIC_COLS = ["lat", "lon", "sum", "min_days", "amt_reviews", "avg_reviews", "total_host"]
CATEGORICAL_COLS = ["host_name", "location_cluster", "location", "type_house"]
DATE_COL = "last_dt"
DROP_COLS = ["name", "_id"]


def fit_preprocessor(
    config: dict,
    train_path: str,
    target_col: str,
    allowed_dirs: list[str],
) -> str:
    """
    Fit preprocessing pipeline and save to artifacts.
    - Numeric: impute missing (median), scale.
    - Categorical: OneHotEncoder (handle_unknown='ignore').
    - last_dt: parse to numeric (days since min), impute.
    Returns path to saved pipeline (joblib).
    """
    path = _resolve_allowed(train_path, allowed_dirs)
    if not path.exists():
        raise FileNotFoundError(str(path))
    train = pd.read_csv(path)

    out_path = config.get("pipeline_save_path")
    if not out_path:
        raise ValueError("config must contain 'pipeline_save_path'")
    out_path = _resolve_allowed(out_path, allowed_dirs)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Дата: переводим в число (дни от минимальной даты), пропуски — медиана
    ref_min = None
    date_fill = None
    if DATE_COL in train.columns:
        train = train.copy()
        train[DATE_COL] = pd.to_datetime(train[DATE_COL], errors="coerce")
        ref_min = train[DATE_COL].min()
        train[DATE_COL] = (train[DATE_COL] - ref_min).dt.days
        date_fill = train[DATE_COL].median()
        train[DATE_COL] = train[DATE_COL].fillna(date_fill)
    numeric_cols = [c for c in NUMERIC_COLS if c in train.columns]
    cat_cols = [c for c in CATEGORICAL_COLS if c in train.columns]
    date_cols = [DATE_COL] if DATE_COL in train.columns else []

    transformers = []
    if numeric_cols:
        transformers.append(
            ("num", SimpleImputer(strategy="median"), numeric_cols),
        )
    if date_cols:
        transformers.append(
            ("date", SimpleImputer(strategy="median"), date_cols),
        )
    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        )

    if not transformers:
        raise ValueError("No feature columns found")

    ct = ColumnTransformer(
        transformers,
        remainder="drop",
    )
    # Масштабирование числовых (после Imputer) — делаем отдельным шагом через Pipeline
    from sklearn.pipeline import Pipeline
    steps = [("preprocess", ct)]
    # Для Ridge нужен масштаб; для дерева не обязателен — добавим в pipeline
    from sklearn.preprocessing import StandardScaler
    steps.append(("scale", StandardScaler()))
    pipeline = Pipeline(steps)

    X = train.drop(columns=[target_col], errors="ignore")
    for c in DROP_COLS:
        if c in X.columns:
            X = X.drop(columns=[c])
    y = train[target_col] if target_col in train.columns else None
    pipeline.fit(X, y)

    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_names_in_": X.columns.tolist(),
            "target_col": target_col,
            "ref_min": ref_min,
            "date_fill": date_fill,
        },
        out_path,
    )
    return str(out_path)


def transform_train_test(
    pipeline_path: str,
    train_path: str,
    test_path: str,
    output_dir: str,
    allowed_dirs: list[str],
) -> dict[str, str]:
    """Transform train/test and save processed CSVs. Returns paths to train_processed, test_processed."""
    pipe_path = _resolve_allowed(pipeline_path, allowed_dirs)
    train_path = _resolve_allowed(train_path, allowed_dirs)
    test_path = _resolve_allowed(test_path, allowed_dirs)
    out_dir = _resolve_allowed(output_dir, allowed_dirs)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = joblib.load(pipe_path)
    pipeline = data["pipeline"]
    target_col = data.get("target_col", "target")
    ref_min = data.get("ref_min")
    date_fill = data.get("date_fill", 0)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    def to_days(ser, ref_min_val, fill_val):
        s = pd.to_datetime(ser, errors="coerce")
        if ref_min_val is None:
            return ser
        return (s - ref_min_val).dt.days.fillna(fill_val)

    if DATE_COL in train.columns and ref_min is not None:
        train = train.copy()
        train[DATE_COL] = to_days(train[DATE_COL], ref_min, date_fill)
    if DATE_COL in test.columns and ref_min is not None:
        test = test.copy()
        test[DATE_COL] = to_days(test[DATE_COL], ref_min, date_fill)

    X_train = train.drop(columns=[target_col], errors="ignore")
    for c in DROP_COLS:
        if c in X_train.columns:
            X_train = X_train.drop(columns=[c])
    X_test = test.copy()
    for c in DROP_COLS:
        if c in X_test.columns:
            X_test = X_test.drop(columns=[c])
    # Выравниваем колонки по train (test может не иметь части категорий)
    for c in X_train.columns:
        if c not in X_test.columns:
            X_test[c] = 0
    X_test = X_test[X_train.columns]

    X_train_t = pipeline.transform(X_train)
    X_test_t = pipeline.transform(X_test)

    import numpy as np
    train_out = pd.DataFrame(X_train_t)
    train_out[target_col] = train[target_col].values
    test_out = pd.DataFrame(X_test_t)

    train_save = out_dir / "train_processed.csv"
    test_save = out_dir / "test_processed.csv"
    train_out.to_csv(train_save, index=False)
    test_out.to_csv(test_save, index=False)

    return {"train_path": str(train_save), "test_path": str(test_save)}


def select_features(
    method: str,
    k: int,
    train_path: str,
    target_col: str,
    allowed_dirs: list[str],
) -> list[str]:
    """Feature selection: 'correlation' (top k by abs corr) or 'kbest' (SelectKBest). Returns list of column names."""
    path = _resolve_allowed(train_path, allowed_dirs)
    df = pd.read_csv(path)
    if target_col not in df.columns:
        return df.drop(columns=[target_col], errors="ignore").columns.tolist()

    X = df.drop(columns=[target_col])
    y = df[target_col]
    numeric = X.select_dtypes(include=["number"])
    if numeric.empty:
        return X.columns.tolist()
    if method == "correlation":
        corr = numeric.corrwith(y).abs().sort_values(ascending=False)
        return corr.head(k).index.tolist()
    if method == "kbest":
        sel = SelectKBest(f_regression, k=min(k, numeric.shape[1]))
        sel.fit(numeric, y)
        return numeric.columns[sel.get_support()].tolist()
    return numeric.columns.tolist()
