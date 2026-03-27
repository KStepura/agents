"""
Инструменты feature engineering для агента Engineer.
Поддерживает OHE (устаревший режим) или te_freq: target encoding + частоты + признаки дат (лучше для моделей деревьев).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.security.validation import validate_path

NUMERIC_COLS = ["lat", "lon", "sum", "min_days", "amt_reviews", "avg_reviews", "total_host"]
CATEGORICAL_COLS = ["host_name", "location_cluster", "location", "type_house"]
DATE_COL = "last_dt"
DROP_COLS = ["name", "_id"]

TE_SMOOTHING = 10.0
OTHER_LABEL = "__OTHER__"


def _collapse_rare_train(series: pd.Series, min_count: int) -> tuple[pd.Series, pd.Index]:
    """Схлопывание редких значений в OTHER_LABEL; возвращает (серия, индекс допустимых значений train)."""
    vc = series.value_counts(dropna=False)
    kept_idx = vc[vc >= min_count].index

    def _repl(x: Any) -> Any:
        return x if x in kept_idx else OTHER_LABEL

    return series.map(_repl), kept_idx


def _collapse_rare_test(series: pd.Series, kept_idx: pd.Index) -> pd.Series:
    return series.map(lambda x: x if x in kept_idx else OTHER_LABEL)


def _parse_date_base(df: pd.DataFrame) -> tuple[pd.DataFrame, Any, Any]:
    """Добавляет признаки last_dt_days, last_dt_month, last_dt_dow; возвращает ref_min и date_fill для расчёта дней."""
    ref_min = None
    date_fill = None
    out = df.copy()
    if DATE_COL not in out.columns:
        return out, ref_min, date_fill
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    ref_min = out[DATE_COL].min()
    out["last_dt_month"] = out[DATE_COL].dt.month.fillna(-1).astype(float)
    out["last_dt_dow"] = out[DATE_COL].dt.dayofweek.fillna(-1).astype(float)
    days = (out[DATE_COL] - ref_min).dt.days
    date_fill = float(days.median()) if days.notna().any() else 0.0
    out[DATE_COL] = days.fillna(date_fill)
    return out, ref_min, date_fill


def _apply_date_test(df: pd.DataFrame, ref_min: Any, date_fill: float) -> pd.DataFrame:
    """Обрабатывает даты в test, используя ref_min из train (из артефакта)."""
    out = df.copy()
    if DATE_COL not in out.columns:
        return out
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    out["last_dt_month"] = out[DATE_COL].dt.month.fillna(-1).astype(float)
    out["last_dt_dow"] = out[DATE_COL].dt.dayofweek.fillna(-1).astype(float)
    out[DATE_COL] = (out[DATE_COL] - ref_min).dt.days.fillna(date_fill)
    return out


def _fit_te_map(series: pd.Series, y: pd.Series, smoothing: float) -> tuple[dict[Any, float], float]:
    global_mean = float(y.mean())
    df = pd.DataFrame({"c": series.astype("object"), "y": y})
    stats = df.groupby("c", dropna=False)["y"].agg(["mean", "count"])
    maps: dict[Any, float] = {}
    for cat, row in stats.iterrows():
        cnt = float(row["count"])
        m = float(row["mean"])
        sm = (cnt * m + smoothing * global_mean) / (cnt + smoothing)
        maps[cat] = sm
    return maps, global_mean


def _fit_freq_map(series: pd.Series) -> dict[Any, float]:
    vc = series.value_counts(dropna=False)
    return {k: float(np.log1p(v)) for k, v in vc.items()}


def _apply_te(series: pd.Series, amap: dict[Any, float], global_mean: float) -> pd.Series:
    def f(x: Any) -> float:
        if pd.isna(x) and np.nan in amap:
            return float(amap[np.nan])
        return float(amap.get(x, global_mean))

    return series.map(f)


def _apply_freq(series: pd.Series, fmap: dict[Any, float]) -> pd.Series:
    def g(x):
        if x in fmap:
            return fmap[x]
        return float(np.log1p(0))

    return series.map(g)


def _engineer_te_freq(
    X: pd.DataFrame,
    y: pd.Series | None,
    *,
    te_maps: dict[str, dict[Any, float]] | None = None,
    freq_maps: dict[str, dict[Any, float]] | None = None,
    global_means: dict[str, float] | None = None,
    fit: bool,
) -> pd.DataFrame:
    """Добавляет колонки col__te и col__freq для каждой категориальной переменной; удаляет исходные категориальные столбцы."""
    out = X.copy()
    cat_cols = [c for c in CATEGORICAL_COLS if c in out.columns]
    if fit:
        assert y is not None
        te_maps = {}
        freq_maps = {}
        global_means = {}
        for c in cat_cols:
            te_maps[c], global_means[c] = _fit_te_map(out[c], y, TE_SMOOTHING)
            freq_maps[c] = _fit_freq_map(out[c])
        out = out.drop(columns=cat_cols, errors="ignore")
        for c in cat_cols:
            out[f"{c}__te"] = _apply_te(X[c], te_maps[c], global_means[c])
            out[f"{c}__freq"] = _apply_freq(X[c], freq_maps[c])
        meta = {"te_maps": te_maps, "freq_maps": freq_maps, "global_means": global_means}
        return out, meta
    assert te_maps is not None and freq_maps is not None and global_means is not None
    out = out.drop(columns=cat_cols, errors="ignore")
    for c in cat_cols:
        out[f"{c}__te"] = _apply_te(X[c], te_maps[c], global_means[c])
        out[f"{c}__freq"] = _apply_freq(X[c], freq_maps[c])
    return out


def fit_preprocessor(
    config: dict,
    train_path: str,
    target_col: str,
    allowed_dirs: list[str],
) -> str:
    """
    Обучает препроцессинг и сохраняет его в artifacts.
    Ключи config:
    - pipeline_save_path (обязательный)
    - encoding: 'te_freq' (по умолчанию) или 'ohe' — te_freq использует target+frequency encoding и признаки дат
    """
    path = validate_path(str(train_path), allowed_dirs, must_exist=True)
    train = pd.read_csv(path)

    out_path = config.get("pipeline_save_path")
    if not out_path:
        raise ValueError("config must contain 'pipeline_save_path'")
    out_path = validate_path(str(out_path), allowed_dirs, must_exist=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    encoding = config.get("encoding", "te_freq")
    rare_min = config.get("rare_category_min_count")

    if encoding == "te_freq":
        train, ref_min, date_fill = _parse_date_base(train)
        y = train[target_col]
        X = train.drop(columns=[target_col], errors="ignore")
        for c in DROP_COLS:
            if c in X.columns:
                X = X.drop(columns=[c])
        rare_kept: dict[str, list[Any]] = {}
        if rare_min is not None and int(rare_min) >= 2:
            mc = int(rare_min)
            for c in CATEGORICAL_COLS:
                if c in X.columns:
                    X[c], kept = _collapse_rare_train(X[c], mc)
                    rare_kept[c] = kept.tolist()
        X_eng, enc_meta = _engineer_te_freq(X, y, fit=True)
        for c in X_eng.columns:
            X_eng[c] = pd.to_numeric(X_eng[c], errors="coerce")

        imputer = SimpleImputer(strategy="median")
        pipeline = Pipeline([("imputer", imputer)])
        pipeline.fit(X_eng, y)

        payload = {
            "version": 2,
            "encoding": "te_freq",
            "pipeline": pipeline,
            "feature_columns": X_eng.columns.tolist(),
            "target_col": target_col,
            "ref_min": ref_min,
            "date_fill": date_fill,
            "enc_meta": enc_meta,
            "rare_category_min_count": rare_min,
            "rare_kept": rare_kept if rare_kept else None,
        }
        joblib.dump(payload, out_path)
        return str(out_path)

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
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_cols))
    if date_cols:
        transformers.append(("date", SimpleImputer(strategy="median"), date_cols))
    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        )

    if not transformers:
        raise ValueError("No feature features found")

    ct = ColumnTransformer(transformers, remainder="drop")
    pipeline = Pipeline([("preprocess", ct), ("scale", StandardScaler())])

    X = train.drop(columns=[target_col], errors="ignore")
    for c in DROP_COLS:
        if c in X.columns:
            X = X.drop(columns=[c])
    y = train[target_col] if target_col in train.columns else None
    pipeline.fit(X, y)

    joblib.dump(
        {
            "version": 1,
            "encoding": "ohe",
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
    """Преобразует train/test и сохраняет обработанные CSV-файлы."""
    pipe_path = validate_path(str(pipeline_path), allowed_dirs, must_exist=True)
    train_path = validate_path(str(train_path), allowed_dirs, must_exist=True)
    test_path = validate_path(str(test_path), allowed_dirs, must_exist=True)
    out_dir = validate_path(str(output_dir), allowed_dirs, must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = joblib.load(pipe_path)
    pipeline = data["pipeline"]
    target_col = data.get("target_col", "target")
    ref_min = data.get("ref_min")
    date_fill = data.get("date_fill", 0)
    encoding = data.get("encoding", "ohe")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if encoding == "te_freq":
        enc_meta = data["enc_meta"]
        train, _, _ = _parse_date_base(train)
        if ref_min is not None:
            test = _apply_date_test(test, ref_min, float(date_fill))
        else:
            test, _, _ = _parse_date_base(test)

        X_train = train.drop(columns=[target_col], errors="ignore")
        for c in DROP_COLS:
            if c in X_train.columns:
                X_train = X_train.drop(columns=[c])
        X_test = test.copy()
        for c in DROP_COLS:
            if c in X_test.columns:
                X_test = X_test.drop(columns=[c])

        rare_kept = data.get("rare_kept")
        if rare_kept:
            for c, kept_list in rare_kept.items():
                if c in X_train.columns:
                    kept_idx = pd.Index(kept_list)
                    X_train[c] = _collapse_rare_test(X_train[c], kept_idx)
                if c in X_test.columns:
                    kept_idx = pd.Index(kept_list)
                    X_test[c] = _collapse_rare_test(X_test[c], kept_idx)

        X_train = _engineer_te_freq(
            X_train,
            None,
            te_maps=enc_meta["te_maps"],
            freq_maps=enc_meta["freq_maps"],
            global_means=enc_meta["global_means"],
            fit=False,
        )
        X_test = _engineer_te_freq(
            X_test,
            None,
            te_maps=enc_meta["te_maps"],
            freq_maps=enc_meta["freq_maps"],
            global_means=enc_meta["global_means"],
            fit=False,
        )
        for c in X_train.columns:
            X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        for c in X_test.columns:
            X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
        for c in X_train.columns:
            if c not in X_test.columns:
                X_test[c] = np.nan
        X_test = X_test[X_train.columns]

        X_train_t = pipeline.transform(X_train)
        X_test_t = pipeline.transform(X_test)
        train_out = pd.DataFrame(X_train_t, columns=data["feature_columns"])
        test_out = pd.DataFrame(X_test_t, columns=data["feature_columns"])
        train_out[target_col] = train[target_col].values
    else:
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
        for c in X_train.columns:
            if c not in X_test.columns:
                X_test[c] = 0
        X_test = X_test[X_train.columns]

        X_train_t = pipeline.transform(X_train)
        X_test_t = pipeline.transform(X_test)
        train_out = pd.DataFrame(X_train_t)
        train_out[target_col] = train[target_col].values
        test_out = pd.DataFrame(X_test_t)

    train_save = out_dir / "train_processed.csv"
    test_save = out_dir / "test_processed.csv"
    train_out.to_csv(train_save, index=False)
    test_out.to_csv(test_save, index=False)

    return {"train_path": str(train_save), "test_path": str(test_save)}
