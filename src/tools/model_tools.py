"""
Model training and submission tools for Builder agent.
Validation: only allowed model names and param ranges.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error

from src.security.validation import validate_model_params, validate_path


def _get_model(name: str, params: dict[str, Any]):
    """Return model instance. Params must be sanitized via validate_model_params."""
    name = name.lower()
    if name == "ridge":
        return Ridge(
            alpha=params.get("alpha", 1.0),
            random_state=params.get("random_state", 42),
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=min(params.get("n_estimators", 100), 500),
            max_depth=params.get("max_depth") or None,
            min_samples_leaf=max(1, params.get("min_samples_leaf", 1)),
            random_state=params.get("random_state", 42),
        )
    if name == "xgboost":
        try:
            import xgboost as xgb
            return xgb.XGBRegressor(
                n_estimators=min(params.get("n_estimators", 100), 500),
                max_depth=min(params.get("max_depth", 6), 20),
                learning_rate=params.get("learning_rate", 0.1),
                random_state=params.get("random_state", 42),
            )
        except ImportError:
            raise ImportError("xgboost not installed")
    if name == "lightgbm":
        try:
            import lightgbm as lgb

            kw: dict[str, Any] = dict(
                n_estimators=min(params.get("n_estimators", 100), 500),
                max_depth=min(params.get("max_depth", 6), 20),
                learning_rate=params.get("learning_rate", 0.1),
                random_state=params.get("random_state", 42),
            )
            if "min_child_samples" in params:
                kw["min_child_samples"] = int(params["min_child_samples"])
            if "num_leaves" in params:
                kw["num_leaves"] = int(params["num_leaves"])
            return lgb.LGBMRegressor(**kw)
        except ImportError:
            raise ImportError("lightgbm not installed")
    if name == "catboost":
        try:
            from catboost import CatBoostRegressor
            return CatBoostRegressor(
                iterations=min(int(params.get("n_estimators", 300)), 2000),
                depth=min(int(params.get("max_depth", 8)), 16),
                learning_rate=float(params.get("learning_rate", 0.1)),
                random_seed=int(params.get("random_state", 42)),
                verbose=False,
            )
        except ImportError:
            raise ImportError("catboost not installed")
    raise ValueError(f"Unknown model: {name}")


def train_regressor(
    name: str,
    params: dict[str, Any],
    train_path: str,
    target_col: str,
    val_path: str | None,
    model_save_path: str,
    allowed_dirs: list[str],
    val_ratio: float = 0.2,
    random_state: int = 42,
    cv_folds: int | None = None,
    fit_full_train: bool = False,
) -> dict[str, Any]:
    """
    Train a regressor and save.
    If val_path is None and not fit_full_train, splits train_path by val_ratio.
    If cv_folds >= 2, runs KFold CV on full train and adds cv_mse_mean / cv_mse_std.
    If fit_full_train, fits on all rows (for final submission); val_mse may come from CV if requested.
    """
    params = validate_model_params(name, params or {})
    params["random_state"] = random_state

    train_path = validate_path(str(train_path), allowed_dirs, must_exist=True)
    model_save_path = validate_path(str(model_save_path), allowed_dirs, must_exist=False)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not in {list(df.columns)}")
    y = df[target_col]
    X = df.drop(columns=[target_col])

    result: dict[str, Any] = {}

    if cv_folds is not None and cv_folds >= 2:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores: list[float] = []
        for train_idx, val_idx in kf.split(X):
            model = _get_model(name, params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[val_idx])
            scores.append(float(mean_squared_error(y.iloc[val_idx], pred)))
        result["cv_mse_mean"] = float(np.mean(scores))
        result["cv_mse_std"] = float(np.std(scores))

    if fit_full_train:
        model = _get_model(name, params)
        model.fit(X, y)
        joblib.dump(model, model_save_path)
        result["model_path"] = str(model_save_path)
        result["fit_full_train"] = True
        result["val_mse"] = result.get("cv_mse_mean")
        return result

    if val_path:
        val_path = validate_path(str(val_path), allowed_dirs, must_exist=True)
        df_val = pd.read_csv(val_path)
        X_val = df_val.drop(columns=[target_col], errors="ignore")
        y_val = df_val[target_col]
        X_train, y_train = X, y
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_ratio, random_state=random_state
        )

    model = _get_model(name, params)
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    val_mse = float(mean_squared_error(y_val, y_pred_val))

    joblib.dump(model, model_save_path)
    result.update(
        {
            "val_mse": val_mse,
            "model_path": str(model_save_path),
            "fit_full_train": False,
        }
    )
    return result


def make_submission(
    model_path: str,
    test_path: str,
    output_path: str,
    index_col: str | None,
    allowed_dirs: list[str],
    robustness: dict[str, Any] | None = None,
) -> str:
    """
    Generate submission CSV with columns index, prediction. Returns output_path.

    robustness (optional):
      clip_predictions: if True, clip predictions to target quantiles from train_path
      clip_quantiles: (low, high) defaults (0.005, 0.995)
      train_path: required for clipping / feature bounds
      clip_features_to_train_range: clip numeric test features to [min,max] from train (mitigates drift/outliers)
    """
    model_path = validate_path(str(model_path), allowed_dirs, must_exist=True)
    test_path = validate_path(str(test_path), allowed_dirs, must_exist=True)
    output_path = validate_path(str(output_path), allowed_dirs, must_exist=False)

    robustness = robustness or {}
    train_path = robustness.get("train_path")
    if train_path and robustness.get("clip_features_to_train_range"):
        train_path = validate_path(str(train_path), allowed_dirs, must_exist=True)

    model = joblib.load(model_path)
    df = pd.read_csv(test_path)
    X = df.drop(columns=[index_col], errors="ignore") if index_col else df
    if "target" in X.columns:
        X = X.drop(columns=["target"])

    if train_path and robustness.get("clip_features_to_train_range"):
        train_df = pd.read_csv(train_path)
        num_cols = X.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            if c in train_df.columns and pd.api.types.is_numeric_dtype(train_df[c]):
                lo = float(train_df[c].min())
                hi = float(train_df[c].max())
                if lo < hi:
                    X[c] = X[c].clip(lower=lo, upper=hi)

    pred = np.asarray(model.predict(X), dtype=float).ravel()

    if train_path and robustness.get("clip_predictions"):
        tp = validate_path(str(train_path), allowed_dirs, must_exist=True)
        train_df = pd.read_csv(tp)
        if "target" in train_df.columns:
            y = train_df["target"]
            q = robustness.get("clip_quantiles") or (0.005, 0.995)
            lo, hi = float(y.quantile(q[0])), float(y.quantile(q[1]))
            pred = np.clip(pred, lo, hi)

    out = pd.DataFrame({"index": range(len(pred)), "prediction": pred})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return str(output_path)


def write_submission_array(
    pred: np.ndarray,
    output_path: str,
    allowed_dirs: list[str],
    robustness: dict[str, Any] | None = None,
) -> str:
    """Write submission CSV from a prediction vector (same clipping rules as make_submission)."""
    output_path = validate_path(str(output_path), allowed_dirs, must_exist=False)
    pred = np.asarray(pred, dtype=float).ravel()
    robustness = robustness or {}
    train_path = robustness.get("train_path")
    if train_path and robustness.get("clip_predictions"):
        tp = validate_path(str(train_path), allowed_dirs, must_exist=True)
        train_df = pd.read_csv(tp)
        if "target" in train_df.columns:
            y = train_df["target"]
            q = robustness.get("clip_quantiles") or (0.005, 0.995)
            lo, hi = float(y.quantile(q[0])), float(y.quantile(q[1]))
            pred = np.clip(pred, lo, hi)

    out = pd.DataFrame({"index": range(len(pred)), "prediction": pred})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return str(output_path)


def train_regressor_kfold_blend(
    name: str,
    params: dict[str, Any],
    train_path: str,
    test_path: str,
    model_save_path: str,
    submission_path: str,
    allowed_dirs: list[str],
    cv_folds: int,
    random_state: int,
    robustness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    K-fold CV MSE на train; предсказания на test — среднее по K моделям (каждая обучена на K-1 фолде).
    Устойчивее к шуму разбиения, чем одна модель на всём train.
    """
    params = validate_model_params(name, params or {})
    train_path = validate_path(str(train_path), allowed_dirs, must_exist=True)
    test_path = validate_path(str(test_path), allowed_dirs, must_exist=True)
    model_save_path = validate_path(str(model_save_path), allowed_dirs, must_exist=False)
    submission_path = validate_path(str(submission_path), allowed_dirs, must_exist=False)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_path)
    y = df["target"]
    X = df.drop(columns=["target"], errors="ignore")
    df_test = pd.read_csv(test_path)
    X_test = df_test.copy()
    if "target" in X_test.columns:
        X_test = X_test.drop(columns=["target"])
    for c in X.columns:
        if c not in X_test.columns:
            X_test[c] = np.nan
    X_test = X_test[X.columns]

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores: list[float] = []
    for train_idx, val_idx in kf.split(X):
        model = _get_model(name, {**params, "random_state": random_state})
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[val_idx])
        scores.append(float(mean_squared_error(y.iloc[val_idx], pred)))

    test_preds: list[np.ndarray] = []
    last_model = None
    for train_idx, _ in kf.split(X):
        p = dict(params)
        p["random_state"] = random_state
        model = _get_model(name, p)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        test_preds.append(np.asarray(model.predict(X_test), dtype=float).ravel())
        last_model = model

    if last_model is not None:
        joblib.dump(last_model, model_save_path)

    blend = np.mean(np.vstack(test_preds), axis=0)
    write_submission_array(blend, str(submission_path), allowed_dirs, robustness=robustness)

    return {
        "cv_mse_mean": float(np.mean(scores)),
        "cv_mse_std": float(np.std(scores)),
        "model_path": str(model_save_path),
        "fit_full_train": False,
        "val_mse": float(np.mean(scores)),
        "oof_kfold_blend": True,
        "n_blend_folds": cv_folds,
    }
