"""
Model training and submission tools for Builder agent.
Validation: only allowed model names and param ranges.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

ALLOWED_MODELS = {"ridge", "random_forest", "xgboost", "lightgbm"}


def _resolve_allowed(path: str, allowed_dirs: list[str]) -> Path:
    path = Path(path).resolve()
    for d in allowed_dirs:
        base = Path(d).resolve()
        try:
            path.relative_to(base)
            return path
        except ValueError:
            continue
    raise PermissionError(f"Path not allowed: {path}")


def _get_model(name: str, params: dict[str, Any]):
    """Return model instance. Validates params."""
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
            return lgb.LGBMRegressor(
                n_estimators=min(params.get("n_estimators", 100), 500),
                max_depth=min(params.get("max_depth", 6), 20),
                learning_rate=params.get("learning_rate", 0.1),
                random_state=params.get("random_state", 42),
            )
        except ImportError:
            raise ImportError("lightgbm not installed")
    raise ValueError(f"Model must be one of {ALLOWED_MODELS}")


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
) -> dict[str, Any]:
    """
    Train a regressor and save. If val_path is None, splits train_path by val_ratio.
    Returns {"val_mse": ..., "model_path": ...}.
    """
    if name.lower() not in ALLOWED_MODELS:
        raise ValueError(f"Model must be one of {ALLOWED_MODELS}")

    train_path = _resolve_allowed(train_path, allowed_dirs)
    if not train_path.exists():
        raise FileNotFoundError(str(train_path))
    model_save_path = _resolve_allowed(model_save_path, allowed_dirs)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not in {list(df.columns)}")
    y = df[target_col]
    X = df.drop(columns=[target_col])

    if val_path:
        val_path = _resolve_allowed(val_path, allowed_dirs)
        df_val = pd.read_csv(val_path)
        X_val = df_val.drop(columns=[target_col], errors="ignore")
        y_val = df_val[target_col]
        X_train, y_train = X, y
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_ratio, random_state=random_state
        )

    model = _get_model(name, params or {})
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    val_mse = float(mean_squared_error(y_val, y_pred_val))

    joblib.dump(model, model_save_path)
    return {"val_mse": val_mse, "model_path": str(model_save_path)}


def evaluate_mse(
    model_path: str,
    val_path: str,
    target_col: str,
    allowed_dirs: list[str],
) -> float:
    """Load model and validation data, return MSE."""
    model_path = _resolve_allowed(model_path, allowed_dirs)
    val_path = _resolve_allowed(val_path, allowed_dirs)
    model = joblib.load(model_path)
    df = pd.read_csv(val_path)
    y = df[target_col]
    X = df.drop(columns=[target_col], errors="ignore")
    y_pred = model.predict(X)
    return float(mean_squared_error(y, y_pred))


def make_submission(
    model_path: str,
    test_path: str,
    output_path: str,
    index_col: str | None,
    allowed_dirs: list[str],
) -> str:
    """Generate submission CSV with columns index, prediction. Returns output_path."""
    model_path = _resolve_allowed(model_path, allowed_dirs)
    test_path = _resolve_allowed(test_path, allowed_dirs)
    output_path = Path(output_path).resolve()
    if not any(
        str(output_path).startswith(str(Path(d).resolve())) for d in allowed_dirs
    ):
        raise PermissionError(f"Path not allowed: {output_path}")

    model = joblib.load(model_path)
    df = pd.read_csv(test_path)
    X = df.drop(columns=[index_col], errors="ignore") if index_col else df
    # Убедимся, что нет колонки target в test
    if "target" in X.columns:
        X = X.drop(columns=["target"])
    pred = model.predict(X)
    out = pd.DataFrame({"index": range(len(pred)), "prediction": pred})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return str(output_path)
