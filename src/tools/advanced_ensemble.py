"""
OOF-стекинг (несколько базовых моделей + мета-регрессор на out-of-fold предсказаниях)
и опциональное дообучение с псевдо-лейблами на test.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from src.security.validation import validate_model_params, validate_path
from src.tools.model_tools import _get_model, write_submission_array


def _params_except_model(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != "model"}


def _get_meta_model(name: str, params: dict[str, Any], random_state: int):
    name = (name or "ridge").lower()
    params = validate_model_params(name, params or {})
    params["random_state"] = random_state
    return _get_model(name, params)


def train_oof_stacking(
    train_path: str,
    test_path: str,
    base_models: list[dict[str, Any]],
    meta_config: dict[str, Any],
    submission_path: str,
    bundle_path: str,
    allowed_dirs: list[str],
    cv_folds: int,
    random_state: int,
    robustness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Уровень 0: K-fold OOF-предсказания каждой базовой модели.
    Уровень 1: мета-модель (ridge/lightgbm/...) на OOF; отчёт MSE через K-fold на OOF-признаках.
    Test: базы обучены на полном train → мета на матрице тестовых предсказаний.
    """
    if not base_models:
        raise ValueError("stacking: base_models must be non-empty")

    train_path = validate_path(str(train_path), allowed_dirs, must_exist=True)
    test_path = validate_path(str(test_path), allowed_dirs, must_exist=True)
    submission_path = validate_path(str(submission_path), allowed_dirs, must_exist=False)
    bundle_path = validate_path(str(bundle_path), allowed_dirs, must_exist=False)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_path)
    y = df["target"].astype(float)
    X = df.drop(columns=["target"], errors="ignore")
    df_test = pd.read_csv(test_path)
    X_test = df_test.drop(columns=["target"], errors="ignore") if "target" in df_test.columns else df_test.copy()
    for c in X.columns:
        if c not in X_test.columns:
            X_test[c] = np.nan
    X_test = X_test[X.columns]

    n = len(X)
    n_base = len(base_models)
    oof = np.zeros((n, n_base), dtype=float)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    for j, row in enumerate(base_models):
        mname = str(row.get("model", "lightgbm")).lower()
        p = validate_model_params(mname, _params_except_model(row))
        for tr_idx, va_idx in kf.split(X):
            model = _get_model(mname, {**p, "random_state": random_state})
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            oof[va_idx, j] = model.predict(X.iloc[va_idx])

    meta_cfg = meta_config or {}
    meta_name = str(meta_cfg.get("model", "ridge")).lower()
    meta_raw = {k: v for k, v in meta_cfg.items() if k != "model"}
    meta_params = validate_model_params(meta_name, meta_raw)

    # OOF MSE мета-модели (вложенный K-fold по строкам train)
    meta_cv = KFold(n_splits=min(cv_folds, 5), shuffle=True, random_state=random_state + 1)
    meta_scores: list[float] = []
    for m_tr, m_va in meta_cv.split(oof):
        meta = _get_meta_model(meta_name, meta_params, random_state)
        meta.fit(oof[m_tr], y.iloc[m_tr])
        pred = meta.predict(oof[m_va])
        meta_scores.append(float(mean_squared_error(y.iloc[m_va], pred)))

    meta_final = _get_meta_model(meta_name, meta_params, random_state)
    meta_final.fit(oof, y)

    test_mat = np.zeros((len(X_test), n_base), dtype=float)
    base_fitted: list[tuple[str, dict, Any]] = []
    for j, row in enumerate(base_models):
        mname = str(row.get("model", "lightgbm")).lower()
        p = validate_model_params(mname, _params_except_model(row))
        model = _get_model(mname, {**p, "random_state": random_state})
        model.fit(X, y)
        test_mat[:, j] = model.predict(X_test)
        base_fitted.append((mname, p, model))

    pred_test = np.asarray(meta_final.predict(test_mat), dtype=float).ravel()

    joblib.dump(
        {
            "kind": "oof_stacking",
            "meta_name": meta_name,
            "meta_params": meta_params,
            "meta_model": meta_final,
            "base_fitted": base_fitted,
            "base_configs": base_models,
        },
        bundle_path,
    )

    write_submission_array(pred_test, str(submission_path), allowed_dirs, robustness=robustness)

    return {
        "val_mse": float(np.mean(meta_scores)),
        "cv_mse_mean": float(np.mean(meta_scores)),
        "cv_mse_std": float(np.std(meta_scores)),
        "model_path": str(bundle_path),
        "stacking_oof": True,
        "n_base_models": n_base,
        "meta_model": meta_name,
    }


def train_with_pseudo_labels(
    name: str,
    params: dict[str, Any],
    train_path: str,
    test_path: str,
    model_save_path: str,
    submission_path: str,
    allowed_dirs: list[str],
    rounds: int,
    random_state: int,
    cv_folds: int | None,
    robustness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Итерации: обучение на train → предсказание test → добавление (X_test, pred) к обучающей выборке → повтор.
    Сабмит — предсказание последней модели, обученной на расширенной выборке (после всех раундов псевдо-лейблов).
    """
    rounds = max(1, int(rounds))
    params = validate_model_params(name, params or {})
    train_path = validate_path(str(train_path), allowed_dirs, must_exist=True)
    test_path = validate_path(str(test_path), allowed_dirs, must_exist=True)
    model_save_path = validate_path(str(model_save_path), allowed_dirs, must_exist=False)
    submission_path = validate_path(str(submission_path), allowed_dirs, must_exist=False)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_path)
    y = df["target"].astype(float)
    X = df.drop(columns=["target"], errors="ignore")
    df_test = pd.read_csv(test_path)
    X_test = df_test.drop(columns=["target"], errors="ignore") if "target" in df_test.columns else df_test.copy()
    for c in X.columns:
        if c not in X_test.columns:
            X_test[c] = np.nan
    X_test = X_test[X.columns]

    X_aug, y_aug = X.copy(), y.copy()
    last_model = None
    for r in range(rounds):
        p = dict(params)
        p["random_state"] = random_state + r
        model = _get_model(name, p)
        model.fit(X_aug, y_aug)
        last_model = model
        pseudo = np.asarray(model.predict(X_test), dtype=float).ravel()
        if r < rounds - 1:
            X_aug = pd.concat([X_aug, X_test], axis=0, ignore_index=True)
            y_aug = pd.concat([y_aug, pd.Series(pseudo)], axis=0, ignore_index=True)

    assert last_model is not None
    pred = np.asarray(last_model.predict(X_test), dtype=float).ravel()
    joblib.dump(last_model, model_save_path)
    write_submission_array(pred, str(submission_path), allowed_dirs, robustness=robustness)

    result: dict[str, Any] = {
        "val_mse": None,
        "model_path": str(model_save_path),
        "pseudo_labels_rounds": rounds,
        "pseudo_labels": True,
    }
    if cv_folds is not None and cv_folds >= 2:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores: list[float] = []
        for tr_idx, va_idx in kf.split(X):
            m = _get_model(name, {**params, "random_state": random_state})
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pred_v = m.predict(X.iloc[va_idx])
            scores.append(float(mean_squared_error(y.iloc[va_idx], pred_v)))
        result["cv_mse_mean"] = float(np.mean(scores))
        result["val_mse"] = result["cv_mse_mean"]
    return result
