"""
Grid / Cartesian search over model configs on processed train (train_regressor).
Used by Coordinator after Engineer when evaluation.hparam_search.enabled is true.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from src.tools import model_tools

# Параметры, которые передаются в validate_model_params / train_regressor
PARAM_KEYS = (
    "n_estimators",
    "max_depth",
    "learning_rate",
    "alpha",
    "min_child_samples",
    "num_leaves",
)

BOOSTING_MODELS = frozenset({"lightgbm", "xgboost", "catboost", "random_forest"})


def _metric_from_result(res: dict[str, Any], cv_folds: int | None) -> float | None:
    if cv_folds is not None and cv_folds >= 2:
        return res.get("cv_mse_mean")
    return res.get("val_mse")


def build_search_grid(hparam_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Собирает список конфигураций из:
    - `grid` — явный список словарей;
    - `cartesian` — декартово произведение списков (одна модель + оси гиперпараметров).
    Порядок: сначала строки из `grid`, затем развёртка `cartesian`.
    Ограничение `max_trials` обрезает итоговый список с начала (после объединения).
    """
    explicit: list[dict[str, Any]] = []
    for row in hparam_cfg.get("grid") or []:
        if isinstance(row, dict):
            explicit.append(dict(row))

    cart = hparam_cfg.get("cartesian")
    if cart and isinstance(cart, dict):
        model = str(cart.get("model", "lightgbm")).lower()
        keys: list[str] = []
        vals: list[list[Any]] = []
        for k, v in cart.items():
            if k == "model":
                continue
            if isinstance(v, (list, tuple)):
                keys.append(k)
                vals.append(list(v))
            else:
                keys.append(k)
                vals.append([v])
        if keys:
            for combo in itertools.product(*vals):
                row = {"model": model}
                for k, val in zip(keys, combo):
                    row[k] = val
                explicit.append(row)

    max_t = hparam_cfg.get("max_trials")
    if max_t is not None and len(explicit) > int(max_t):
        explicit = explicit[: int(max_t)]
    return explicit


def _params_from_row(item: dict[str, Any]) -> dict[str, Any]:
    return {k: item[k] for k in PARAM_KEYS if k in item}


def _trial_to_best(trial: dict[str, Any]) -> dict[str, Any]:
    cfg = trial["config"]
    model_name = str(cfg.get("model", "lightgbm")).lower()
    return {
        "model": model_name,
        "params": _params_from_row(cfg),
        "val_mse": trial["val_mse"],
        "trial_index": trial["trial_index"],
    }


def select_best_with_policy(
    best: dict[str, Any] | None,
    trials: list[dict[str, Any]],
    policy: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Если лучший по CV — линейная модель (ridge), но есть бустинг с MSE не хуже
    чем (1 + tie_relative_tolerance) × лучший MSE — выбираем бустинг (стабильнее на Kaggle).
    """
    policy = policy or {}
    if best is None or not trials:
        return best, None
    if not policy.get("prefer_boosting", True):
        return best, None
    if best["model"] in BOOSTING_MODELS:
        return best, None
    rel = float(policy.get("tie_relative_tolerance", 0.05))
    threshold = float(best["val_mse"]) * (1.0 + rel)
    ranked = [t for t in trials if t.get("val_mse") is not None]
    ranked.sort(key=lambda x: float(x["val_mse"]))
    for t in ranked:
        m = str(t["config"].get("model", "")).lower()
        if m in BOOSTING_MODELS and float(t["val_mse"]) <= threshold:
            return _trial_to_best(t), "prefer_boosting_within_tolerance"
    return best, None


def run_hparam_grid(
    train_path: str,
    allowed_dirs: list[str],
    artifacts_dir: str,
    grid: list[dict[str, Any]],
    evaluation_cfg: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """
    Evaluate each grid row with fit_full_train=False (holdout or CV per evaluation_cfg).
    evaluation_cfg may contain _hparam_cv_folds to override cv_folds только для поиска.
    Returns (best, all_trials) where best has keys: model, params, val_mse, trial_index.
    """
    if not grid:
        return None, []

    train_path = str(Path(train_path).resolve())
    artifacts_dir = str(Path(artifacts_dir).resolve())
    model_save_path = str(Path(artifacts_dir) / "model_hparam_trial.joblib")

    val_ratio = float(evaluation_cfg.get("val_ratio", 0.2))
    cv_raw = evaluation_cfg.get("_hparam_cv_folds", evaluation_cfg.get("cv_folds", 0))
    cv_eff: int | None = None if cv_raw is None or int(cv_raw) == 0 else int(cv_raw)
    rs = int(evaluation_cfg.get("random_state", 42))

    best: dict[str, Any] | None = None
    best_mse = float("inf")
    trials: list[dict[str, Any]] = []

    for i, item in enumerate(grid):
        model_name = str(item.get("model", "lightgbm")).lower()
        params = _params_from_row(item)
        res = model_tools.train_regressor(
            name=model_name,
            params=params,
            train_path=train_path,
            target_col="target",
            val_path=None,
            model_save_path=model_save_path,
            allowed_dirs=allowed_dirs,
            val_ratio=val_ratio,
            random_state=rs,
            cv_folds=cv_eff,
            fit_full_train=False,
        )
        mse = _metric_from_result(res, cv_eff)
        row = {
            "trial_index": i,
            "config": dict(item),
            "val_mse": mse,
            "result_keys": list(res.keys()),
        }
        trials.append(row)
        if mse is not None and mse < best_mse:
            best_mse = mse
            best = {
                "model": model_name,
                "params": params,
                "val_mse": mse,
                "trial_index": i,
            }

    return best, trials
