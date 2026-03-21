"""
Optuna hyperparameter search (Bayesian optimization) for tabular regression.
Requires: pip install optuna

Поддержка: одна модель или список `optuna.models` (например lightgbm + catboost),
доп. регуляризация LightGBM (reg_alpha, reg_lambda, subsample, colsample_bytree).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.tools import model_tools

from src.evaluation.hparam_search import _metric_from_result


def _resolve_models(oc: dict[str, Any]) -> list[str]:
    m = oc.get("models")
    if isinstance(m, (list, tuple)) and len(m) >= 1:
        return [str(x).lower().strip() for x in m]
    return [str(oc.get("model", "lightgbm")).lower().strip()]


def _suggest_params(trial: Any, model_name: str, oc: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if model_name == "lightgbm":
        params["n_estimators"] = trial.suggest_int(
            "n_estimators", int(oc.get("n_estimators_low", 100)), int(oc.get("n_estimators_high", 500))
        )
        params["max_depth"] = trial.suggest_int(
            "max_depth", int(oc.get("max_depth_low", 4)), int(oc.get("max_depth_high", 12))
        )
        params["learning_rate"] = trial.suggest_float(
            "learning_rate",
            float(oc.get("learning_rate_low", 0.02)),
            float(oc.get("learning_rate_high", 0.2)),
            log=True,
        )
        params["num_leaves"] = trial.suggest_int(
            "num_leaves", int(oc.get("num_leaves_low", 16)), int(oc.get("num_leaves_high", 127))
        )
        params["min_child_samples"] = trial.suggest_int(
            "min_child_samples",
            int(oc.get("min_child_samples_low", 5)),
            int(oc.get("min_child_samples_high", 100)),
        )
        if oc.get("lgbm_regularization", True):
            params["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-9, 10.0, log=True)
            params["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-9, 10.0, log=True)
            params["subsample"] = trial.suggest_float(
                "subsample",
                float(oc.get("subsample_low", 0.6)),
                float(oc.get("subsample_high", 1.0)),
            )
            params["colsample_bytree"] = trial.suggest_float(
                "colsample_bytree",
                float(oc.get("colsample_bytree_low", 0.6)),
                float(oc.get("colsample_bytree_high", 1.0)),
            )
    elif model_name == "ridge":
        params["alpha"] = trial.suggest_float("alpha", 1e-4, 100.0, log=True)
    elif model_name == "catboost":
        # Те же имена гиперпараметров, что у LightGBM, чтобы Optuna не дублировала пространство поиска.
        params["n_estimators"] = trial.suggest_int(
            "n_estimators", int(oc.get("n_estimators_low", 100)), int(oc.get("n_estimators_high", 500))
        )
        params["max_depth"] = trial.suggest_int(
            "max_depth", int(oc.get("max_depth_low", 4)), int(oc.get("max_depth_high", 12))
        )
        params["learning_rate"] = trial.suggest_float(
            "learning_rate",
            float(oc.get("learning_rate_low", 0.02)),
            float(oc.get("learning_rate_high", 0.2)),
            log=True,
        )
    else:
        params["n_estimators"] = trial.suggest_int("n_estimators", 50, 400)
        params["max_depth"] = trial.suggest_int("max_depth", 4, 16)
        params["learning_rate"] = trial.suggest_float("learning_rate", 0.02, 0.2, log=True)
    return params


def _params_for_best(best_params: dict[str, Any], model_name: str) -> dict[str, Any]:
    if model_name == "ridge":
        return {"alpha": float(best_params.get("alpha", 1.0))}
    if model_name == "lightgbm":
        keys = (
            "n_estimators",
            "max_depth",
            "learning_rate",
            "num_leaves",
            "min_child_samples",
            "reg_alpha",
            "reg_lambda",
            "subsample",
            "colsample_bytree",
        )
        return {k: best_params[k] for k in keys if k in best_params}
    if model_name == "catboost":
        keys = ("n_estimators", "max_depth", "learning_rate")
        return {k: best_params[k] for k in keys if k in best_params}
    return {k: v for k, v in best_params.items() if k != "model"}


def run_optuna_search(
    train_path: str,
    allowed_dirs: list[str],
    artifacts_dir: str,
    evaluation_cfg: dict[str, Any],
    hparam_cfg: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """
    Minimize CV / holdout MSE via Optuna. Returns (best, trials_log) same shape as grid search.
    hparam_cfg.optuna: model или models, n_trials, seed, диапазоны.
    """
    try:
        import optuna
    except ImportError as e:
        raise ImportError("Optuna search requires: pip install optuna") from e

    oc = hparam_cfg.get("optuna") or {}
    n_trials = int(oc.get("n_trials", 25))
    seed = int(oc.get("seed", evaluation_cfg.get("random_state", 42)))
    models = _resolve_models(oc)
    multi = len(models) > 1

    train_path = str(Path(train_path).resolve())
    artifacts_dir = str(Path(artifacts_dir).resolve())
    model_save_path = str(Path(artifacts_dir) / "model_hparam_trial.joblib")

    val_ratio = float(evaluation_cfg.get("val_ratio", 0.2))
    cv_raw = evaluation_cfg.get("_hparam_cv_folds", evaluation_cfg.get("cv_folds", 0))
    cv_eff: int | None = None if cv_raw is None or int(cv_raw) == 0 else int(cv_raw)
    rs = int(evaluation_cfg.get("random_state", 42))

    trials_log: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        if multi:
            model_name = str(trial.suggest_categorical("model", models)).lower()
        else:
            model_name = models[0]
        params = _suggest_params(trial, model_name, oc)

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
        val = float(mse) if mse is not None else 1e18
        cfg = {"model": model_name, **params}
        trials_log.append(
            {
                "trial_index": trial.number,
                "config": cfg,
                "val_mse": mse,
                "result_keys": list(res.keys()),
            }
        )
        return val

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    if len(study.trials) == 0:
        return None, trials_log
    try:
        bt = study.best_trial
    except ValueError:
        return None, trials_log
    best_params = dict(bt.params)
    mn = best_params.get("model")
    if mn is not None:
        mn = str(mn).lower()
    else:
        mn = models[0]

    best = {
        "model": mn,
        "params": _params_for_best(best_params, mn),
        "val_mse": float(bt.value),
        "trial_index": bt.number,
    }
    return best, trials_log
