"""
Optuna hyperparameter search (Bayesian optimization) for tabular regression.
Requires: pip install optuna
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.tools import model_tools

from src.evaluation.hparam_search import _metric_from_result


def run_optuna_search(
    train_path: str,
    allowed_dirs: list[str],
    artifacts_dir: str,
    evaluation_cfg: dict[str, Any],
    hparam_cfg: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """
    Minimize CV / holdout MSE via Optuna. Returns (best, trials_log) same shape as grid search.
    hparam_cfg.optuna: model, n_trials, seed, and int/float ranges.
    """
    try:
        import optuna
    except ImportError as e:
        raise ImportError("Optuna search requires: pip install optuna") from e

    oc = hparam_cfg.get("optuna") or {}
    n_trials = int(oc.get("n_trials", 25))
    seed = int(oc.get("seed", evaluation_cfg.get("random_state", 42)))
    model_name = str(oc.get("model", "lightgbm")).lower()

    train_path = str(Path(train_path).resolve())
    artifacts_dir = str(Path(artifacts_dir).resolve())
    model_save_path = str(Path(artifacts_dir) / "model_hparam_trial.joblib")

    val_ratio = float(evaluation_cfg.get("val_ratio", 0.2))
    cv_raw = evaluation_cfg.get("_hparam_cv_folds", evaluation_cfg.get("cv_folds", 0))
    cv_eff: int | None = None if cv_raw is None or int(cv_raw) == 0 else int(cv_raw)
    rs = int(evaluation_cfg.get("random_state", 42))

    trials_log: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        params: dict[str, Any] = {}
        if model_name == "lightgbm":
            params["n_estimators"] = trial.suggest_int(
                "n_estimators", int(oc.get("n_estimators_low", 100)), int(oc.get("n_estimators_high", 500))
            )
            params["max_depth"] = trial.suggest_int(
                "max_depth", int(oc.get("max_depth_low", 4)), int(oc.get("max_depth_high", 12))
            )
            params["learning_rate"] = trial.suggest_float(
                "learning_rate", float(oc.get("learning_rate_low", 0.02)), float(oc.get("learning_rate_high", 0.2)), log=True
            )
            params["num_leaves"] = trial.suggest_int(
                "num_leaves", int(oc.get("num_leaves_low", 16)), int(oc.get("num_leaves_high", 127))
            )
            params["min_child_samples"] = trial.suggest_int(
                "min_child_samples", int(oc.get("min_child_samples_low", 5)), int(oc.get("min_child_samples_high", 100))
            )
        elif model_name == "ridge":
            params["alpha"] = trial.suggest_float("alpha", 1e-4, 100.0, log=True)
        else:
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 400)
            params["max_depth"] = trial.suggest_int("max_depth", 4, 16)
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.02, 0.2, log=True)

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
        trials_log.append(
            {
                "trial_index": trial.number,
                "config": {"model": model_name, **params},
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
    if model_name == "ridge":
        best = {
            "model": model_name,
            "params": {"alpha": best_params.get("alpha", 1.0)},
            "val_mse": float(bt.value),
            "trial_index": bt.number,
        }
    elif model_name == "lightgbm":
        best = {
            "model": model_name,
            "params": {
                k: best_params[k]
                for k in (
                    "n_estimators",
                    "max_depth",
                    "learning_rate",
                    "num_leaves",
                    "min_child_samples",
                )
                if k in best_params
            },
            "val_mse": float(bt.value),
            "trial_index": bt.number,
        }
    else:
        best = {
            "model": model_name,
            "params": {k: v for k, v in best_params.items()},
            "val_mse": float(bt.value),
            "trial_index": bt.number,
        }

    return best, trials_log
