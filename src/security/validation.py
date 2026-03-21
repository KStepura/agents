"""
Input validation for tool calls: paths, model params, etc.
"""

from __future__ import annotations

from pathlib import Path

ALLOWED_MODELS = frozenset({"ridge", "random_forest", "xgboost", "lightgbm", "catboost"})


def validate_path(path: str, allowed_dirs: list[str], must_exist: bool = True) -> Path:
    """Ensure path is under one of allowed_dirs and optionally exists."""
    resolved = Path(path).resolve()
    for d in allowed_dirs:
        base = Path(d).resolve()
        try:
            resolved.relative_to(base)
            if must_exist and not resolved.exists():
                raise FileNotFoundError(f"Path does not exist: {resolved}")
            return resolved
        except ValueError:
            continue
    raise PermissionError(f"Path not allowed: {path}")


def validate_model_params(name: str, params: dict | None) -> dict:
    """
    Whitelist model name and bound hyperparameters.
    Returns a sanitized dict safe to pass to model_tools._get_model / train_regressor.
    """
    if params is None:
        params = {}
    name_l = (name or "lightgbm").lower().strip()
    if name_l not in ALLOWED_MODELS:
        raise ValueError(f"Model must be one of {sorted(ALLOWED_MODELS)}, got {name!r}")

    out: dict = {"random_state": 42}

    if name_l == "ridge":
        alpha = float(params.get("alpha", 1.0))
        return {
            "alpha": max(1e-8, min(alpha, 1e6)),
            "random_state": 42,
        }

    n_est = int(params.get("n_estimators", 100))
    out["n_estimators"] = max(1, min(n_est, 500))

    md = params.get("max_depth")
    if md is not None:
        md = int(md)
        out["max_depth"] = max(1, min(md, 64))

    lr = params.get("learning_rate")
    if lr is not None:
        lr = float(lr)
        out["learning_rate"] = max(1e-4, min(lr, 1.0))

    if name_l == "random_forest":
        msl = params.get("min_samples_leaf", 1)
        out["min_samples_leaf"] = max(1, min(int(msl), 100))

    if name_l == "lightgbm":
        mcs = params.get("min_child_samples")
        if mcs is not None:
            out["min_child_samples"] = max(1, min(int(mcs), 500))
        nl = params.get("num_leaves")
        if nl is not None:
            out["num_leaves"] = max(2, min(int(nl), 512))

    return out
