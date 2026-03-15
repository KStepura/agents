"""
Input validation for tool calls: paths, model params, etc.
"""

from __future__ import annotations

from pathlib import Path


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


def validate_model_params(name: str, params: dict) -> dict:
    """Whitelist and bound model hyperparameters (e.g. max_depth in [1, 100])."""
    # Define allowed ranges per model
    raise NotImplementedError("Implement: allowed keys and ranges per model name")
