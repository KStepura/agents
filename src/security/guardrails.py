"""
Guardrails: no arbitrary code execution; only whitelisted tool calls.
"""

from __future__ import annotations

ALLOWED_TOOL_NAMES = {
    "load_dataset", "get_dtypes", "describe", "missing_report",
    "correlation_with_target", "save_eda_report",
    "load_eda_report", "fit_preprocessor", "transform_train_test", "select_features",
    "train_regressor", "evaluate_mse", "make_submission",
}


def is_allowed_tool(name: str) -> bool:
    """Check if tool name is in whitelist."""
    return name in ALLOWED_TOOL_NAMES


def validate_tool_call(name: str, arguments: dict) -> None:
    """Raise if tool is not allowed or arguments are invalid."""
    if not is_allowed_tool(name):
        raise ValueError(f"Tool not allowed: {name}")
    # Optional: validate argument keys and types per tool
    # ...
