"""
Guardrails: no arbitrary code execution; only whitelisted tool calls.
"""

from __future__ import annotations

from src.security.validation import validate_model_params

ALLOWED_TOOL_NAMES = frozenset(
    {
        "load_and_summarize",
        "get_missing_and_correlation",
        "save_eda_report",
        "fit_preprocessor",
        "transform_train_test",
        "train_regressor",
        "make_submission",
    }
)


def is_allowed_tool(name: str) -> bool:
    """Check if tool name is in whitelist."""
    return name in ALLOWED_TOOL_NAMES


def validate_tool_call(name: str, arguments: dict | None) -> None:
    """Raise if tool is not allowed or arguments are invalid."""
    if arguments is None:
        arguments = {}
    if not is_allowed_tool(name):
        raise ValueError(f"Tool not allowed: {name}")

    if name in ("load_and_summarize", "get_missing_and_correlation"):
        path = arguments.get("path") or arguments.get("file_path")
        if not path or not str(path).strip():
            raise ValueError(f"{name}: path is required")

    if name == "save_eda_report":
        if "content" not in arguments and "report" not in arguments:
            raise ValueError("save_eda_report: content or report is required")
        out = arguments.get("output_path")
        if not out or not str(out).strip():
            raise ValueError("save_eda_report: output_path is required")

    if name == "train_regressor":
        model = arguments.get("model", "lightgbm")
        params = {
            "n_estimators": arguments.get("n_estimators"),
            "max_depth": arguments.get("max_depth"),
            "learning_rate": arguments.get("learning_rate"),
            "alpha": arguments.get("alpha"),
            "min_samples_leaf": arguments.get("min_samples_leaf"),
            "min_child_samples": arguments.get("min_child_samples"),
            "num_leaves": arguments.get("num_leaves"),
        }
        params = {k: v for k, v in params.items() if v is not None}
        validate_model_params(str(model), params)
