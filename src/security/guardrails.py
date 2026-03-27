"""
Ограничения (guardrails): запрет на выполнение произвольного кода; разрешены только вызовы инструментов из белого списка.
"""

from __future__ import annotations

from src.security.sanitize import sanitize_large_text, sanitize_path_argument
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

_tool_calls_seen = 0
DEFAULT_MAX_TOOL_CALLS_PER_RUN = 5000


def reset_tool_call_budget() -> None:
    global _tool_calls_seen
    _tool_calls_seen = 0


def is_allowed_tool(name: str) -> bool:
    """Проверяет, входит ли имя инструмента в белый список."""
    return name in ALLOWED_TOOL_NAMES


def validate_tool_call(name: str, arguments: dict | None) -> None:
    """Вызывает ошибку, если инструмент не разрешён или аргументы некорректны."""
    global _tool_calls_seen
    if arguments is None:
        arguments = {}
    if not is_allowed_tool(name):
        raise ValueError(f"Tool not allowed: {name}")

    _tool_calls_seen += 1
    if _tool_calls_seen > DEFAULT_MAX_TOOL_CALLS_PER_RUN:
        raise ValueError(
            f"Превышен лимит вызовов инструментов за прогон ({DEFAULT_MAX_TOOL_CALLS_PER_RUN})"
        )

    if name in ("load_and_summarize", "get_missing_and_correlation"):
        path = arguments.get("path") or arguments.get("file_path")
        if not path or not str(path).strip():
            raise ValueError(f"{name}: path is required")
        sanitize_path_argument(str(path).strip(), field=f"{name}.path")

    if name == "save_eda_report":
        raw = arguments.get("content")
        if raw is None:
            raw = arguments.get("report")
        if raw is None:
            raise ValueError("save_eda_report: content or report is required")
        sanitize_large_text(raw, field="save_eda_report.content")
        out = arguments.get("output_path")
        if not out or not str(out).strip():
            raise ValueError("save_eda_report: output_path is required")
        sanitize_path_argument(str(out).strip(), field="save_eda_report.output_path")

    if name == "fit_preprocessor":
        for key in ("train_path", "pipeline_save_path"):
            v = arguments.get(key)
            if v is not None and str(v).strip():
                sanitize_path_argument(str(v).strip(), field=f"fit_preprocessor.{key}")

    if name == "transform_train_test":
        for key in ("pipeline_path", "train_path", "test_path"):
            v = arguments.get(key)
            if v is not None and str(v).strip():
                sanitize_path_argument(str(v).strip(), field=f"transform_train_test.{key}")

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
            "reg_alpha": arguments.get("reg_alpha"),
            "reg_lambda": arguments.get("reg_lambda"),
            "subsample": arguments.get("subsample"),
            "colsample_bytree": arguments.get("colsample_bytree"),
        }
        params = {k: v for k, v in params.items() if v is not None}
        validate_model_params(str(model), params)
        tp = arguments.get("train_path")
        if tp is not None and str(tp).strip():
            sanitize_path_argument(str(tp).strip(), field="train_regressor.train_path")

    if name == "make_submission":
        for key in ("model_path", "test_path", "output_path"):
            v = arguments.get(key)
            if v is not None and str(v).strip():
                sanitize_path_argument(str(v).strip(), field=f"make_submission.{key}")
