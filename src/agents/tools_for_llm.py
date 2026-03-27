"""
Схемы инструментов (в формате функций OpenAI) и их исполнители для агентов.
Все исполнители принимают (name, args) и возвращают строку для LLM.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from src.monitoring.logger import log_tool_call
from src.security.guardrails import validate_tool_call

def _explorer_tools_executor(allowed_dirs: list[str], data_dir: str, agent: str = "explorer"):
    from src.tools import eda_tools

    data_dir = str(Path(data_dir).resolve())

    def _run(name: str, args: dict, fn: Callable[[], str]) -> str:
        validate_tool_call(name, args)
        t0 = time.perf_counter()
        try:
            return fn()
        finally:
            log_tool_call(agent, name, (time.perf_counter() - t0) * 1000)

    def executor(name: str, args: dict) -> str:
        path = args.get("path", args.get("file_path", ""))
        if path and not path.startswith("/") and path not in ("data", "artifacts"):
            full = Path(data_dir) / path
            if full.exists() or path in ("train.csv", "test.csv"):
                path = str(full)
        if name == "load_and_summarize":
            def fn() -> str:
                df = eda_tools.load_dataset(path, allowed_dirs)
                dtypes = eda_tools.get_dtypes(df)
                desc = eda_tools.describe(df)
                return (
                    f"Shape: {df.shape}. Dtypes: {json.dumps(dtypes)}. "
                    f"Describe (numeric): {json.dumps(desc, default=str)[:2000]}."
                )
            return _run(name, args, fn)
        if name == "get_missing_and_correlation":
            def fn() -> str:
                df = eda_tools.load_dataset(path, allowed_dirs)
                miss = eda_tools.missing_report(df)
                corr = eda_tools.correlation_with_target(df, "target") if "target" in df.columns else {}
                return f"Missing: {json.dumps(miss)}. Correlation with target: {json.dumps(corr)}."
            return _run(name, args, fn)
        if name == "save_eda_report":
            def fn() -> str:
                content = args.get("content", args.get("report", ""))
                out = args.get("output_path", "artifacts/eda_report.txt")
                out = str(Path(out).resolve())
                return eda_tools.save_eda_report(content, out, allowed_dirs)
            return _run(name, args, fn)
        return "Unknown tool"
    return executor


EXPLORER_TOOLS_SCHEMA = [
    {
        "name": "load_and_summarize",
        "description": "Load a CSV file (train or test) and return shape, dtypes, and basic stats. Path can be relative to data dir, e.g. train.csv or data/train.csv.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to CSV file"}},
            "required": ["path"],
        },
    },
    {
        "name": "get_missing_and_correlation",
        "description": "Load CSV and return missing value counts and correlation with target (if column exists).",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "save_eda_report",
        "description": "Save EDA report (text or JSON string) to a file in artifacts. Use output_path like artifacts/eda_report.txt or artifacts/eda_report.json.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Report content"},
                "output_path": {"type": "string", "description": "Path to save, e.g. artifacts/eda_report.txt"},
            },
            "required": ["content", "output_path"],
        },
    },
]


def _engineer_tools_executor(
    allowed_dirs: list[str],
    artifacts_dir: str,
    data_dir: str = "data",
    agent: str = "engineer",
    *,
    default_encoding: str = "te_freq",
    pipeline_defaults: dict[str, Any] | None = None,
):
    from src.tools import feature_tools

    data_dir = str(Path(data_dir).resolve())
    pd = pipeline_defaults or {}

    def _run(name: str, args: dict, fn: Callable[[], str]) -> str:
        validate_tool_call(name, args)
        t0 = time.perf_counter()
        try:
            return fn()
        finally:
            log_tool_call(agent, name, (time.perf_counter() - t0) * 1000)

    def executor(name: str, args: dict) -> str:
        if name == "fit_preprocessor":
            def fn() -> str:
                train_path = args.get("train_path") or f"{data_dir}/train.csv"
                pipeline_path = args.get("pipeline_save_path", f"{artifacts_dir}/preprocessor.joblib")
                enc = args.get("encoding") or pd.get("encoding") or default_encoding
                cfg: dict[str, Any] = {"pipeline_save_path": pipeline_path, "encoding": enc}
                rc = args.get("rare_category_min_count")
                if rc is not None:
                    cfg["rare_category_min_count"] = int(rc)
                elif pd.get("rare_category_min_count") is not None:
                    cfg["rare_category_min_count"] = int(pd["rare_category_min_count"])
                p = feature_tools.fit_preprocessor(
                    config=cfg,
                    train_path=train_path,
                    target_col="target",
                    allowed_dirs=allowed_dirs,
                )
                return f"Preprocessor saved to {p} (encoding={enc})"
            return _run(name, args, fn)
        if name == "transform_train_test":
            def fn() -> str:
                pipeline_path = args.get("pipeline_path", f"{artifacts_dir}/preprocessor.joblib")
                train_path = args.get("train_path") or f"{data_dir}/train.csv"
                test_path = args.get("test_path") or f"{data_dir}/test.csv"
                out = feature_tools.transform_train_test(
                    pipeline_path=pipeline_path,
                    train_path=train_path,
                    test_path=test_path,
                    output_dir=artifacts_dir,
                    allowed_dirs=allowed_dirs,
                )
                return json.dumps(out)
            return _run(name, args, fn)
        return "Unknown tool"
    return executor


ENGINEER_TOOLS_SCHEMA = [
    {
        "name": "fit_preprocessor",
        "description": "Fit preprocessing pipeline on train data and save to artifacts. Use train_path e.g. data/train.csv and pipeline_save_path e.g. artifacts/preprocessor.joblib.",
        "parameters": {
            "type": "object",
            "properties": {
                "train_path": {"type": "string"},
                "pipeline_save_path": {"type": "string"},
                "encoding": {
                    "type": "string",
                    "enum": ["te_freq", "ohe"],
                    "description": "Optional; defaults to project config (te_freq recommended).",
                },
                "rare_category_min_count": {
                    "type": "integer",
                    "description": "Optional; categories with count < this on train → Other (te_freq).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "transform_train_test",
        "description": "Apply saved preprocessor to train and test, save processed CSVs to artifacts. Returns paths to train_processed.csv and test_processed.csv.",
        "parameters": {
            "type": "object",
            "properties": {
                "pipeline_path": {"type": "string"},
                "train_path": {"type": "string"},
                "test_path": {"type": "string"},
            },
            "required": [],
        },
    },
]


def _builder_tools_executor(
    allowed_dirs: list[str],
    artifacts_dir: str,
    submissions_dir: str,
    agent: str = "builder",
    *,
    val_ratio: float = 0.2,
    cv_folds: int | None = None,
    fit_full_train: bool = False,
    random_state: int = 42,
    robustness: dict[str, Any] | None = None,
):
    from src.tools import model_tools

    def _run(name: str, args: dict, fn: Callable[[], str]) -> str:
        validate_tool_call(name, args)
        t0 = time.perf_counter()
        try:
            return fn()
        finally:
            log_tool_call(agent, name, (time.perf_counter() - t0) * 1000)

    def executor(name: str, args: dict) -> str:
        if name == "train_regressor":
            def fn() -> str:
                train_path = args.get("train_path", f"{artifacts_dir}/train_processed.csv")
                model_name = args.get("model", "lightgbm")
                params: dict[str, Any] = {}
                if args.get("n_estimators") is not None:
                    params["n_estimators"] = args["n_estimators"]
                if args.get("max_depth") is not None:
                    params["max_depth"] = args["max_depth"]
                if args.get("learning_rate") is not None:
                    params["learning_rate"] = args["learning_rate"]
                if args.get("alpha") is not None:
                    params["alpha"] = args["alpha"]
                if args.get("min_child_samples") is not None:
                    params["min_child_samples"] = args["min_child_samples"]
                if args.get("num_leaves") is not None:
                    params["num_leaves"] = args["num_leaves"]
                vr = float(args["val_ratio"]) if "val_ratio" in args else val_ratio
                if "cv_folds" in args:
                    cv_raw = args["cv_folds"]
                    if cv_raw is None or cv_raw == 0:
                        cv_eff = None
                    else:
                        cv_eff = int(cv_raw)
                else:
                    cv_eff = cv_folds
                fft = bool(args["fit_full_train"]) if "fit_full_train" in args else fit_full_train
                rs = int(args["random_state"]) if "random_state" in args else random_state
                res = model_tools.train_regressor(
                    name=model_name,
                    params=params,
                    train_path=train_path,
                    target_col="target",
                    val_path=None,
                    model_save_path=f"{artifacts_dir}/model.joblib",
                    allowed_dirs=allowed_dirs,
                    val_ratio=vr,
                    random_state=rs,
                    cv_folds=cv_eff,
                    fit_full_train=fft,
                )
                return json.dumps(res)
            return _run(name, args, fn)
        if name == "make_submission":
            def fn() -> str:
                model_path = args.get("model_path", f"{artifacts_dir}/model.joblib")
                test_path = args.get("test_path", f"{artifacts_dir}/test_processed.csv")
                out_path = args.get("output_path", f"{submissions_dir}/submission.csv")
                p = model_tools.make_submission(
                    model_path=model_path,
                    test_path=test_path,
                    output_path=out_path,
                    index_col=None,
                    allowed_dirs=allowed_dirs,
                    robustness=robustness,
                )
                return f"Submission saved to {p}"
            return _run(name, args, fn)
        return "Unknown tool"
    return executor


BUILDER_TOOLS_SCHEMA = [
    {
        "name": "train_regressor",
        "description": "Train a regressor. Defaults for val_ratio, cv_folds, fit_full_train come from project config if omitted. Recommended: lightgbm n_estimators 300, max_depth 8.",
        "parameters": {
            "type": "object",
            "properties": {
                "train_path": {"type": "string"},
                "model": {"type": "string", "enum": ["ridge", "random_forest", "xgboost", "lightgbm", "catboost"]},
                "n_estimators": {"type": "integer"},
                "max_depth": {"type": "integer"},
                "learning_rate": {"type": "number"},
                "min_child_samples": {"type": "integer", "description": "LightGBM only"},
                "num_leaves": {"type": "integer", "description": "LightGBM only"},
                "val_ratio": {"type": "number", "description": "Holdout fraction if not fit_full_train"},
                "cv_folds": {"type": "integer", "description": "K-fold CV (0 to skip); default from config"},
                "fit_full_train": {"type": "boolean", "description": "Train on 100% train for submission"},
                "random_state": {"type": "integer"},
            },
            "required": [],
        },
    },
    {
        "name": "make_submission",
        "description": "Generate submission CSV from trained model and test data. Use model_path, test_path (artifacts/test_processed.csv), output_path (submissions/submission.csv).",
        "parameters": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string"},
                "test_path": {"type": "string"},
                "output_path": {"type": "string"},
            },
            "required": [],
        },
    },
]
