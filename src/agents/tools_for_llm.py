"""
Tool schemas (OpenAI function format) and executors for agents.
All executors receive (name, args) and return a string for the LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# EDA: wrappers that return strings for LLM
def _explorer_tools_executor(allowed_dirs: list[str], data_dir: str):
    from src.tools import eda_tools
    def executor(name: str, args: dict) -> str:
        path = args.get("path", args.get("file_path", ""))
        if path and not path.startswith("/") and path not in ("data", "artifacts"):
            full = Path(data_dir) / path
            if full.exists() or path in ("train.csv", "test.csv"):
                path = str(full)
        if name == "load_and_summarize":
            df = eda_tools.load_dataset(path, allowed_dirs)
            dtypes = eda_tools.get_dtypes(df)
            desc = eda_tools.describe(df)
            return f"Shape: {df.shape}. Dtypes: {json.dumps(dtypes)}. Describe (numeric): {json.dumps(desc, default=str)[:2000]}."
        if name == "get_missing_and_correlation":
            df = eda_tools.load_dataset(path, allowed_dirs)
            miss = eda_tools.missing_report(df)
            corr = eda_tools.correlation_with_target(df, "target") if "target" in df.columns else {}
            return f"Missing: {json.dumps(miss)}. Correlation with target: {json.dumps(corr)}."
        if name == "save_eda_report":
            content = args.get("content", args.get("report", ""))
            out = args.get("output_path", "artifacts/eda_report.txt")
            out = str(Path(out).resolve())
            return eda_tools.save_eda_report(content, out, allowed_dirs)
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


def _engineer_tools_executor(allowed_dirs: list[str], artifacts_dir: str, data_dir: str = "data"):
    from src.tools import feature_tools
    data_dir = str(Path(data_dir).resolve())
    def executor(name: str, args: dict) -> str:
        if name == "fit_preprocessor":
            train_path = args.get("train_path") or f"{data_dir}/train.csv"
            pipeline_path = args.get("pipeline_save_path", f"{artifacts_dir}/preprocessor.joblib")
            p = feature_tools.fit_preprocessor(
                config={"pipeline_save_path": pipeline_path},
                train_path=train_path,
                target_col="target",
                allowed_dirs=allowed_dirs,
            )
            return f"Preprocessor saved to {p}"
        if name == "transform_train_test":
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


def _builder_tools_executor(allowed_dirs: list[str], artifacts_dir: str, submissions_dir: str):
    from src.tools import model_tools
    def executor(name: str, args: dict) -> str:
        if name == "train_regressor":
            train_path = args.get("train_path", f"{artifacts_dir}/train_processed.csv")
            model_name = args.get("model", "lightgbm")
            n_est = args.get("n_estimators", 300)
            max_d = args.get("max_depth", 8)
            lr = args.get("learning_rate")
            params = {"n_estimators": n_est, "max_depth": max_d}
            if lr is not None:
                params["learning_rate"] = lr
            res = model_tools.train_regressor(
                name=model_name,
                params=params,
                train_path=train_path,
                target_col="target",
                val_path=None,
                model_save_path=f"{artifacts_dir}/model.joblib",
                allowed_dirs=allowed_dirs,
                val_ratio=0.2,
                random_state=42,
            )
            return json.dumps(res)
        if name == "make_submission":
            model_path = args.get("model_path", f"{artifacts_dir}/model.joblib")
            test_path = args.get("test_path", f"{artifacts_dir}/test_processed.csv")
            out_path = args.get("output_path", f"{submissions_dir}/submission.csv")
            p = model_tools.make_submission(
                model_path=model_path,
                test_path=test_path,
                output_path=out_path,
                index_col=None,
                allowed_dirs=allowed_dirs,
            )
            return f"Submission saved to {p}"
        return "Unknown tool"
    return executor


BUILDER_TOOLS_SCHEMA = [
    {
        "name": "train_regressor",
        "description": "Train a regressor. Use train_path (e.g. artifacts/train_processed.csv), model one of: ridge, random_forest, xgboost, lightgbm. Recommended: lightgbm with n_estimators 300, max_depth 8.",
        "parameters": {
            "type": "object",
            "properties": {
                "train_path": {"type": "string"},
                "model": {"type": "string", "enum": ["ridge", "random_forest", "xgboost", "lightgbm"]},
                "n_estimators": {"type": "integer"},
                "max_depth": {"type": "integer"},
                "learning_rate": {"type": "number"},
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
