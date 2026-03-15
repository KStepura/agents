"""
Engineer agent: feature engineering and preprocessing from EDA report.
Pattern: Chain-of-Thought + Tool Use.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.agents.tools_for_llm import ENGINEER_TOOLS_SCHEMA, _engineer_tools_executor
from src.llm.client import get_client, run_chat_with_tools


class EngineerAgent:
    """Agent that builds feature pipeline and saves processed data for Builder."""

    def __init__(self, llm_config: dict, allowed_dirs: list[str], system_prompt: str, artifacts_dir: str, data_dir: str = "data"):
        self.llm_config = llm_config
        self.allowed_dirs = [str(Path(d).resolve()) for d in allowed_dirs]
        self.system_prompt = system_prompt
        self.artifacts_dir = str(Path(artifacts_dir).resolve())
        self.data_dir = str(Path(data_dir).resolve())

    def run(self, eda_artifact: dict) -> dict:
        """Produce preprocessing and processed data paths. Returns train_path, test_path, pipeline_path."""
        tools_exec = _engineer_tools_executor(self.allowed_dirs, self.artifacts_dir, self.data_dir)
        client = get_client(self.llm_config)
        report_path = eda_artifact.get("report_path", "artifacts/eda_report.txt")
        user_msg = (
            f"EDA report is at {report_path}. "
            "Fit a preprocessor on data/train.csv and save it to artifacts/preprocessor.joblib. "
            "Then transform train and test (data/train.csv, data/test.csv) and save processed files to artifacts. "
            "Reply with the paths to train_processed.csv and test_processed.csv."
        )
        final_text, _ = run_chat_with_tools(
            client,
            model=self.llm_config.get("model", "qwen/qwen-2.5-72b-instruct"),
            system_prompt=self.system_prompt,
            user_message=user_msg,
            tools_schema=ENGINEER_TOOLS_SCHEMA,
            tool_executor=tools_exec,
            max_steps=10,
            temperature=self.llm_config.get("temperature", 0.2),
            max_tokens=self.llm_config.get("max_tokens", 4096),
        )
        train_path = f"{self.artifacts_dir}/train_processed.csv"
        test_path = f"{self.artifacts_dir}/test_processed.csv"
        return {
            "train_path": train_path,
            "test_path": test_path,
            "pipeline_path": f"{self.artifacts_dir}/preprocessor.joblib",
            "description": final_text[:300] if final_text else "Preprocessing done.",
        }
