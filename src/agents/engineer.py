"""
Агент Engineer: выполняет feature engineering и препроцессинг на основе EDA-отчёта.
Паттерн: Chain-of-Thought + Tool Use.
"""

from __future__ import annotations

from pathlib import Path

from src.agents.tools_for_llm import ENGINEER_TOOLS_SCHEMA, _engineer_tools_executor
from src.llm.client import get_client, run_chat_with_tools


class EngineerAgent:
    """Агент, который строит пайплайн признаков и сохраняет обработанные данные для Builder."""
    def __init__(
        self,
        llm_config: dict,
        allowed_dirs: list[str],
        system_prompt: str,
        artifacts_dir: str,
        data_dir: str = "data",
        pipeline_config: dict | None = None,
    ):
        self.llm_config = llm_config
        self.allowed_dirs = [str(Path(d).resolve()) for d in allowed_dirs]
        self.system_prompt = system_prompt
        self.artifacts_dir = str(Path(artifacts_dir).resolve())
        self.data_dir = str(Path(data_dir).resolve())
        self.pipeline_config = pipeline_config or {}

    def run(self, eda_artifact: dict, feedback_message: str | None = None) -> dict:
        """
        Формирует пайплайн препроцессинга и пути к обработанным данным.
        Возвращает: train_path, test_path, pipeline_path.
        feedback_message: если задан — предыдущая проверка артефактов не прошла; агент должен исправить пайплайн.
        """
        enc = self.pipeline_config.get("encoding", "te_freq")
        tools_exec = _engineer_tools_executor(
            self.allowed_dirs,
            self.artifacts_dir,
            self.data_dir,
            default_encoding=enc,
            pipeline_defaults=self.pipeline_config,
        )
        client = get_client(self.llm_config)
        report_path = eda_artifact.get("report_path", "artifacts/eda_report.txt")
        user_msg = (
            f"EDA report is at {report_path}. "
            "Fit a preprocessor on data/train.csv and save it to artifacts/preprocessor.joblib. "
            "Then transform train and test (data/train.csv, data/test.csv) and save processed files to artifacts. "
            f"The project uses encoding '{enc}' by default in fit_preprocessor (you may omit encoding). "
            "Reply with the paths to train_processed.csv and test_processed.csv."
        )
        if feedback_message:
            user_msg = (
                "[VALIDATION FAILED — your previous preprocessing did not pass automatic checks. "
                "Fix the issue using ONLY the provided tools (fit_preprocessor, transform_train_test).]\n"
                f"Problems reported:\n{feedback_message}\n\n"
                + user_msg
            )
        final_text, _, usage = run_chat_with_tools(
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
            "llm_usage": usage,
            "agent": "engineer",
        }
