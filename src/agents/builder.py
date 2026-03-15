"""
Builder agent: train regressors, evaluate MSE, produce submission.
Pattern: Planner–Executor–Critic.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.agents.tools_for_llm import BUILDER_TOOLS_SCHEMA, _builder_tools_executor
from src.llm.client import get_client, run_chat_with_tools


class BuilderAgent:
    """Agent that trains models, evaluates MSE, and generates submission.csv."""

    def __init__(
        self,
        llm_config: dict,
        allowed_dirs: list[str],
        system_prompt: str,
        artifacts_dir: str,
        submissions_dir: str,
        critic_max_iterations: int = 3,
    ):
        self.llm_config = llm_config
        self.allowed_dirs = [str(Path(d).resolve()) for d in allowed_dirs]
        self.system_prompt = system_prompt
        self.artifacts_dir = str(Path(artifacts_dir).resolve())
        self.submissions_dir = str(Path(submissions_dir).resolve())
        self.critic_max_iterations = critic_max_iterations

    def run(self, engineer_artifact: dict) -> dict:
        """Train model, produce submission. Returns submission_path, val_mse, model_summary."""
        train_path = engineer_artifact.get("train_path", f"{self.artifacts_dir}/train_processed.csv")
        test_path = engineer_artifact.get("test_path", f"{self.artifacts_dir}/test_processed.csv")
        tools_exec = _builder_tools_executor(
            self.allowed_dirs, self.artifacts_dir, self.submissions_dir
        )
        client = get_client(self.llm_config)
        user_msg = (
            f"Processed train: {train_path}, test: {test_path}. "
            "Train a regressor (recommended: lightgbm with n_estimators 300, max_depth 8). "
            "Then create submission at submissions/submission.csv. "
            "Reply with the validation MSE and the path to the submission file."
        )
        final_text, messages = run_chat_with_tools(
            client,
            model=self.llm_config.get("model", "qwen/qwen-2.5-72b-instruct"),
            system_prompt=self.system_prompt,
            user_message=user_msg,
            tools_schema=BUILDER_TOOLS_SCHEMA,
            tool_executor=tools_exec,
            max_steps=10,
            temperature=self.llm_config.get("temperature", 0.2),
            max_tokens=self.llm_config.get("max_tokens", 4096),
        )
        val_mse = _extract_mse(final_text) or _extract_val_mse_from_messages(messages)
        sub_path = f"{self.submissions_dir}/submission.csv"
        return {
            "submission_path": sub_path,
            "val_mse": val_mse,
            "model_summary": final_text[:400] if final_text else "Model trained, submission written.",
        }


def _extract_mse(text: str) -> float | None:
    m = re.search(r"(\d+\.?\d*)\s*(?:MSE|mse|val_mse)", text, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"val_mse[\"']?\s*:\s*(\d+\.?\d*)", text, re.I)
    if m:
        return float(m.group(1))
    return None


def _extract_val_mse_from_messages(messages: list) -> float | None:
    """Get val_mse from last tool response that looks like train_regressor result."""
    for m in reversed(messages):
        if m.get("role") == "tool" and "content" in m:
            try:
                data = json.loads(m["content"])
                if "val_mse" in data:
                    return float(data["val_mse"])
            except (json.JSONDecodeError, TypeError):
                continue
    return None
