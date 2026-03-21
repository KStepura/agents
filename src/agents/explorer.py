"""
Explorer agent: EDA and structured report for the Engineer.
Pattern: ReAct (Reasoning + Act) with Tool Use.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.agents.tools_for_llm import EXPLORER_TOOLS_SCHEMA, _explorer_tools_executor
from src.llm.client import get_client, run_chat_with_tools


class ExplorerAgent:
    """Agent that performs EDA and produces an artifact for the Engineer."""

    def __init__(
        self,
        llm_config: dict,
        allowed_dirs: list[str],
        system_prompt: str,
        rag_config: dict | None = None,
        project_root: Path | None = None,
    ):
        self.llm_config = llm_config
        self.allowed_dirs = [str(Path(d).resolve()) for d in allowed_dirs]
        self.system_prompt = system_prompt
        self.rag_config = rag_config or {}
        self.project_root = project_root or Path.cwd()

    def run(self, data_dir: str) -> dict:
        """Run EDA on train/test in data_dir. Returns {"report_path": ..., "summary": ...}."""
        data_dir = str(Path(data_dir).resolve())
        rag_hint = self._rag_context()

        tools_exec = _explorer_tools_executor(self.allowed_dirs, data_dir)
        client = get_client(self.llm_config)
        user_msg = (
            f"Data directory is: {data_dir}. "
            "Load train.csv and test.csv, analyze them (shape, dtypes, missing values, correlation with target). "
            "Then save a concise EDA report to artifacts/eda_report.txt. "
            "Reply with the path where you saved the report and a one-sentence summary."
            f"{rag_hint}"
        )
        final_text, _, usage = run_chat_with_tools(
            client,
            model=self.llm_config.get("model", "qwen/qwen-2.5-72b-instruct"),
            system_prompt=self.system_prompt,
            user_message=user_msg,
            tools_schema=EXPLORER_TOOLS_SCHEMA,
            tool_executor=tools_exec,
            max_steps=12,
            temperature=self.llm_config.get("temperature", 0.2),
            max_tokens=self.llm_config.get("max_tokens", 4096),
        )
        report_path = _extract_path(final_text) or "artifacts/eda_report.txt"
        return {
            "report_path": report_path,
            "summary": final_text[:500] if final_text else "EDA done.",
            "llm_usage": usage,
            "agent": "explorer",
        }

    def _rag_context(self) -> str:
        if not self.rag_config.get("enabled", True):
            return ""
        rel = self.rag_config.get("persist_path", "artifacts/chroma_rag")
        persist = Path(rel)
        if not persist.is_absolute():
            persist = (self.project_root / rel).resolve()
        if not persist.exists() or not any(persist.iterdir()):
            return ""
        try:
            from src.rag.retriever import retrieve

            q = (
                "Exploratory data analysis tabular regression missing values correlation target "
                "Kaggle MSE feature engineering"
            )
            chunks = retrieve(
                q,
                str(persist),
                top_k=int(self.rag_config.get("top_k", 5)),
                embedding_model=self.rag_config.get(
                    "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
                ),
            )
            if not chunks:
                return ""
            return "\n\nRelevant knowledge (RAG):\n" + "\n---\n".join(chunks)
        except Exception as e:
            return f"\n\n(RAG unavailable: {e})"


def _extract_path(text: str) -> str | None:
    m = re.search(r"artifacts/[^\s\)\]\"]+", text)
    return m.group(0) if m else None
