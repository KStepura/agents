"""
Coordinator: orchestrates Explorer → Engineer → Builder workflow.
Pattern: Supervisor.
"""

from __future__ import annotations


class CoordinatorAgent:
    """Orchestrates the full pipeline and passes artifacts between agents."""

    def __init__(self, explorer, engineer, builder, llm_config: dict):
        self.explorer = explorer
        self.engineer = engineer
        self.builder = builder
        self.llm_config = llm_config

    def run(self, data_dir: str) -> dict:
        """
        Run full pipeline: EDA → features → model → submission.
        Returns: {"submission_path": ..., "val_mse": ..., "summary": ...}.
        """
        eda_artifact = self.explorer.run(data_dir)
        engineer_artifact = self.engineer.run(eda_artifact)
        builder_artifact = self.builder.run(engineer_artifact)
        return builder_artifact
