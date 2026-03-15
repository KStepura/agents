"""
Store and load experiment runs: config, MSE, paths to artifacts.
Enables benchmarking and reproducibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_experiment(
    run_id: str,
    config: dict[str, Any],
    val_mse: float,
    submission_path: str,
    artifacts_dir: str,
) -> str:
    """Append or write experiment record (e.g. JSONL or SQLite). Returns path to log."""
    raise NotImplementedError("Implement: append run to experiments.jsonl or DB")


def list_experiments(artifacts_dir: str) -> list[dict[str, Any]]:
    """List all recorded experiments for comparison."""
    raise NotImplementedError("Implement: read experiments and return list of dicts")
