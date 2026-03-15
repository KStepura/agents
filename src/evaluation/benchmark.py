"""
Benchmark: run multiple configs and compare MSE / submission quality.
"""

from __future__ import annotations

from typing import Any


def run_benchmark(
    configs: list[dict[str, Any]],
    data_dir: str,
    artifacts_dir: str,
) -> list[dict[str, Any]]:
    """Run pipeline for each config, record MSE and paths. Return list of results."""
    raise NotImplementedError("Implement: loop over configs, run coordinator, save experiments")
