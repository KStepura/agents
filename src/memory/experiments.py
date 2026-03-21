"""
Store and load experiment runs: config, MSE, paths to artifacts.
Enables benchmarking and reproducibility.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def save_experiment(
    run_id: str | None,
    config: dict[str, Any],
    val_mse: float | None,
    submission_path: str,
    artifacts_dir: str,
) -> str:
    """Append experiment record as JSONL. Returns path to log file."""
    rid = run_id or uuid.uuid4().hex[:12]
    log_path = Path(artifacts_dir) / "experiments.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": rid,
        "ts": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "val_mse": val_mse,
        "submission_path": submission_path,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return str(log_path)


def list_experiments(artifacts_dir: str) -> list[dict[str, Any]]:
    """List all recorded experiments for comparison."""
    log_path = Path(artifacts_dir) / "experiments.jsonl"
    if not log_path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
