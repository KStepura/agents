#!/usr/bin/env python3
"""
Запуск нескольких конфигураций моделей без LLM (см. src/evaluation/benchmark.py).

  python scripts/run_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark multiple model configs")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--submissions-dir", default="submissions")
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()

    import yaml

    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)
    paths = cfg.get("paths", {})

    data_dir = str(ROOT / (args.data_dir or paths.get("data_dir", "data")))
    artifacts_dir = str(ROOT / (args.artifacts_dir or paths.get("artifacts_dir", "artifacts")))
    submissions_dir = str(ROOT / (args.submissions_dir or paths.get("submissions_dir", "submissions")))

    # Пример набора конфигураций для сравнения Val MSE
    configs = [
        {"model": "ridge"},
        {"model": "random_forest", "n_estimators": 200},
        {"model": "lightgbm", "n_estimators": 300, "max_depth": 8},
    ]

    from src.evaluation.benchmark import run_benchmark

    results = run_benchmark(configs, data_dir, artifacts_dir, submissions_dir)
    print(json.dumps(results, indent=2, default=str))
    best = min(results, key=lambda r: r["val_mse"])
    print("\nBest val_mse:", best["val_mse"], "config:", best["config"])


if __name__ == "__main__":
    main()
