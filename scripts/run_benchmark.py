#!/usr/bin/env python3
"""
Вспомогательный скрипт (не входит в python run.py): несколько конфигураций моделей
на одном препроцессинге, без LLM.

  python scripts/run_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_benchmark(
    configs: list[dict[str, Any]],
    data_dir: str,
    artifacts_dir: str,
    submissions_dir: str,
) -> list[dict[str, Any]]:
    """Для каждого элемента configs: fit preprocessor → transform → train → submission."""
    from src.tools import feature_tools, model_tools

    data_dir = Path(data_dir).resolve()
    base_art = Path(artifacts_dir).resolve()
    sub_root = Path(submissions_dir).resolve()
    base_art.mkdir(parents=True, exist_ok=True)
    sub_root.mkdir(parents=True, exist_ok=True)

    train_csv = str(data_dir / "train.csv")
    test_csv = str(data_dir / "test.csv")
    if not Path(train_csv).exists() or not Path(test_csv).exists():
        raise FileNotFoundError(f"Need train.csv and test.csv in {data_dir}")

    results: list[dict[str, Any]] = []

    for i, cfg in enumerate(configs):
        run_dir = base_art / f"benchmark_{i}"
        run_dir.mkdir(parents=True, exist_ok=True)
        allowed = [str(data_dir), str(run_dir), str(sub_root)]

        model_name = cfg.get("model", "lightgbm")
        val_ratio = float(cfg.get("val_ratio", 0.2))
        params = {k: cfg[k] for k in ("n_estimators", "max_depth", "learning_rate", "alpha") if k in cfg}

        pipeline_path = str(run_dir / "preprocessor.joblib")
        feature_tools.fit_preprocessor(
            config={
                "pipeline_save_path": pipeline_path,
                "encoding": cfg.get("encoding", "te_freq"),
            },
            train_path=train_csv,
            target_col="target",
            allowed_dirs=allowed,
        )
        out = feature_tools.transform_train_test(
            pipeline_path=pipeline_path,
            train_path=train_csv,
            test_path=test_csv,
            output_dir=str(run_dir),
            allowed_dirs=allowed,
        )
        train_processed = out["train_path"]
        test_processed = out["test_path"]

        result = model_tools.train_regressor(
            name=model_name,
            params=params,
            train_path=train_processed,
            target_col="target",
            val_path=None,
            model_save_path=str(run_dir / "model.joblib"),
            allowed_dirs=allowed,
            val_ratio=val_ratio,
            random_state=int(cfg.get("random_state", 42)),
            cv_folds=cfg.get("cv_folds"),
            fit_full_train=bool(cfg.get("fit_full_train", False)),
        )
        sub_path = sub_root / f"submission_benchmark_{i}.csv"
        model_tools.make_submission(
            model_path=result["model_path"],
            test_path=test_processed,
            output_path=str(sub_path),
            index_col=None,
            allowed_dirs=allowed,
        )
        results.append(
            {
                "config_index": i,
                "config": cfg,
                "val_mse": result["val_mse"],
                "model_path": result["model_path"],
                "submission_path": str(sub_path),
            }
        )

    return results


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

    configs = [
        {"model": "ridge"},
        {"model": "random_forest", "n_estimators": 200},
        {"model": "lightgbm", "n_estimators": 300, "max_depth": 8},
    ]

    results = run_benchmark(configs, data_dir, artifacts_dir, submissions_dir)
    print(json.dumps(results, indent=2, default=str))
    best = min(results, key=lambda r: r["val_mse"])
    print("\nBest val_mse:", best["val_mse"], "config:", best["config"])


if __name__ == "__main__":
    main()
