"""
Benchmark: run multiple configs and compare MSE / submission quality.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.tools import feature_tools, model_tools


def run_benchmark(
    configs: list[dict[str, Any]],
    data_dir: str,
    artifacts_dir: str,
    submissions_dir: str,
) -> list[dict[str, Any]]:
    """
    Run preprocessing + training for each config in an isolated subdirectory.
    Each config may contain: model, n_estimators, max_depth, learning_rate, alpha, val_ratio.
    """
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
        row = {
            "config_index": i,
            "config": cfg,
            "val_mse": result["val_mse"],
            "model_path": result["model_path"],
            "submission_path": str(sub_path),
        }
        results.append(row)

    return results
