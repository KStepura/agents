#!/usr/bin/env python3
"""
Ручной запуск пайплайна без агентов: данные → препроцессинг → модель → submission.

Запуск из корня проекта:
  python scripts/run_pipeline_manual.py
  python scripts/run_pipeline_manual.py --data-dir data --model ridge
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Добавляем корень проекта в путь
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pipeline manually (no LLM)")
    parser.add_argument("--data-dir", default="data", help="Directory with train.csv, test.csv")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Output for pipeline, models")
    parser.add_argument("--submissions-dir", default="submissions", help="Output for submission CSV")
    parser.add_argument("--model", default="ridge", choices=["ridge", "random_forest", "xgboost", "lightgbm"])
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--config", default=None, help="Optional YAML config (uses paths from it)")
    # Гиперпараметры для деревьев / бустинга (игнорируются для ridge)
    parser.add_argument("--n-estimators", type=int, default=None, help="e.g. 200, 300 (RF/XGB/LGB)")
    parser.add_argument("--max-depth", type=int, default=None, help="e.g. 8, 12 (RF/XGB/LGB)")
    parser.add_argument("--learning-rate", type=float, default=None, help="e.g. 0.05 (XGB/LGB)")
    parser.add_argument("--alpha", type=float, default=None, help="Ridge regularization (ridge only)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    artifacts_dir = Path(args.artifacts_dir)
    submissions_dir = Path(args.submissions_dir)

    if not (data_dir / "train.csv").exists():
        print(f"Error: train.csv not found in {data_dir}")
        sys.exit(1)
    if not (data_dir / "test.csv").exists():
        print(f"Error: test.csv not found in {data_dir}")
        sys.exit(1)

    # Разрешённые директории (относительно текущей рабочей директории)
    allowed = [str(Path(p).resolve()) for p in [args.data_dir, args.artifacts_dir, args.submissions_dir]]

    from src.tools import feature_tools
    from src.tools import model_tools

    target_col = "target"
    train_csv = str(data_dir / "train.csv")
    test_csv = str(data_dir / "test.csv")
    pipeline_path = str(artifacts_dir / "preprocessor.joblib")

    print("1. Fitting preprocessor...")
    feature_tools.fit_preprocessor(
        config={"pipeline_save_path": pipeline_path},
        train_path=train_csv,
        target_col=target_col,
        allowed_dirs=allowed,
    )
    print(f"   Saved: {pipeline_path}")

    print("2. Transforming train and test...")
    out = feature_tools.transform_train_test(
        pipeline_path=pipeline_path,
        train_path=train_csv,
        test_path=test_csv,
        output_dir=str(artifacts_dir),
        allowed_dirs=allowed,
    )
    train_processed = out["train_path"]
    test_processed = out["test_path"]
    print(f"   Train: {train_processed}, Test: {test_processed}")

    params = {}
    if args.n_estimators is not None:
        params["n_estimators"] = args.n_estimators
    if args.max_depth is not None:
        params["max_depth"] = args.max_depth
    if args.learning_rate is not None:
        params["learning_rate"] = args.learning_rate
    if args.alpha is not None and args.model == "ridge":
        params["alpha"] = args.alpha
    if params:
        print(f"   Params: {params}")

    print("3. Training regressor...")
    result = model_tools.train_regressor(
        name=args.model,
        params=params,
        train_path=train_processed,
        target_col=target_col,
        val_path=None,
        model_save_path=str(artifacts_dir / "model.joblib"),
        allowed_dirs=allowed,
        val_ratio=args.val_ratio,
        random_state=42,
    )
    print(f"   Val MSE: {result['val_mse']:.4f}")
    print(f"   Model: {result['model_path']}")

    print("4. Writing submission...")
    sub_path = submissions_dir / "submission.csv"
    path = model_tools.make_submission(
        model_path=result["model_path"],
        test_path=test_processed,
        output_path=str(sub_path),
        index_col=None,
        allowed_dirs=allowed,
    )
    print(f"   Submission: {path}")

    print("Done.")
    print(f"  Val MSE = {result['val_mse']:.4f}")
    print(f"  Submission file: {path}")

    baseline = 10986.9346
    if result["val_mse"] < baseline:
        print(f"  Baseline beaten: Val MSE {result['val_mse']:.4f} < {baseline}")
    else:
        print(f"  Baseline to beat: {baseline} (current Val MSE {result['val_mse']:.4f})")


if __name__ == "__main__":
    main()
