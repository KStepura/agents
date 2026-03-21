"""
Внешний цикл по вариантам препроцессинга (encoding, rare_category_min_count) + Optuna на каждом.
Используется, когда LLM Engineer не нужен или нужно сравнить фичи детерминированно.
"""

from __future__ import annotations

import shutil
import traceback
from pathlib import Path
from typing import Any

from src.evaluation.artifact_validation import validate_processed_datasets
from src.evaluation.optuna_hparam import run_optuna_search
from src.evaluation.hparam_search import select_best_with_policy
from src.tools import feature_tools


def run_nested_preprocessing_optuna(
    *,
    data_dir: str,
    artifacts_dir: str,
    allowed_dirs: list[str],
    pipeline_cfg: dict[str, Any],
    evaluation_cfg: dict[str, Any],
    hparam_cfg: dict[str, Any],
    variants: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Для каждого варианта: fit_preprocessor → transform_train_test → Optuna по MSE.
    Лучший вариант копируется в train_processed.csv / test_processed.csv / preprocessor.joblib.

    Требует evaluation.hparam_search.mode == optuna и непустой variants.
    """
    if not variants:
        raise ValueError("preprocessing_search.enabled requires non-empty hparam_search.preprocessing_search.variants")

    data_dir_p = Path(data_dir).resolve()
    train_raw = str(data_dir_p / "train.csv")
    test_raw = str(data_dir_p / "test.csv")
    artifacts_dir_p = Path(artifacts_dir).resolve()
    artifacts_dir_p.mkdir(parents=True, exist_ok=True)

    best_overall: dict[str, Any] | None = None
    best_mse = float("inf")
    best_variant_idx = -1
    best_sel_note: str | None = None
    best_sub: Path | None = None
    all_trials: list[dict[str, Any]] = []
    ev_search = dict(evaluation_cfg)
    if hparam_cfg.get("eval_cv_folds") is not None:
        ev_search["_hparam_cv_folds"] = int(hparam_cfg["eval_cv_folds"])

    for i, variant in enumerate(variants):
        enc = str(variant.get("encoding", pipeline_cfg.get("encoding", "te_freq"))).lower()
        rc = variant.get("rare_category_min_count", pipeline_cfg.get("rare_category_min_count"))
        sub = artifacts_dir_p / f"preproc_variant_{i}"
        try:
            sub.mkdir(parents=True, exist_ok=True)
            pipe_path = sub / "preprocessor.joblib"
            cfg: dict[str, Any] = {"pipeline_save_path": str(pipe_path), "encoding": enc}
            if rc is not None:
                cfg["rare_category_min_count"] = int(rc)

            feature_tools.fit_preprocessor(
                config=cfg,
                train_path=train_raw,
                target_col="target",
                allowed_dirs=allowed_dirs,
            )
            feature_tools.transform_train_test(
                pipeline_path=str(pipe_path),
                train_path=train_raw,
                test_path=test_raw,
                output_dir=str(sub),
                allowed_dirs=allowed_dirs,
            )
            train_proc = sub / "train_processed.csv"
            best_raw, trials = run_optuna_search(
                train_path=str(train_proc),
                allowed_dirs=allowed_dirs,
                artifacts_dir=str(artifacts_dir_p),
                evaluation_cfg=ev_search,
                hparam_cfg=hparam_cfg,
            )
            for t in trials:
                t["preprocessing_variant_index"] = i
                t["preprocessing_variant"] = dict(variant)
            all_trials.extend(trials)

            policy = hparam_cfg.get("selection_policy") or {}
            best, sel_note = select_best_with_policy(best_raw, trials, policy)
            if best is None:
                print(f"  Preprocessing variant {i} ({enc}): Optuna produced no successful trial.")
                continue
            mse = float(best.get("val_mse") or 1e18)
            print(
                f"  Preprocessing variant {i} ({enc}, rc={rc}): best MSE={mse:.4f} "
                f"model={best.get('model')}"
            )
            if mse < best_mse:
                best_mse = mse
                best_overall = best
                best_variant_idx = i
                best_sel_note = sel_note
                best_sub = sub
        except Exception as e:
            print(
                f"  Preprocessing variant {i} ({enc}, rc={rc}) skipped due to error: {e!r}"
            )
            traceback.print_exc()
            continue

    if best_overall is None or best_variant_idx < 0:
        raise RuntimeError(
            "Nested preprocessing search: no successful Optuna result for any variant."
        )
    assert best_sub is not None

    dst_train = artifacts_dir_p / "train_processed.csv"
    dst_test = artifacts_dir_p / "test_processed.csv"
    dst_pipe = artifacts_dir_p / "preprocessor.joblib"
    shutil.copy2(best_sub / "train_processed.csv", dst_train)
    shutil.copy2(best_sub / "test_processed.csv", dst_test)
    shutil.copy2(best_sub / "preprocessor.joblib", dst_pipe)

    ok, errs = validate_processed_datasets(
        str(dst_train),
        str(dst_test),
        target_col="target",
        min_train_rows=5,
        min_test_rows=1,
    )
    val_meta: dict[str, Any] = {
        "validation_ok": ok,
        "last_errors": errs if not ok else [],
        "preprocessing_nested": True,
        "best_variant_index": best_variant_idx,
        "best_variant": dict(variants[best_variant_idx]),
    }

    engineer_artifact: dict[str, Any] = {
        "train_path": str(dst_train),
        "test_path": str(dst_test),
        "pipeline_path": str(dst_pipe),
        "description": f"Nested preprocessing+Optuna (variant {best_variant_idx})",
        "llm_usage": {},
        "agent": "engineer",
        "hparam_best": best_overall,
        "hparam_trials": all_trials,
        "hparam_selection_note": best_sel_note if best_overall else None,
        "hparam_mode": "optuna_preprocessing_nested",
    }
    return engineer_artifact, val_meta
