"""
Coordinator: оркестрирует процесс Explorer → Engineer → (опционально подбор гиперпараметров) → Builder.
Паттерн: Supervisor.
"""

from __future__ import annotations

from typing import Any


def _merge_llm_usage(*parts: dict[str, int]) -> dict[str, int]:
    acc: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "llm_api_calls": 0,
    }
    for p in parts:
        for k in acc:
            acc[k] += int(p.get(k, 0))
    return acc


class CoordinatorAgent:
    """Управляет выполнением полного пайплайна и передаёт артефакты между агентами."""

    def __init__(self, explorer, engineer, builder, llm_config: dict, config: dict | None = None):
        self.explorer = explorer
        self.engineer = engineer
        self.builder = builder
        self.llm_config = llm_config
        self.config = config or {}

    def run(self, data_dir: str) -> dict[str, Any]:
        """
        Запускает полный пайплайн: EDA → признаки → (опционально подбор гиперпараметров) → модель → submission.
        Возвращает: submission_path, val_mse, model_summary, agent_metrics, hparam_trials, ...
        """
        eda_artifact = self.explorer.run(data_dir)

        ev = self.config.get("evaluation", {})
        hs = ev.get("hparam_search") or {}
        pre = hs.get("preprocessing_search") or {}
        nested_done = False

        if pre.get("enabled"):
            if not hs.get("enabled"):
                raise ValueError("preprocessing_search требует evaluation.hparam_search.enabled: true")
            mode_pre = (hs.get("mode") or "grid").lower()
            if mode_pre != "optuna":
                raise ValueError("preprocessing_search требует evaluation.hparam_search.mode: optuna")
            from src.evaluation.preprocessing_search import run_nested_preprocessing_optuna

            engineer_artifact, val_meta = run_nested_preprocessing_optuna(
                data_dir=data_dir,
                artifacts_dir=str(self.engineer.artifacts_dir),
                allowed_dirs=self.builder.allowed_dirs,
                pipeline_cfg=self.config.get("pipeline", {}),
                evaluation_cfg=ev,
                hparam_cfg=hs,
                variants=pre.get("variants") or [],
            )
            nested_done = True
            print(
                f"  Nested preprocessing+Optuna: best MSE={engineer_artifact.get('hparam_best', {}).get('val_mse')} "
                f"model={engineer_artifact.get('hparam_best', {}).get('model')}"
            )
        else:
            engineer_artifact, val_meta = self._run_engineer_with_validation(eda_artifact)

        if val_meta:
            engineer_artifact["artifact_validation"] = val_meta

        from src.evaluation.hparam_search import (
            build_search_grid,
            run_hparam_grid,
            select_best_with_policy,
        )

        mode = (hs.get("mode") or "grid").lower()
        grid = build_search_grid(hs)
        if hs.get("enabled") and mode == "optuna" and not nested_done:
            from src.evaluation.optuna_hparam import run_optuna_search

            artifacts_dir = str(self.engineer.artifacts_dir)
            ev_search = dict(ev)
            if hs.get("eval_cv_folds") is not None:
                ev_search["_hparam_cv_folds"] = int(hs["eval_cv_folds"])
            best_raw, trials = run_optuna_search(
                train_path=engineer_artifact["train_path"],
                allowed_dirs=self.builder.allowed_dirs,
                artifacts_dir=artifacts_dir,
                evaluation_cfg=ev_search,
                hparam_cfg=hs,
            )
            policy = hs.get("selection_policy") or {}
            best, sel_note = select_best_with_policy(best_raw, trials, policy)
            engineer_artifact["hparam_best"] = best
            engineer_artifact["hparam_trials"] = trials
            engineer_artifact["hparam_selection_note"] = sel_note
            engineer_artifact["hparam_mode"] = "optuna"
            if best:
                msg = (
                    f"  Hparam (Optuna): {len(trials)} trials, best MSE={best.get('val_mse')} "
                    f"model={best.get('model')} params={best.get('params')}"
                )
                if sel_note:
                    msg += f" [{sel_note}]"
                print(msg)
            else:
                print("  Hparam Optuna: no successful trial.")
        elif hs.get("enabled") and grid and not nested_done:
            artifacts_dir = str(self.engineer.artifacts_dir)
            ev_search = dict(ev)
            if hs.get("eval_cv_folds") is not None:
                ev_search["_hparam_cv_folds"] = int(hs["eval_cv_folds"])
            best_raw, trials = run_hparam_grid(
                train_path=engineer_artifact["train_path"],
                allowed_dirs=self.builder.allowed_dirs,
                artifacts_dir=artifacts_dir,
                grid=grid,
                evaluation_cfg=ev_search,
            )
            policy = hs.get("selection_policy") or {}
            best, sel_note = select_best_with_policy(best_raw, trials, policy)
            engineer_artifact["hparam_best"] = best
            engineer_artifact["hparam_trials"] = trials
            engineer_artifact["hparam_selection_note"] = sel_note
            engineer_artifact["hparam_mode"] = "grid"
            if best:
                msg = (
                    f"  Hparam (grid): {len(trials)} trials, best MSE={best.get('val_mse')} "
                    f"model={best.get('model')} params={best.get('params')}"
                )
                if sel_note:
                    msg += f" [{sel_note}]"
                print(msg)
            else:
                print("  Hparam grid: no successful trial (check data / grid).")

        builder_artifact = self.builder.run(engineer_artifact)

        ex_u = eda_artifact.get("llm_usage") or {}
        en_u = engineer_artifact.get("llm_usage") or {}
        bu_u = builder_artifact.get("llm_usage") or {}
        agent_metrics = {
            "explorer": ex_u,
            "engineer": en_u,
            "builder": bu_u,
            "total": _merge_llm_usage(ex_u, en_u, bu_u),
            "model": self.llm_config.get("model", ""),
        }

        builder_artifact["agent_metrics"] = agent_metrics
        if engineer_artifact.get("hparam_trials") is not None:
            builder_artifact["hparam_trials"] = engineer_artifact["hparam_trials"]
        if engineer_artifact.get("hparam_best") is not None:
            builder_artifact["hparam_best"] = engineer_artifact["hparam_best"]
        if engineer_artifact.get("hparam_selection_note") is not None:
            builder_artifact["hparam_selection_note"] = engineer_artifact["hparam_selection_note"]
        if engineer_artifact.get("hparam_mode") is not None:
            builder_artifact["hparam_mode"] = engineer_artifact["hparam_mode"]
        if val_meta:
            builder_artifact["artifact_validation"] = val_meta
        return builder_artifact

    def _run_engineer_with_validation(self, eda_artifact: dict) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Feedback loop: после Engineer проверяем train_processed/test_processed;
        при ошибках — повторный вызов Engineer с текстом замечаний (до max_rounds раундов).
        """
        agents_cfg = self.config.get("agents", {}) or {}
        eng_cfg = agents_cfg.get("engineer", {}) or {}
        av = eng_cfg.get("artifact_validation") or {}
        if not av.get("enabled", True):
            art = self.engineer.run(eda_artifact)
            return art, {}

        from src.evaluation.artifact_validation import validate_processed_datasets
        from src.evaluation.code_validation import (
            validate_preprocessor_joblib,
            validate_python_scripts_in_artifacts,
        )

        max_rounds = max(1, int(av.get("max_rounds", 3)))
        strict = bool(av.get("strict", False))
        min_tr = int(av.get("min_train_rows", 5))
        min_te = int(av.get("min_test_rows", 1))
        check_py = bool(av.get("check_python_scripts", True))
        check_joblib = bool(av.get("check_preprocessor_joblib", True))

        allowed_dirs = self.builder.allowed_dirs
        artifacts_dir = str(self.engineer.artifacts_dir)

        feedback: str | None = None
        merged_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "llm_api_calls": 0,
        }
        engineer_artifact: dict[str, Any] | None = None
        last_ok = False
        last_errors: list[str] = []

        for round_i in range(max_rounds):
            engineer_artifact = self.engineer.run(eda_artifact, feedback_message=feedback)
            u = engineer_artifact.get("llm_usage") or {}
            for k in merged_usage:
                merged_usage[k] += int(u.get(k, 0))

            train_path = engineer_artifact.get("train_path", "")
            test_path = engineer_artifact.get("test_path", "")
            pipeline_path = engineer_artifact.get("pipeline_path", f"{artifacts_dir}/preprocessor.joblib")

            ok_ds, errs_ds = validate_processed_datasets(
                train_path,
                test_path,
                target_col="target",
                min_train_rows=min_tr,
                min_test_rows=min_te,
            )
            merged_errors: list[str] = list(errs_ds)

            if check_py:
                ok_py, e_py = validate_python_scripts_in_artifacts(artifacts_dir, allowed_dirs)
                if not ok_py:
                    merged_errors.extend(e_py)

            if check_joblib:
                ok_j, e_j = validate_preprocessor_joblib(pipeline_path, allowed_dirs)
                if not ok_j:
                    merged_errors.extend(e_j)

            last_errors = merged_errors
            last_ok = len(merged_errors) == 0
            if last_ok:
                if round_i > 0:
                    print(f"  Engineer validation: OK after feedback round {round_i + 1}/{max_rounds}")
                break
            msg = "; ".join(last_errors)
            print(f"  Engineer validation failed (round {round_i + 1}/{max_rounds}): {msg[:400]}")
            feedback = msg
            if round_i == max_rounds - 1 and strict:
                raise RuntimeError(
                    "Engineer preprocessing failed validation after all rounds. "
                    f"Last errors: {last_errors}"
                )
            if round_i == max_rounds - 1 and not strict:
                print("  Warning: proceeding despite validation errors (strict: false).")

        assert engineer_artifact is not None
        engineer_artifact["llm_usage"] = merged_usage
        meta = {
            "rounds_used": round_i + 1,
            "validation_ok": last_ok,
            "last_errors": last_errors if not last_ok else [],
            "code_checks": {"python_scripts": check_py, "preprocessor_joblib": check_joblib},
        }
        return engineer_artifact, meta
