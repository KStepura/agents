"""
Coordinator: orchestrates Explorer → Engineer → (optional hparam grid) → Builder workflow.
Pattern: Supervisor.
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
    """Orchestrates the full pipeline and passes artifacts between agents."""

    def __init__(self, explorer, engineer, builder, llm_config: dict, config: dict | None = None):
        self.explorer = explorer
        self.engineer = engineer
        self.builder = builder
        self.llm_config = llm_config
        self.config = config or {}

    def run(self, data_dir: str) -> dict[str, Any]:
        """
        Run full pipeline: EDA → features → (optional hparam grid) → model → submission.
        Returns: submission_path, val_mse, model_summary, agent_metrics, hparam_trials, ...
        """
        eda_artifact = self.explorer.run(data_dir)
        engineer_artifact = self.engineer.run(eda_artifact)

        ev = self.config.get("evaluation", {})
        hs = ev.get("hparam_search") or {}
        from src.evaluation.hparam_search import (
            build_search_grid,
            run_hparam_grid,
            select_best_with_policy,
        )

        mode = (hs.get("mode") or "grid").lower()
        grid = build_search_grid(hs)
        if hs.get("enabled") and mode == "optuna":
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
        elif hs.get("enabled") and grid:
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
        return builder_artifact
