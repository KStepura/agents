"""
Builder agent: train regressors, evaluate MSE, produce submission.
Pattern: Planner–Executor–Critic (retry when MSE above threshold or for another attempt).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.agents.tools_for_llm import BUILDER_TOOLS_SCHEMA, _builder_tools_executor
from src.llm.client import get_client, run_chat_with_tools
from src.tools import model_tools


class BuilderAgent:
    """Agent that trains models, evaluates MSE, and generates submission.csv."""

    def __init__(
        self,
        llm_config: dict,
        allowed_dirs: list[str],
        system_prompt: str,
        artifacts_dir: str,
        submissions_dir: str,
        critic_max_iterations: int = 3,
        evaluation_config: dict[str, Any] | None = None,
        builder_config: dict[str, Any] | None = None,
        robustness_config: dict[str, Any] | None = None,
    ):
        self.llm_config = llm_config
        self.allowed_dirs = [str(Path(d).resolve()) for d in allowed_dirs]
        self.system_prompt = system_prompt
        self.artifacts_dir = str(Path(artifacts_dir).resolve())
        self.submissions_dir = str(Path(submissions_dir).resolve())
        self.critic_max_iterations = max(1, int(critic_max_iterations))
        self.evaluation_config = evaluation_config or {}
        self.builder_config = builder_config or {}
        self.robustness_config = robustness_config or {}

    def run(self, engineer_artifact: dict) -> dict:
        """Train model, produce submission. Returns submission_path, val_mse, model_summary, llm_usage."""
        train_path = engineer_artifact.get("train_path", f"{self.artifacts_dir}/train_processed.csv")
        test_path = engineer_artifact.get("test_path", f"{self.artifacts_dir}/test_processed.csv")
        ev = self.evaluation_config
        hs = ev.get("hparam_search") or {}
        use_best_only = bool(hs.get("use_best_only")) and engineer_artifact.get("hparam_best")

        if use_best_only:
            return self._run_deterministic_best(train_path, test_path, engineer_artifact)

        return self._run_llm_with_critic(train_path, test_path, engineer_artifact)

    def _robustness_kwargs(self, train_path: str) -> dict[str, Any]:
        r = dict(self.robustness_config)
        if not r.get("clip_predictions") and not r.get("clip_features_to_train_range"):
            return {}
        out: dict[str, Any] = {**r, "train_path": train_path}
        return out

    def _run_deterministic_best(
        self, train_path: str, test_path: str, engineer_artifact: dict
    ) -> dict:
        best = engineer_artifact["hparam_best"]
        ev = self.evaluation_config
        cv = ev.get("cv_folds", 0)
        cv_folds = None if cv is None or int(cv) == 0 else int(cv)
        fft = bool(ev.get("fit_full_train", False))
        rs = int(ev.get("random_state", 42))
        vr = float(ev.get("val_ratio", 0.2))

        model_name = best["model"]
        params = dict(best.get("params") or {})
        ens = ev.get("ensemble") or {}
        rk = self._robustness_kwargs(train_path)
        blend = bool(ens.get("enabled")) and bool(ens.get("kfold_blend_test"))
        n_blend = max(2, int(ens.get("n_folds") or cv_folds or 5))

        stack_ev = ev.get("stacking") or {}
        if bool(stack_ev.get("enabled")):
            from src.tools import advanced_ensemble

            base_models = stack_ev.get("base_models") or []
            if not base_models:
                raise ValueError(
                    "evaluation.stacking.enabled requires non-empty evaluation.stacking.base_models"
                )
            meta_cfg = stack_ev.get("meta") or {"model": "ridge"}
            stack_cv = stack_ev.get("cv_folds")
            if stack_cv is None:
                stack_cv = cv_folds if cv_folds is not None else 5
            else:
                stack_cv = int(stack_cv)
            res = advanced_ensemble.train_oof_stacking(
                train_path=train_path,
                test_path=test_path,
                base_models=list(base_models),
                meta_config=dict(meta_cfg),
                submission_path=f"{self.submissions_dir}/submission.csv",
                bundle_path=f"{self.artifacts_dir}/stacking_bundle.joblib",
                allowed_dirs=self.allowed_dirs,
                cv_folds=max(2, stack_cv),
                random_state=rs,
                robustness=rk or None,
            )
        elif bool((ev.get("pseudo_labels") or {}).get("enabled")):
            from src.tools import advanced_ensemble

            pseudo_ev = ev.get("pseudo_labels") or {}
            rounds = max(1, int(pseudo_ev.get("rounds", 2)))
            p_cv = pseudo_ev.get("cv_folds")
            if p_cv is None:
                p_cv = cv_folds
            elif int(p_cv) < 2:
                p_cv = None
            else:
                p_cv = int(p_cv)
            res = advanced_ensemble.train_with_pseudo_labels(
                name=model_name,
                params=params,
                train_path=train_path,
                test_path=test_path,
                model_save_path=f"{self.artifacts_dir}/model.joblib",
                submission_path=f"{self.submissions_dir}/submission.csv",
                allowed_dirs=self.allowed_dirs,
                rounds=rounds,
                random_state=rs,
                cv_folds=p_cv,
                robustness=rk or None,
            )
        elif blend:
            res = model_tools.train_regressor_kfold_blend(
                name=model_name,
                params=params,
                train_path=train_path,
                test_path=test_path,
                model_save_path=f"{self.artifacts_dir}/model.joblib",
                submission_path=f"{self.submissions_dir}/submission.csv",
                allowed_dirs=self.allowed_dirs,
                cv_folds=n_blend,
                random_state=rs,
                robustness=rk or None,
            )
        else:
            res = model_tools.train_regressor(
                name=model_name,
                params=params,
                train_path=train_path,
                target_col="target",
                val_path=None,
                model_save_path=f"{self.artifacts_dir}/model.joblib",
                allowed_dirs=self.allowed_dirs,
                val_ratio=vr,
                random_state=rs,
                cv_folds=cv_folds,
                fit_full_train=fft,
            )
            model_tools.make_submission(
                model_path=res["model_path"],
                test_path=test_path,
                output_path=f"{self.submissions_dir}/submission.csv",
                index_col=None,
                allowed_dirs=self.allowed_dirs,
                robustness=rk or None,
            )
        val_mse = res.get("val_mse")
        if val_mse is None:
            val_mse = res.get("cv_mse_mean")
        if res.get("stacking_oof"):
            mode_tag = "oof_stacking"
            summary = (
                f"Deterministic Builder ({mode_tag}): n_base={res.get('n_base_models')}, "
                f"meta={res.get('meta_model')}, val_mse/cv={val_mse}"
            )
        elif res.get("pseudo_labels"):
            mode_tag = "pseudo_labels"
            summary = (
                f"Deterministic Builder ({mode_tag}): model={model_name}, params={params}, "
                f"rounds={res.get('pseudo_labels_rounds')}, val_mse/cv={val_mse}"
            )
        elif res.get("oof_kfold_blend"):
            mode_tag = "kfold_blend"
            summary = (
                f"Deterministic Builder ({mode_tag}): model={model_name}, params={params}, "
                f"val_mse/cv={val_mse}"
            )
        else:
            mode_tag = "single_model"
            summary = (
                f"Deterministic Builder ({mode_tag}): model={model_name}, params={params}, "
                f"val_mse/cv={val_mse}"
            )
        return {
            "submission_path": f"{self.submissions_dir}/submission.csv",
            "val_mse": float(val_mse) if val_mse is not None else None,
            "model_summary": summary,
            "llm_usage": _empty_usage(),
            "agent": "builder",
            "builder_mode": f"deterministic_hparam_best_{mode_tag}",
        }

    def _run_llm_with_critic(
        self, train_path: str, test_path: str, engineer_artifact: dict
    ) -> dict:
        ev = self.evaluation_config
        cv = ev.get("cv_folds", 0)
        cv_folds = None if cv is None or int(cv) == 0 else int(cv)
        tools_exec = _builder_tools_executor(
            self.allowed_dirs,
            self.artifacts_dir,
            self.submissions_dir,
            val_ratio=float(ev.get("val_ratio", 0.2)),
            cv_folds=cv_folds,
            fit_full_train=bool(ev.get("fit_full_train", False)),
            random_state=int(ev.get("random_state", 42)),
            robustness=self._robustness_kwargs(train_path) or None,
        )
        client = get_client(self.llm_config)
        fft = bool(self.evaluation_config.get("fit_full_train", False))
        cv_k = self.evaluation_config.get("cv_folds", 0)
        mse_threshold = self.builder_config.get("mse_threshold")
        grid_hint = ""
        hb = engineer_artifact.get("hparam_best")
        if hb and not (self.evaluation_config.get("hparam_search") or {}).get("use_best_only"):
            grid_hint = (
                f"\nOptional grid-search result: model={hb['model']}, params={hb.get('params')}, "
                f"estimated MSE≈{hb.get('val_mse')}. You may start from these or improve.\n"
            )
        base_msg = (
            f"Processed train: {train_path}, test: {test_path}. "
            "Train a regressor (recommended: lightgbm with n_estimators 300, max_depth 8). "
            f"Project evaluation: fit_full_train={fft}, cv_folds={cv_k} (defaults apply to train_regressor if you omit them). "
            "Then create submission at submissions/submission.csv. "
            "Reply with the validation or CV MSE and the path to the submission file."
            + grid_hint
        )
        feedback = ""
        usage_acc: dict[str, int] = _empty_usage()
        final_text = ""
        messages: list[dict] = []
        always_iterate = bool(self.builder_config.get("critic_iterate_without_threshold", False))
        max_rounds = (
            self.critic_max_iterations
            if (mse_threshold is not None or always_iterate)
            else 1
        )

        for attempt in range(max_rounds):
            user_msg = base_msg + feedback
            final_text, messages, usage = run_chat_with_tools(
                client,
                model=self.llm_config.get("model", "qwen/qwen-2.5-72b-instruct"),
                system_prompt=self.system_prompt,
                user_message=user_msg,
                tools_schema=BUILDER_TOOLS_SCHEMA,
                tool_executor=tools_exec,
                max_steps=10,
                temperature=self.llm_config.get("temperature", 0.2),
                max_tokens=self.llm_config.get("max_tokens", 4096),
            )
            for k in ("prompt_tokens", "completion_tokens", "total_tokens", "llm_api_calls"):
                usage_acc[k] = usage_acc.get(k, 0) + usage.get(k, 0)

            val_mse = _extract_mse(final_text) or _extract_val_mse_from_messages(messages)
            if mse_threshold is not None and val_mse is not None and val_mse <= float(mse_threshold):
                break
            if attempt < max_rounds - 1:
                prev = val_mse if val_mse is not None else "unknown"
                feedback = (
                    f"\n\n[Critic] Previous attempt reported val/cv MSE ≈ {prev}. "
                    "Try a different model family or hyperparameters (e.g. lightgbm depth, n_estimators, learning_rate) "
                    "to reduce MSE, then call train_regressor and make_submission again."
                )

        return {
            "submission_path": f"{self.submissions_dir}/submission.csv",
            "val_mse": _extract_mse(final_text) or _extract_val_mse_from_messages(messages),
            "model_summary": final_text[:400] if final_text else "Model trained, submission written.",
            "llm_usage": usage_acc,
            "agent": "builder",
            "builder_mode": "llm_critic",
            "critic_rounds": max_rounds,
        }


def _empty_usage() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "llm_api_calls": 0,
    }


def _extract_mse(text: str) -> float | None:
    m = re.search(r"(\d+\.?\d*)\s*(?:MSE|mse|val_mse)", text, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"val_mse[\"']?\s*:\s*(\d+\.?\d*)", text, re.I)
    if m:
        return float(m.group(1))
    return None


def _extract_val_mse_from_messages(messages: list) -> float | None:
    """Get val_mse or cv_mse_mean from train_regressor tool JSON."""
    for m in reversed(messages):
        if m.get("role") == "tool" and "content" in m:
            try:
                data = json.loads(m["content"])
                if data.get("val_mse") is not None:
                    return float(data["val_mse"])
                if data.get("cv_mse_mean") is not None:
                    return float(data["cv_mse_mean"])
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
    return None
