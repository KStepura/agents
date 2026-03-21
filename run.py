#!/usr/bin/env python3
"""
Точка входа: запуск мультиагентного pipeline для Kaggle-регрессии (MSE).

Использование:
  python run.py
  python run.py --data-dir data --config config/settings.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_system_prompt(prompt_path: str, root: Path) -> str:
    """Load system prompt from file (relative to project root)."""
    path = root / prompt_path if not Path(prompt_path).is_absolute() else Path(prompt_path)
    if path.exists():
        return path.read_text().strip()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-agent Kaggle regression pipeline")
    parser.add_argument("--data-dir", default=None, help="Directory with train.csv, test.csv")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to config YAML")
    parser.add_argument(
        "--experiment-label",
        default=None,
        help="Optional tag stored in artifacts/experiments.jsonl (e.g. arch_rag_on)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    config = load_config(str(cfg_path))
    paths = config.get("paths", {})
    data_dir = str(root / (args.data_dir or paths.get("data_dir", "data")))
    artifacts_dir = str(root / paths.get("artifacts_dir", "artifacts"))
    submissions_dir = str(root / paths.get("submissions_dir", "submissions"))

    if not (Path(data_dir) / "train.csv").exists():
        raise FileNotFoundError(
            f"train.csv not found in {data_dir}. "
            "Place train.csv, test.csv in data/."
        )

    # Load .env for OPENROUTER_API_KEY
    try:
        from dotenv import load_dotenv
        load_dotenv(root / ".env")
    except ImportError:
        pass

    from src.monitoring.logger import configure_logging

    configure_logging()

    from src.rag.bootstrap import ensure_rag_index_if_needed

    ensure_rag_index_if_needed(rag_config=config.get("rag", {}), paths=paths, project_root=root)

    llm_config = config.get("llm", {})
    security = config.get("security", {})
    allowed_dirs = security.get("allowed_data_dirs", ["data", "artifacts", "submissions"])
    allowed_dirs = [str(root / d) for d in allowed_dirs]
    agents_cfg = config.get("agents", {})
    rag_config = config.get("rag", {})

    explorer_prompt = load_system_prompt(agents_cfg.get("explorer", {}).get("system_prompt_file", "config/prompts/explorer_system.txt"), root)
    engineer_prompt = load_system_prompt(agents_cfg.get("engineer", {}).get("system_prompt_file", "config/prompts/engineer_system.txt"), root)
    builder_prompt = load_system_prompt(agents_cfg.get("builder", {}).get("system_prompt_file", "config/prompts/builder_system.txt"), root)

    from src.agents.explorer import ExplorerAgent
    from src.agents.engineer import EngineerAgent
    from src.agents.builder import BuilderAgent
    from src.agents.coordinator import CoordinatorAgent

    explorer = ExplorerAgent(
        llm_config,
        allowed_dirs,
        explorer_prompt,
        rag_config=rag_config,
        project_root=root,
    )
    pipeline_cfg = config.get("pipeline", {})
    evaluation_cfg = config.get("evaluation", {})

    engineer = EngineerAgent(
        llm_config,
        allowed_dirs,
        engineer_prompt,
        artifacts_dir,
        data_dir,
        pipeline_config=pipeline_cfg,
    )
    builder = BuilderAgent(
        llm_config,
        allowed_dirs,
        builder_prompt,
        artifacts_dir,
        submissions_dir,
        critic_max_iterations=agents_cfg.get("builder", {}).get("critic_max_iterations", 3),
        evaluation_config=evaluation_cfg,
        builder_config=agents_cfg.get("builder", {}),
        robustness_config=config.get("robustness", {}),
    )
    coordinator = CoordinatorAgent(explorer, engineer, builder, llm_config, config=config)

    print("Running multi-agent pipeline: Explorer → Engineer → Builder")
    print("Data dir:", data_dir)
    hs = evaluation_cfg.get("hparam_search") or {}
    print(
        "  Preprocessing encoding:",
        pipeline_cfg.get("encoding", "te_freq"),
        "| evaluation: cv_folds=",
        evaluation_cfg.get("cv_folds", 0),
        "fit_full_train=",
        evaluation_cfg.get("fit_full_train", False),
        "| hparam_search=",
        hs.get("enabled", False),
    )
    result = coordinator.run(data_dir)
    print("Done.")
    print("  Submission:", result.get("submission_path"))
    print("  Val MSE:", result.get("val_mse"))
    if result.get("model_summary"):
        print("  Summary:", result["model_summary"][:200])
    am = result.get("agent_metrics") or {}
    tot = am.get("total") or {}
    if tot.get("total_tokens") or tot.get("llm_api_calls"):
        print(
            "  LLM usage (all agents): total_tokens=",
            tot.get("total_tokens"),
            "api_calls=",
            tot.get("llm_api_calls"),
        )

    from src.memory.experiments import save_experiment

    log_path = save_experiment(
        None,
        {
            "entry": "run.py",
            "config": str(args.config),
            "experiment_label": args.experiment_label,
            "pipeline": pipeline_cfg,
            "evaluation": evaluation_cfg,
            "agent_metrics": am,
            "hparam_trials": result.get("hparam_trials"),
            "hparam_best": result.get("hparam_best"),
            "hparam_selection_note": result.get("hparam_selection_note"),
            "hparam_mode": result.get("hparam_mode"),
            "builder_mode": result.get("builder_mode"),
        },
        result.get("val_mse"),
        str(result.get("submission_path", "")),
        artifacts_dir,
    )
    print("  Experiment log:", log_path)


if __name__ == "__main__":
    main()
