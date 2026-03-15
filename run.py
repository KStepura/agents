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
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config = load_config(args.config)
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

    llm_config = config.get("llm", {})
    security = config.get("security", {})
    allowed_dirs = security.get("allowed_data_dirs", ["data", "artifacts", "submissions"])
    allowed_dirs = [str(root / d) for d in allowed_dirs]
    agents_cfg = config.get("agents", {})

    explorer_prompt = load_system_prompt(agents_cfg.get("explorer", {}).get("system_prompt_file", "config/prompts/explorer_system.txt"), root)
    engineer_prompt = load_system_prompt(agents_cfg.get("engineer", {}).get("system_prompt_file", "config/prompts/engineer_system.txt"), root)
    builder_prompt = load_system_prompt(agents_cfg.get("builder", {}).get("system_prompt_file", "config/prompts/builder_system.txt"), root)

    from src.agents.explorer import ExplorerAgent
    from src.agents.engineer import EngineerAgent
    from src.agents.builder import BuilderAgent
    from src.agents.coordinator import CoordinatorAgent

    explorer = ExplorerAgent(llm_config, allowed_dirs, explorer_prompt)
    engineer = EngineerAgent(llm_config, allowed_dirs, engineer_prompt, artifacts_dir, data_dir)
    builder = BuilderAgent(
        llm_config,
        allowed_dirs,
        builder_prompt,
        artifacts_dir,
        submissions_dir,
        critic_max_iterations=agents_cfg.get("builder", {}).get("critic_max_iterations", 3),
    )
    coordinator = CoordinatorAgent(explorer, engineer, builder, llm_config)

    print("Running multi-agent pipeline: Explorer → Engineer → Builder")
    print("Data dir:", data_dir)
    result = coordinator.run(data_dir)
    print("Done.")
    print("  Submission:", result.get("submission_path"))
    print("  Val MSE:", result.get("val_mse"))
    if result.get("model_summary"):
        print("  Summary:", result["model_summary"][:200])


if __name__ == "__main__":
    main()
