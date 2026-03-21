#!/usr/bin/env python3
"""
Сравнение двух сценариев мультиагентной системы (пример: RAG вкл / выкл).
Запускает run.py дважды с разными конфигами, сохраняет копии submission и метки в experiments.jsonl.

  python scripts/compare_architectures.py
  python scripts/compare_architectures.py --skip-rag-on   # только no-RAG
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two architecture configs (e.g. RAG on/off)")
    parser.add_argument("--config-rag", default="config/settings.yaml", help="Scenario A (default: RAG on)")
    parser.add_argument("--config-norag", default="config/settings.norag.yaml", help="Scenario B (default: RAG off)")
    parser.add_argument("--skip-rag-on", action="store_true", help="Run only norag scenario")
    args = parser.parse_args()

    py = sys.executable
    run_py = ROOT / "run.py"

    def run_scenario(cfg_rel: str, label: str) -> None:
        cfg = ROOT / cfg_rel
        print(f"\n=== Scenario {label} ({cfg_rel}) ===\n")
        r = subprocess.run(
            [py, str(run_py), "--config", str(cfg), "--experiment-label", label],
            cwd=str(ROOT),
        )
        if r.returncode != 0:
            print(f"Warning: run exited with {r.returncode}", file=sys.stderr)
        sub = ROOT / "submissions" / "submission.csv"
        if sub.exists():
            out = ROOT / "submissions" / f"submission_{label}.csv"
            shutil.copy2(sub, out)
            print(f"Saved copy: {out.relative_to(ROOT)}")

    if not args.skip_rag_on:
        run_scenario(args.config_rag, "arch_rag_on")
    run_scenario(args.config_norag, "arch_rag_off")

    print(
        "\nCompare Val MSE and agent_metrics in artifacts/experiments.jsonl "
        "(lines with experiment_label arch_rag_on / arch_rag_off)."
    )


if __name__ == "__main__":
    main()
