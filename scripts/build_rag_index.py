#!/usr/bin/env python3
"""
Построить векторный индекс RAG из каталога knowledge/ (Markdown/текст).

Запуск из корня проекта:
  pip install chromadb sentence-transformers
  python scripts/build_rag_index.py
  python scripts/build_rag_index.py --config config/settings.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chroma RAG index from knowledge/")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--knowledge-dir", default=None, help="Override knowledge directory")
    parser.add_argument("--persist-path", default=None, help="Override Chroma persist directory")
    args = parser.parse_args()

    import yaml

    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)
    paths = cfg.get("paths", {})
    rag = cfg.get("rag", {})

    knowledge_dir = args.knowledge_dir or paths.get("knowledge_dir", "knowledge")
    persist = args.persist_path or rag.get("persist_path", "artifacts/chroma_rag")
    embedding_model = rag.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")

    kdir = ROOT / knowledge_dir
    out = ROOT / persist if not Path(persist).is_absolute() else Path(persist)

    from src.rag.indexer import build_index

    p = build_index(str(kdir), str(out), embedding_model=embedding_model)
    print("RAG index built at:", p)
    print("Documents from:", kdir)


if __name__ == "__main__":
    main()
