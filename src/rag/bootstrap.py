"""
Гарантирует наличие индекса Chroma для RAG при запуске мультиагентного пайплайна (полностью автоматический режим).
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("mws_agents.rag")


def ensure_rag_index_if_needed(
    *,
    rag_config: dict,
    paths: dict,
    project_root: Path,
) -> None:
    """"
    Если RAG включён и директория индекса отсутствует или пуста, создаёт индекс из knowledge_dir.
    Ничего не делает, если индекс уже существует, RAG отключён или нет документов для индексации.
    """
    if not rag_config.get("enabled", True):
        return

    rel_persist = rag_config.get("persist_path", "artifacts/chroma_rag")
    persist = Path(rel_persist)
    if not persist.is_absolute():
        persist = (project_root / rel_persist).resolve()

    if persist.exists() and any(persist.iterdir()):
        return

    knowledge_rel = paths.get("knowledge_dir", "knowledge")
    kdir = Path(knowledge_rel)
    if not kdir.is_absolute():
        kdir = (project_root / knowledge_rel).resolve()

    if not kdir.is_dir():
        logger.warning("RAG enabled but knowledge directory not found: %s", kdir)
        return

    has_docs = any(
        p.suffix.lower() in (".md", ".txt", ".markdown")
        for p in kdir.rglob("*")
        if p.is_file()
    )
    if not has_docs:
        logger.warning("RAG enabled but no .md/.txt under %s; Explorer runs without RAG context", kdir)
        return

    from src.rag.indexer import build_index

    embedding_model = rag_config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    print(
        "RAG: building vector index (first run; may download the embedding model)...",
        flush=True,
    )
    build_index(str(kdir), str(persist), embedding_model=embedding_model)
    print(f"RAG: index ready at {persist}", flush=True)
