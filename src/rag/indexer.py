"""
Index documents for RAG (e.g. EDA guides, Kaggle regression tips).
Uses ChromaDB persistent store + sentence-transformers embeddings.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


def _chunk_text(text: str, max_chars: int = 900) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    for p in paragraphs:
        if len(p) <= max_chars:
            chunks.append(p)
        else:
            for i in range(0, len(p), max_chars):
                chunks.append(p[i : i + max_chars])
    return chunks if chunks else [text[:max_chars]]


def build_index(
    knowledge_dir: str,
    persist_path: str,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> str:
    """Index documents from knowledge_dir; save index to persist_path. Returns persist_path."""
    import chromadb

    kdir = Path(knowledge_dir)
    if not kdir.is_dir():
        raise FileNotFoundError(f"Knowledge dir not found: {knowledge_dir}")

    Path(persist_path).parent.mkdir(parents=True, exist_ok=True)
    ef = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    client = chromadb.PersistentClient(path=persist_path)

    try:
        client.delete_collection("knowledge")
    except Exception:
        pass

    collection = client.create_collection(name="knowledge", embedding_function=ef)

    docs: List[str] = []
    ids: List[str] = []
    for path in sorted(kdir.rglob("*")):
        if path.suffix.lower() not in (".md", ".txt", ".markdown"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(kdir)
        for j, chunk in enumerate(_chunk_text(text)):
            docs.append(chunk)
            ids.append(f"{rel.as_posix().replace('/', '_')}_{j}")

    if not docs:
        raise ValueError(
            f"No .md/.txt documents found under {knowledge_dir}. Add knowledge files first."
        )

    collection.add(ids=ids, documents=docs)
    return persist_path
