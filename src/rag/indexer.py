"""
Index documents for RAG (e.g. EDA guides, Kaggle regression tips).
ChromaDB or FAISS + embedding model.
"""

from __future__ import annotations

from pathlib import Path
from typing import List


def build_index(
    knowledge_dir: str,
    persist_path: str,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> str:
    """Index documents from knowledge_dir; save index to persist_path. Returns persist_path."""
    raise NotImplementedError("Implement: load docs, embed, save Chroma/FAISS index")


def add_documents(chunked_docs: List[dict], persist_path: str) -> None:
    """Add more documents to existing index."""
    raise NotImplementedError("Optional: add docs to existing index")
