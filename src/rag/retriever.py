"""
Retrieve relevant chunks for agent context (RAG).
"""

from __future__ import annotations

from typing import List


def retrieve(query: str, persist_path: str, top_k: int = 5) -> List[str]:
    """Return top_k relevant text chunks for the query."""
    raise NotImplementedError("Implement: load index, embed query, search, return chunks")
