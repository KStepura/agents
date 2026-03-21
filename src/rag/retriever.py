"""
Retrieve relevant chunks for agent context (RAG).
"""

from __future__ import annotations

from typing import List

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


def retrieve(
    query: str,
    persist_path: str,
    top_k: int = 5,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[str]:
    """Return top_k relevant text chunks for the query."""
    import chromadb

    if not query or not query.strip():
        return []

    ef = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    client = chromadb.PersistentClient(path=persist_path)
    collection = client.get_collection(name="knowledge", embedding_function=ef)
    k = max(1, min(int(top_k), 20))
    res = collection.query(query_texts=[query], n_results=k)
    docs = res.get("documents") or []
    if docs and docs[0]:
        return list(docs[0])
    return []
