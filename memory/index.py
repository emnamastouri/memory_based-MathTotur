from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import faiss

from memory.schema import MemoryItem
from memory.embedder import Embedder


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a cosine-sim index. We already normalize embeddings, so inner product == cosine similarity.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D (n, dim)")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)


def load_index(path: str) -> faiss.Index:
    return faiss.read_index(path)


def search(
    index: faiss.Index,
    memory_items: List[MemoryItem],
    embedder: Embedder,
    topic: str,
    problem: str,
    student_attempt: str,
    error_type: str,
    k: int = 3,
) -> List[Tuple[MemoryItem, float]]:
    """
    Returns top-k (MemoryItem, score).
    """
    if len(memory_items) == 0 or index.ntotal == 0:
        return []

    query = Embedder.build_query_text(topic, problem, student_attempt, error_type)
    q_emb = embedder.embed_texts([query])  # (1, dim)

    scores, idxs = index.search(q_emb, k)  # (1, k)
    results: List[Tuple[MemoryItem, float]] = []

    for j in range(idxs.shape[1]):
        i = int(idxs[0, j])
        if i < 0 or i >= len(memory_items):
            continue
        results.append((memory_items[i], float(scores[0, j])))

    return results


def build_or_load_index(
    memory_items: List[MemoryItem],
    embedder: Embedder,
    index_path: str,
) -> faiss.Index:
    """
    If index exists, load it. Otherwise build from memory_items and save.
    """
    if os.path.exists(index_path):
        return load_index(index_path)

    texts = [
        Embedder.build_query_text(m.topic, m.problem, m.student_attempt, m.error_type)
        for m in memory_items
    ]
    if len(texts) == 0:
        # create an empty index with a default dim (MiniLM dim is 384)
        index = faiss.IndexFlatIP(384)
        save_index(index, index_path)
        return index

    emb = embedder.embed_texts(texts)
    index = build_faiss_index(emb)
    save_index(index, index_path)
    return index
