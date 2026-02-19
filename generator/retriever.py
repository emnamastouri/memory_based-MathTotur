
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import faiss

from generator.data_loader import Exercise
from memory.embedder import Embedder


@dataclass
class RetrievalResult:
    exercise: Exercise
    score: float

def filter_pool(exercises: List[Exercise], section: str, topic: str) -> List[Exercise]:
    return [
        e for e in exercises
        if e.section == section and e.topic == topic
    ]


def build_index_for_pool(embedder: Embedder, pool: List[Exercise]) -> Tuple[faiss.Index, np.ndarray]:
    """
    Builds FAISS index on pool.enonce embeddings.
    Returns: (index, embeddings_matrix)
    """
    texts = [e.enonce for e in pool]
    emb = embedder.embed_texts(texts)  # normalized float32
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    return index, emb


def retrieve_similar(
    embedder: Embedder,
    pool: List[Exercise],
    k: int = 3,
    query: str = "",
) -> List[RetrievalResult]:
    """
    Query can be empty. If empty, we use a generic "style query".
    Because pool is already filtered by grade/section/topic, this still works well.
    """
    if not pool:
        return []

    if not query.strip():
        # default style query; you can tune this later
        query = "Exercice baccalauréat tunisien, même style, même difficulté, même chapitre."

    index, _ = build_index_for_pool(embedder, pool)

    q = embedder.embed_texts([query])  # (1, dim)
    scores, idxs = index.search(q, k)

    results: List[RetrievalResult] = []
    for j in range(idxs.shape[1]):
        i = int(idxs[0, j])
        if 0 <= i < len(pool):
            results.append(RetrievalResult(exercise=pool[i], score=float(scores[0, j])))
    return results
