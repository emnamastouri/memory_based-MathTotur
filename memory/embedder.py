
from __future__ import annotations

from typing import List
import numpy as np

from sentence_transformers import SentenceTransformer


DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Returns float32 matrix: shape (n, dim)
        """
        emb = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return emb.astype("float32")

    @staticmethod
    def build_query_text(topic: str, problem: str, student_attempt: str, error_type: str) -> str:
        """
        Pack fields into one retrieval query string.
        """
        return f"topic: {topic}\nproblem: {problem}\nstudent_attempt: {student_attempt}\nerror_type: {error_type}"
    
    def add_document(self, text: str, metadata: dict | None = None):
        """
        Minimal 'upsert' alternative:
        - store the text + metadata in the embedder's internal store (if you have one)
        - or compute embedding and append to your vector index
        """
        metadata = metadata or {}

        # If your Embedder already has something like `add_texts`, `index`, `documents`, etc,
        # adapt the next lines accordingly.
        if hasattr(self, "add_texts"):
            return self.add_texts([text], metadatas=[metadata])

        if hasattr(self, "index") and hasattr(self.index, "add"):
            return self.index.add(text=text, metadata=metadata)

        # Fallback: keep an in-memory list (won't persist unless you implement save())
        if not hasattr(self, "_docs"):
            self._docs = []
        self._docs.append({"text": text, "metadata": metadata})
        return True
