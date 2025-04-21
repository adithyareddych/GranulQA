# raptor/vector_store.py

import pickle
from typing import List, Tuple, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
    """
    Simple FAISS-backed vector store using SentenceTransformer.
    Supports incremental .add(node_id, text) and .search(query, k).
    """

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # 1) Load the embedder
        self.embedder = SentenceTransformer(embedding_model)
        # 2) Build an inner‑product index (with L2‑normalized vectors → cosine sim)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)
        # 3) Track the mapping idx → node_id and store texts
        self._ids: List[int] = []
        self._id2text: Dict[int, str] = {}

    def add(self, node_id: int, text: str) -> None:
        """
        Embed `text` and add it under `node_id`.
        """
        vec = self.embedder.encode(text, convert_to_numpy=True)
        vec = np.asarray(vec, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(vec)
        self.index.add(vec)
        self._ids.append(node_id)
        self._id2text[node_id] = text

    def search(self, query: str, k: int = 4) -> List[Tuple[int, str]]:
        """
        Return up to `k` (node_id, text) pairs most similar to `query`.
        """
        qvec = self.embedder.encode(query, convert_to_numpy=True)
        qvec = np.asarray(qvec, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(qvec)

        _, indices = self.index.search(qvec, k)
        results: List[Tuple[int, str]] = []
        for idx in indices[0]:
            if idx < len(self._ids):
                nid = self._ids[idx]
                results.append((nid, self._id2text[nid]))
        return results

    def save(self, path: str) -> None:
        """
        Persist the FAISS index and the id mapping.
        """
        faiss.write_index(self.index, path + ".index")
        with open(path + ".ids", "wb") as f:
            pickle.dump(self._ids, f)

    def load(self, path: str) -> None:
        """
        Load a previously saved index and id mapping.
        """
        self.index = faiss.read_index(path + ".index")
        with open(path + ".ids", "rb") as f:
            self._ids = pickle.load(f)
        # Note: you may want to rebuild _id2text from source or store it separately.
