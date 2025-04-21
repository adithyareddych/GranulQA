from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer, util

from .llama_backend import generate

_EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # fast & tiny
_TOP_K_EXTRACT = 5


def _key_sentence_indices(text: str, k: int = _TOP_K_EXTRACT) -> List[int]:
    sents = text.split(". ")
    if len(sents) <= k:
        return list(range(len(sents)))
    embeddings = _EMBED_MODEL.encode(sents, convert_to_tensor=True, normalize_embeddings=True)
    centroid = embeddings.mean(dim=0, keepdim=True)
    scores = (embeddings @ centroid.T).squeeze()
    return scores.topk(k).indices.cpu().tolist()


def extractive(text: str, k: int = _TOP_K_EXTRACT) -> str:
    idx = sorted(_key_sentence_indices(text, k))
    return ". ".join(text.split(". ")[i] for i in idx)


def abstractive(text: str, query: Union[str,None] = None) -> str:
    q = f"\nQuestion: {query}\n" if query else ""
    prompt = (
        f"Summarise the following context{q}concisely in 3‑6 sentences. "
        f"Preserve named entities and numbers.\n\nContext:\n{text}\n\nSummary:"
    )
    return generate(prompt).strip()


def hybrid_summary(text: str, query: Union[str,None] = None) -> str:
    ext = extractive(text)
    return abstractive(ext, query=query)


def headline(text: str) -> str:
    prompt = f"Write a short, 7‑word title that best describes the topic of this passage:\n\n{text}\n\nTitle:"
    return generate(prompt)
