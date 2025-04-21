# structure_aware_clustering.py
import re
from typing import List

_MAX_CLUSTER_TOKENS = 350      # comfortably fits in 7‑B context window
_HEADING_PATTERN = re.compile(r"^(#+\s|[A-Z][^a-z]{3,})")   # crude markdown / ALL‑CAPS

def split_on_structure(paragraphs: List[str]) -> List[List[str]]:
    """
    Group paragraphs by heading markers (markdown '#', ALL‑CAPS lines, digits '.', etc.).
    """
    clusters, current = [], []
    for p in paragraphs:
        if _HEADING_PATTERN.match(p.strip()):
            if current:
                clusters.append(current)
            current = [p]
        else:
            current.append(p)
    if current:
        clusters.append(current)
    return clusters


def enforce_max_tokens(clusters: List[List[str]], tokenizer) -> List[List[str]]:
    """
    Ensure each cluster is <= _MAX_CLUSTER_TOKENS. Split greedily otherwise.
    """
    final = []
    for cl in clusters:
        tmp, buf = [], []
        for para in cl:
            if len(tokenizer.encode(" ".join(buf + [para]))) > _MAX_CLUSTER_TOKENS:
                final.append(buf)
                buf = [para]
            else:
                buf.append(para)
        if buf:
            final.append(buf)
    return final
