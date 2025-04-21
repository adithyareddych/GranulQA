from dataclasses import dataclass
from typing import List
from .llama_backend import generate, get_pipeline
from .vector_store import VectorStore

@dataclass
class Candidate:
    node_id: int
    score: float
    text: str


class LlamaReranker:
    """
    Lightweight yes/no relevance check using 7-B in greedy mode.
    """

    _prompt_tpl = (
        "Question:\n{q}\n\nContext:\n{text}\n\n"
        "Answer YES or NO — does this context fully or partially answer the question?"
    )

    def score(self, q: str, ctx: str) -> float:
        prompt = self._prompt_tpl.format(q=q, text=ctx)
        pipe = get_pipeline("text-generation")
        # do_sample=False for greedy decoding; no temperature=0
        out = pipe(
            prompt,
            do_sample=False,
            return_full_text=False,
            # you can still set max_new_tokens if desired, e.g. max_new_tokens=16
        )
        answer = out[0]["generated_text"].strip().lower()
        return 1.0 if answer.startswith("yes") else 0.0


class MultiBranchRetriever:
    def __init__(self, tree_index, vector_store: VectorStore, top_k: int = 3):
        self.tree = tree_index
        self.vs = vector_store
        self.top_k = top_k
        self.reranker = LlamaReranker()

    def retrieve(self, query: str) -> List[Candidate]:
        # retrieve oversampled candidates
        initial = self.vs.search(query, k=self.top_k * 4)
        # rerank with greedy LLM yes/no
        rescored: List[Candidate] = []
        for node_id, text in initial:
            score = self.reranker.score(query, text)
            rescored.append(Candidate(node_id, score, text))
        # sort and take top_k
        rescored.sort(key=lambda c: c.score, reverse=True)
        return rescored[: self.top_k]
