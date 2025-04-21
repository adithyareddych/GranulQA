from typing import List
from .tree_builder import TreeBuilder
from .vector_store import VectorStore
from .hybrid_summarizer import hybrid_summary, headline
from .structure_aware_clustering import split_on_structure, enforce_max_tokens
from .multi_branch_retriever import MultiBranchRetriever
from .llama_backend import generate
from .forgiving_cfg import ForgivingCfg           # ← keeps TreeBuilder happy

from transformers import AutoTokenizer            # the only HF import we use


# --- tiny wrapper so TreeBuilder always has an embedding model ------------
from sentence_transformers import SentenceTransformer
from .EmbeddingModels import BaseEmbeddingModel          # already in your repo

class MiniLMEmbeddingModel(BaseEmbeddingModel):
    _model = SentenceTransformer("all-MiniLM-L6-v2")     # cached on first use
    def create_embedding(self, text: str):
        return self._model.encode(text).tolist()
# --------------------------------------------------------------------------


class RAPTORLLM:
    """Drop‑in replacement that adds all seven extensions."""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf", use_fast=True
        )
        self.tree = None
        self.vs = VectorStore(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.retriever = None

    # ---------- Indexing --------------------------------------------------
    def index_corpus(self, docs: List[str]) -> None:
        cfg_tb = ForgivingCfg(
            tokenizer            = self.tokenizer,   # ← safer than None
            max_tokens           = 100,
            num_layers           = 5,
            threshold            = 0.5,
            top_k                = 5,
            selection_mode       = "top_k",
            summarization_length = 100,
            embedding_models         = {"MINILM": MiniLMEmbeddingModel()},
            cluster_embedding_model  = "MINILM"
        )
        tb = TreeBuilder(cfg_tb)

        nodes = []
        for doc_id, raw in enumerate(docs):
            paras    = raw.split("\n\n")
            clusters = enforce_max_tokens(split_on_structure(paras), self.tokenizer)

            for cl in clusters:
                text   = "\n\n".join(cl)
                summ   = hybrid_summary(text)
                title  = headline(text)
                node_id = tb.add_leaf(
                    doc_id, text=text, summary=summ, label=title
                )
                self.vs.add(node_id, f"{title} {summ}")
                nodes.append(node_id)

        self.tree      = tb.build_tree(nodes)
        self.retriever = MultiBranchRetriever(self.tree, self.vs)

    def index_corpus_nqa(self, docs: List[str]) -> None:
        cfg_tb = ForgivingCfg(
            tokenizer            = self.tokenizer,   # ← safer than None
            max_tokens           = 100,
            num_layers           = 5,
            threshold            = 0.5,
            top_k                = 5,
            selection_mode       = "top_k",
            summarization_length = 100,
            embedding_models         = {"MINILM": MiniLMEmbeddingModel()},
            cluster_embedding_model  = "MINILM"
        )
        tb = TreeBuilder(cfg_tb)

        nodes = []
        for doc_id, raw in enumerate(docs):
            # NarrativeQA gives raw as a dict—grab the story text:
            if isinstance(raw, dict):
                story = raw.get("story") or raw.get("text") or raw.get("summary", "")
            else:
                story = raw

            paras    = story.split("\n\n")
            clusters = enforce_max_tokens(split_on_structure(paras), self.tokenizer)

            for cl in clusters:
                text   = "\n\n".join(cl)
                summ   = hybrid_summary(text)
                title  = headline(text)
                node_id = tb.add_leaf(doc_id, text=text, summary=summ, label=title)
                self.vs.add(node_id, f"{title} {summ}")
                nodes.append(node_id)

        self.tree      = tb.build_tree(nodes)
        self.retriever = MultiBranchRetriever(self.tree, self.vs)


    # ---------- Retrieval -------------------------------------------------
    def retrieve(self, query: str, k: int = 3):
        cands = self.retriever.retrieve(query)
        return [
            (c.node_id, c.text, self.tree.get_summary(c.node_id)) for c in cands
        ]

    # ---------- QA synthesis ----------------------------------------------
    def answer(self, query: str) -> str:
        contexts = [
            hybrid_summary(text, query=query)
            for _, text, _ in self.retrieve(query)
        ]
        prompt = (
            "Answer the following question using ONLY the provided contexts.\n\n"
            f"Question: {query}\n\n"
            + "\n\n".join(f"Context {i+1}: {c}" for i, c in enumerate(contexts))
            + "\n\nAnswer (cite facts explicitly):"
        )
        return generate(prompt, temperature=0.2)
