"""
tree_builder.py  – Replaces the original file.

• Keeps TreeBuilderConfig and TreeBuilder exactly as in your fork.
• Adds add_leaf() and build_tree() only if they don’t already exist.
  This avoids breaking future upstream updates.
"""

import copy
import logging
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple, Union

import tiktoken

from .EmbeddingModels import BaseEmbeddingModel, OpenAIEmbeddingModel
from .SummarizationModels import BaseSummarizationModel, GPT3TurboSummarizationModel
from .tree_structures import Node, Tree
from .utils import (
    distances_from_embeddings,
    get_text,
    get_embeddings,
    indices_of_nearest_neighbors_from_distances,
    split_text,
)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class TreeBuilderConfig:
    def __init__(
        self,
        tokenizer=None,
        max_tokens: int = 100,
        num_layers: int = 5,
        threshold: float = 0.5,
        top_k: int = 5,
        selection_mode: str = "top_k",
        summarization_length: int = 100,
        summarization_model: Union[BaseSummarizationModel,None] = None,
        embedding_models: Union[Dict[str, BaseEmbeddingModel], None] = None,
        cluster_embedding_model: Union[str, None] = None,
    ):
        # ---- tokenizer ----------------------------------------------------
        self.tokenizer = tokenizer or tiktoken.get_encoding("cl100k_base")

        # ---- basic params --------------------------------------------------
        if max_tokens < 1:
            raise ValueError("max_tokens must be ≥ 1")
        self.max_tokens = max_tokens

        if num_layers < 1:
            raise ValueError("num_layers must be ≥ 1")
        self.num_layers = num_layers

        if not (0 <= threshold <= 1):
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = threshold

        if top_k < 1:
            raise ValueError("top_k must be ≥ 1")
        self.top_k = top_k

        if selection_mode not in ("top_k", "threshold"):
            raise ValueError("selection_mode must be 'top_k' or 'threshold'")
        self.selection_mode = selection_mode

        # ---- summarisation -------------------------------------------------
        self.summarization_length = summarization_length
        self.summarization_model = (
            summarization_model or GPT3TurboSummarizationModel()
        )

        # ---- embeddings ----------------------------------------------------
        self.embedding_models = embedding_models or {
            "EMB": OpenAIEmbeddingModel()
        }
        if not isinstance(self.embedding_models, dict):
            raise ValueError("embedding_models must be a dict of name → model")

        for m in self.embedding_models.values():
            if not isinstance(m, BaseEmbeddingModel):
                raise ValueError("Every embedding model must derive BaseEmbeddingModel")

        self.cluster_embedding_model = (
            cluster_embedding_model or next(iter(self.embedding_models))
        )
        if self.cluster_embedding_model not in self.embedding_models:
            raise ValueError(
                "cluster_embedding_model key not present in embedding_models"
            )

    # ------------------------------------------------------------------- #
    def log_config(self) -> str:
        return str(self.__dict__)



class TreeBuilder:
    """
    Builds a hierarchical abstraction (“tree”) over a document by recursively
    clustering, summarizing, and embedding chunks of text.
    """

    def __init__(self, config: TreeBuilderConfig):
        # copy config params onto self
        self.tokenizer                 = config.tokenizer
        self.max_tokens                = config.max_tokens
        self.num_layers                = config.num_layers
        self.top_k                     = config.top_k
        self.threshold                 = config.threshold
        self.selection_mode            = config.selection_mode
        self.summarization_length      = config.summarization_length
        self.summarization_model       = config.summarization_model
        self.embedding_models          = config.embedding_models
        self.cluster_embedding_model   = config.cluster_embedding_model

        logging.info(
            f"Successfully initialized TreeBuilder with Config {config.log_config()}"
        )

    def create_node(
        self, index: int, text: str, children_indices: Optional[Set[int]] = None
    ) -> Tuple[int, Node]:
        if children_indices is None:
            children_indices = set()

        embeddings = {
            name: model.create_embedding(text)
            for name, model in self.embedding_models.items()
        }
        return index, Node(text, index, children_indices, embeddings)

    def summarize(self, context: str, max_tokens: int = 150) -> str:
        return self.summarization_model.summarize(context, max_tokens)

    def get_relevant_nodes(self, current_node, list_nodes) -> List[Node]:
        embeddings = get_embeddings(list_nodes, self.cluster_embedding_model)
        distances  = distances_from_embeddings(
            current_node.embeddings[self.cluster_embedding_model], embeddings
        )
        indices = indices_of_nearest_neighbors_from_distances(distances)

        if self.selection_mode == "threshold":
            best = [idx for idx in indices if distances[idx] > self.threshold]
        else:  # "top_k"
            best = indices[: self.top_k]

        return [list_nodes[i] for i in best]

    def multithreaded_create_leaf_nodes(self, chunks: List[str]) -> Dict[int, Node]:
        with ThreadPoolExecutor() as pool:
            futures = {
                pool.submit(self.create_node, i, txt): i for i, txt in enumerate(chunks)
            }
            leaves = {i: fut.result()[1] for fut, i in futures.items()}
        return leaves

    def build_from_text(self, text: str, use_multithreading: bool = True) -> Tree:
        chunks = split_text(text, self.tokenizer, self.max_tokens)
        logging.info("Creating Leaf Nodes")

        leaf_nodes = (
            self.multithreaded_create_leaf_nodes(chunks)
            if use_multithreading
            else {i: self.create_node(i, t)[1] for i, t in enumerate(chunks)}
        )

        layer_to_nodes = {0: list(leaf_nodes.values())}
        logging.info(f"Created {len(leaf_nodes)} Leaf Embeddings")

        all_nodes = copy.deepcopy(leaf_nodes)
        root_nodes = self.construct_tree(
            all_nodes, all_nodes, layer_to_nodes, use_multithreading
        )
        return Tree(all_nodes, root_nodes, leaf_nodes,
                    self.num_layers, layer_to_nodes)

    @abstractmethod
    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = True,
    ) -> Dict[int, Node]:
        """Return dict of root nodes (implementation specific)."""
        ...


from types import SimpleNamespace

if not hasattr(TreeBuilder, "add_leaf"):

    def _add_leaf(self: TreeBuilder, doc_id: int, text: str,
                  summary: str = "", label: str = "") -> int:
        """
        Minimal helper for external pipelines:
        – stores the text as a leaf Node (ignores summary/label unless you want
          to persist them yourself)
        – returns its index
        """
        if not hasattr(self, "_ext_nodes"):
            self._ext_nodes = {}
        idx, node = self.create_node(len(self._ext_nodes), text)
        # Optionally stash extra metadata on the node object:
        node.summary = summary
        node.label   = label
        self._ext_nodes[idx] = node
        return idx

    TreeBuilder.add_leaf = _add_leaf


if not hasattr(TreeBuilder, "build_tree"):

    def _build_tree(self: TreeBuilder, node_indices):
        """
        Converts nodes added via add_leaf() into a complete Tree.
        Delegates clustering to self.construct_tree().
        """
        if not hasattr(self, "_ext_nodes") or not self._ext_nodes:
            raise ValueError("No nodes added via add_leaf()")

        all_nodes       = self._ext_nodes
        layer_to_nodes  = {0: list(all_nodes.values())}
        root_nodes = self.construct_tree(
            all_nodes, all_nodes, layer_to_nodes, use_multithreading=True
        )
        return Tree(all_nodes, root_nodes, all_nodes,
                    self.num_layers, layer_to_nodes)

    TreeBuilder.build_tree = _build_tree
