"""Dense index settings and FAISS search class shared by build_beir_index.py and serve_beir.py."""

import importlib.util  # noqa: F401  (beir 2.2.0's dense/exact_search.py calls importlib.util without importing it)
import os

from beir.retrieval.search.dense import FlatIPFaissSearch

from dataloaders import PROCESSED_DATA_ROOT

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
SCORE_FUNCTION = "dot"  # BGE embeddings are unit-normalised, so this is cosine similarity
INDEX_PREFIX = "bge"
INDEX_EXT = "flat"

# Directory under <processed_data> whose faiss-index/ holds each corpus's index.
INDEX_PARENT_DIRECTORIES = {"hotpotqa": "hotpotqa", "2wikimultihopqa": "2wikimultihopqa", "musique": "musique", "wiki": "nq"}


def index_directory(corpus_name: str) -> str:
    """Directory holding <INDEX_PREFIX>.<INDEX_EXT>.faiss and .tsv (FAISS index, BEIR-id mapping)."""
    return os.path.join(PROCESSED_DATA_ROOT, INDEX_PARENT_DIRECTORIES[corpus_name], "faiss-index")


class ConcreteFlatIPFaissSearch(FlatIPFaissSearch):
    """FlatIPFaissSearch (exact inner-product search) that can be instantiated under beir 2.2.0.

    beir 2.2.0's BaseSearch declares encode() and search_from_files() abstract, but its FAISS
    search classes do not implement them.
    """

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def search_from_files(self, *args, **kwargs):
        raise NotImplementedError
