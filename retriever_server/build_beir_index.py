"""Build the dense index that serve_beir.py loads for one corpus.

Encodes every passage of the corpus (dataloaders.load_corpus) as "<title><separator><text>"
with BAAI/bge-large-en-v1.5 through BEIR's SentenceBERT wrapper, stores the embeddings in a
FAISS IndexFlatIP and writes, under <processed_data>/<dir>/faiss-index/ (utils.index_directory):
    bge.flat.faiss   the index
    bge.flat.tsv     BEIR doc id -> FAISS id
The separator is BEIR's default " ", except for 2WikiMultiHopQA, whose index in the paper was
built from "title: text".

Run from the repository root, e.g.
    python retriever_server/build_beir_index.py hotpotqa
Encoding uses a GPU when one is available. Building the `wiki` index (21M passages) keeps all
embeddings in memory (86 GB) and needs roughly twice that while they are concatenated.
"""

import argparse
import logging
import os

from beir.retrieval import models

from dataloaders import DENSE_CORPORA, load_corpus
from utils import EMBEDDING_MODEL, INDEX_EXT, INDEX_PREFIX, SCORE_FUNCTION, ConcreteFlatIPFaissSearch, index_directory

TITLE_TEXT_SEPARATORS = {"2wikimultihopqa": ": "}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Paths follow SPARC_RAG_RAW_DATA_ROOT and SPARC_RAG_PROCESSED_DATA_ROOT "
        "(defaults: raw_data, processed_data).",
    )
    parser.add_argument("corpus_name", choices=DENSE_CORPORA)
    parser.add_argument("--batch_size", type=int, default=128, help="Encoding batch size (default: 128).")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing index.")
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    output_directory = index_directory(args.corpus_name)
    index_file = os.path.join(output_directory, f"{INDEX_PREFIX}.{INDEX_EXT}.faiss")
    if os.path.exists(index_file) and not args.force:
        parser.error(f"{index_file} already exists; pass --force to rebuild it.")

    corpus = load_corpus(args.corpus_name)
    model = models.SentenceBERT(EMBEDDING_MODEL, sep=TITLE_TEXT_SEPARATORS.get(args.corpus_name, " "))
    faiss_search = ConcreteFlatIPFaissSearch(model, batch_size=args.batch_size)
    faiss_search.index(corpus, score_function=SCORE_FUNCTION)

    os.makedirs(output_directory, exist_ok=True)
    faiss_search.save(output_directory, prefix=INDEX_PREFIX, ext=INDEX_EXT)
    print(f"Indexed {len(corpus)} passages of {args.corpus_name} in {output_directory}")


if __name__ == "__main__":
    main()
