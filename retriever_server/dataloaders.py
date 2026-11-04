"""Passage corpora of the dense retriever.

load_corpus(corpus_name) returns {doc_id: {"title": ..., "text": ...}}. serve_beir.py and
build_beir_index.py both load corpora through this module, so the doc ids stored with a
FAISS index always refer to the passages the server returns.

  corpus           source (relative to the repository root by default)
  hotpotqa         <processed_data>/hotpotqa/corpus.jsonl         (preprocess_hotpotqa_corpus.py)
  2wikimultihopqa  <processed_data>/2wikimultihopqa/corpus.jsonl  (preprocess_2wikimultihopqa_corpus.py)
  musique          <raw_data>/musique/musique_{full,ans}_v1.0_{train,dev,test}.jsonl
  wiki             <raw_data>/nq/psgs_w100.tsv                    (DPR Wikipedia passages; NQ, Bamboogle)

<raw_data> and <processed_data> are SPARC_RAG_RAW_DATA_ROOT and SPARC_RAG_PROCESSED_DATA_ROOT
(defaults: raw_data, processed_data).
"""

import csv
import json
import os
from typing import Dict

from tqdm import tqdm

RAW_DATA_ROOT = os.environ.get("SPARC_RAG_RAW_DATA_ROOT", "raw_data")
PROCESSED_DATA_ROOT = os.environ.get("SPARC_RAG_PROCESSED_DATA_ROOT", "processed_data")

DENSE_CORPORA = ("hotpotqa", "2wikimultihopqa", "musique", "wiki")

Corpus = Dict[str, Dict[str, str]]


def load_jsonl_corpus(corpus_file: str) -> Corpus:
    """Load a corpus.jsonl ({"_id", "title", "text"} per line) written by a preprocess_*_corpus.py script."""
    if not os.path.exists(corpus_file):
        raise FileNotFoundError(
            f"{corpus_file} not found. Create it with retriever_server/preprocess_<dataset>_corpus.py."
        )
    corpus = {}
    with open(corpus_file, encoding="utf8") as file:
        for line in tqdm(file, desc=f"Loading {corpus_file}"):
            doc = json.loads(line)
            corpus[doc["_id"]] = {"text": doc["text"], "title": doc["title"]}
    return corpus


def load_musique_corpus(data_directory: str) -> Corpus:
    """Load the paragraphs of all MuSiQue-Full and MuSiQue-Ans splits.

    Ids are "<question id>_<paragraph idx>". An id that occurs again in a later file is
    overwritten, then exact (title, text) duplicates are dropped, keeping the first id.
    """
    corpus = {}
    for variant in ("full", "ans"):
        for split in ("train", "dev", "test"):
            file_path = os.path.join(data_directory, f"musique_{variant}_v1.0_{split}.jsonl")
            with open(file_path, encoding="utf8") as file:
                for line in tqdm(file, desc=f"Loading {file_path}"):
                    item = json.loads(line)
                    for paragraph in item["paragraphs"]:
                        corpus[f"{item['id']}_{paragraph['idx']}"] = {
                            "title": paragraph["title"],
                            "text": paragraph["paragraph_text"],
                        }
    seen, deduplicated = set(), {}
    for doc_id, doc in corpus.items():
        key = (doc["title"], doc["text"])
        if key not in seen:
            seen.add(key)
            deduplicated[doc_id] = doc
    return deduplicated


def load_dpr_wiki_corpus(tsv_file: str) -> Corpus:
    """Load the DPR Wikipedia passages (psgs_w100.tsv with columns id, text, title)."""
    corpus = {}
    with open(tsv_file, encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t")
        next(reader)  # header
        for doc_id, text, title in tqdm(reader, desc=f"Loading {tsv_file}"):
            corpus[doc_id] = {"title": title.strip(), "text": text.strip()}
    return corpus


def load_corpus(corpus_name: str) -> Corpus:
    if corpus_name == "hotpotqa":
        return load_jsonl_corpus(os.path.join(PROCESSED_DATA_ROOT, "hotpotqa", "corpus.jsonl"))
    if corpus_name == "2wikimultihopqa":
        return load_jsonl_corpus(os.path.join(PROCESSED_DATA_ROOT, "2wikimultihopqa", "corpus.jsonl"))
    if corpus_name == "musique":
        return load_musique_corpus(os.path.join(RAW_DATA_ROOT, "musique"))
    if corpus_name == "wiki":
        return load_dpr_wiki_corpus(os.path.join(RAW_DATA_ROOT, "nq", "psgs_w100.tsv"))
    raise ValueError(f"Unknown corpus {corpus_name!r}; expected one of {DENSE_CORPORA}.")
