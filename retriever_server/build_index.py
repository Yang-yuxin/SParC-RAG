"""Build the Elasticsearch BM25 index of one corpus (IRCoT's indexing procedure).

Each paragraph is indexed with the fields title, paragraph_text, paragraph_index, url and
is_abstract; title, paragraph_text and url use Elasticsearch's English analyzer. The index
is named after the corpus, which is the corpus_name the retriever servers expect.

  corpus           paragraphs (from <raw_data>, see download/raw_data.sh)
  hotpotqa         Wikipedia abstracts in hotpotqa/wikpedia-paragraphs/
  2wikimultihopqa  context paragraphs of 2wikimultihopqa/{train,dev,test}.json, deduplicated
  musique          paragraphs of musique/musique_{ans,full}_v1.0_{dev,test,train}.jsonl, deduplicated
  wiki             DPR Wikipedia passages, nq/psgs_w100.tsv (used for NQ and Bamboogle)

Run from the repository root with Elasticsearch 7.10 running, e.g.
    python retriever_server/build_index.py hotpotqa
Elasticsearch is reached at ELASTICSEARCH_HOST:ELASTICSEARCH_PORT (default localhost:9200).
"""

import argparse
import bz2
import csv
import glob
import hashlib
import io
import json
import os
from typing import Any, Dict, Iterator

import base58
import dill
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm

RAW_DATA_ROOT = os.environ.get("SPARC_RAG_RAW_DATA_ROOT", "raw_data")
ELASTICSEARCH_HOST = os.environ.get("ELASTICSEARCH_HOST", "localhost")
ELASTICSEARCH_PORT = int(os.environ.get("ELASTICSEARCH_PORT", "9200"))

INDEX_SETTINGS = {
    "mappings": {
        "properties": {
            "title": {"type": "text", "analyzer": "english"},
            "paragraph_index": {"type": "integer"},
            "paragraph_text": {"type": "text", "analyzer": "english"},
            "url": {"type": "text", "analyzer": "english"},
            "is_abstract": {"type": "boolean"},
        }
    }
}


def hash_object(o: Any) -> str:
    """Return a base58 blake2b hash of the dill serialisation of an arbitrary Python object."""
    m = hashlib.blake2b()
    with io.BytesIO() as buffer:
        dill.dump(o, buffer)
        m.update(buffer.getbuffer())
        return base58.b58encode(m.digest()).decode()


def make_document(elasticsearch_index: str, doc_number: int, paragraph: Dict) -> Dict:
    return {"_op_type": "create", "_index": elasticsearch_index, "_id": doc_number, "_source": paragraph}


def make_hotpotqa_documents(elasticsearch_index: str) -> Iterator[Dict]:
    raw_glob_filepath = os.path.join(RAW_DATA_ROOT, "hotpotqa", "wikpedia-paragraphs", "*", "wiki_*.bz2")
    doc_number = 1
    for filepath in tqdm(sorted(glob.glob(raw_glob_filepath))):
        with bz2.BZ2File(filepath) as file:
            data = file.readlines()
        for datum in data:
            instance = json.loads(datum.strip())
            paragraph = {
                "id": hash_object(instance)[:32],
                "title": instance["title"],
                "paragraph_index": 0,
                "paragraph_text": " ".join(e.strip() for e in instance["text"]),
                "url": instance["url"],
                "is_abstract": True,
            }
            yield make_document(elasticsearch_index, doc_number, paragraph)
            doc_number += 1


def make_2wikimultihopqa_documents(elasticsearch_index: str) -> Iterator[Dict]:
    raw_filepaths = [os.path.join(RAW_DATA_ROOT, "2wikimultihopqa", f"{split}.json") for split in ("train", "dev", "test")]
    doc_number = 1
    used_full_ids = set()
    for raw_filepath in raw_filepaths:
        with open(raw_filepath, "r") as file:
            full_data = json.load(file)
        for instance in tqdm(full_data):
            for title, sentences in instance["context"]:
                paragraph_text = " ".join(sentences)
                full_id = hash_object(" ".join([title, paragraph_text]))
                if full_id in used_full_ids:
                    continue
                used_full_ids.add(full_id)
                paragraph = {
                    "id": full_id[:32],
                    "title": title,
                    "paragraph_index": 0,
                    "paragraph_text": paragraph_text,
                    "url": "",
                    "is_abstract": True,
                }
                yield make_document(elasticsearch_index, doc_number, paragraph)
                doc_number += 1


def make_musique_documents(elasticsearch_index: str) -> Iterator[Dict]:
    raw_filepaths = [
        os.path.join(RAW_DATA_ROOT, "musique", f"musique_{variant}_v1.0_{split}.jsonl")
        for variant in ("ans", "full")
        for split in ("dev", "test", "train")
    ]
    doc_number = 1
    used_full_ids = set()
    for raw_filepath in raw_filepaths:
        with open(raw_filepath, "r") as file:
            for line in tqdm(file.readlines()):
                if not line.strip():
                    continue
                instance = json.loads(line)
                for raw_paragraph in instance["paragraphs"]:
                    title = raw_paragraph["title"]
                    paragraph_text = raw_paragraph["paragraph_text"]
                    full_id = hash_object(" ".join([title, paragraph_text]))
                    if full_id in used_full_ids:
                        continue
                    used_full_ids.add(full_id)
                    paragraph = {
                        "id": full_id[:32],
                        "title": title,
                        "paragraph_index": 0,
                        "paragraph_text": paragraph_text,
                        "url": "",
                        "is_abstract": True,
                    }
                    yield make_document(elasticsearch_index, doc_number, paragraph)
                    doc_number += 1


def make_wiki_documents(elasticsearch_index: str) -> Iterator[Dict]:
    raw_filepath = os.path.join(RAW_DATA_ROOT, "nq", "psgs_w100.tsv")
    doc_number = 1
    with open(raw_filepath) as input_file:
        reader = csv.reader(input_file, delimiter="\t")
        next(reader)  # header: id, text, title
        for _, paragraph_text, title in tqdm(reader):
            paragraph = {
                "id": hash_object(" ".join([title, paragraph_text]))[:32],
                "title": title,
                "paragraph_index": 0,
                "paragraph_text": paragraph_text,
                "url": "",
                "is_abstract": True,
            }
            yield make_document(elasticsearch_index, doc_number, paragraph)
            doc_number += 1


DOCUMENT_GENERATORS = {
    "hotpotqa": make_hotpotqa_documents,
    "2wikimultihopqa": make_2wikimultihopqa_documents,
    "musique": make_musique_documents,
    "wiki": make_wiki_documents,
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Paths follow SPARC_RAG_RAW_DATA_ROOT (default: raw_data).",
    )
    parser.add_argument("corpus_name", choices=tuple(DOCUMENT_GENERATORS))
    parser.add_argument("--force", action="store_true", help="Delete and rebuild the index if it already exists.")
    args = parser.parse_args()

    elasticsearch_index = args.corpus_name
    es = Elasticsearch(
        [ELASTICSEARCH_HOST],
        scheme="http",
        port=ELASTICSEARCH_PORT,
        max_retries=20,
        timeout=2000,
        retry_on_timeout=True,
    )

    if es.indices.exists(elasticsearch_index):
        if not args.force:
            parser.error(f"Index {elasticsearch_index} already exists; pass --force to delete and rebuild it.")
        es.indices.delete(index=elasticsearch_index)

    print(f"Creating index {elasticsearch_index} ...")
    es.indices.create(index=elasticsearch_index, ignore=400, body=INDEX_SETTINGS)

    print("Inserting paragraphs ...")
    result = bulk(
        es,
        DOCUMENT_GENERATORS[args.corpus_name](elasticsearch_index),
        raise_on_error=True,  # otherwise failed documents are dropped silently
        raise_on_exception=True,
        max_retries=2,  # exponential backoff starting at 2 s
        request_timeout=500,
    )
    es.indices.refresh(elasticsearch_index)
    print(f"Index {elasticsearch_index} is ready. Added {result[0]} documents.")


if __name__ == "__main__":
    main()
