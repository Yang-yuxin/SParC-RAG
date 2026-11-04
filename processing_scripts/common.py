"""Helpers shared by the dataset-processing scripts.

Run the scripts from the repository root. Data is read from and written to raw_data/ and
processed_data/ there unless SPARC_RAG_RAW_DATA_ROOT / SPARC_RAG_PROCESSED_DATA_ROOT point
elsewhere.
"""

import json
import os
import sys
from typing import Dict, List, Optional

import _jsonnet
import requests
from rapidfuzz import fuzz

RAW_DATA_ROOT = os.environ.get("SPARC_RAG_RAW_DATA_ROOT", "raw_data")
PROCESSED_DATA_ROOT = os.environ.get("SPARC_RAG_PROCESSED_DATA_ROOT", "processed_data")

RETRIEVER_ADDRESS_FILE = ".retriever_address.jsonnet"
RETRIEVER_TIMEOUT = 300  # seconds


def read_json(file_path: str):
    with open(file_path, "r", encoding="utf8", errors="ignore") as file:
        return json.load(file)


def read_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf8") as file:
        return [json.loads(line.strip()) for line in file if line.strip()]


def write_jsonl(instances: List[Dict], file_path: str):
    with open(file_path, "w", encoding="utf8") as file:
        for instance in instances:
            file.write(json.dumps(instance) + "\n")


def get_retriever_url(retriever_type: str) -> str:
    """Return the /retrieve/ URL of the "bm25" or "dense" server from .retriever_address.jsonnet."""
    if not os.path.exists(RETRIEVER_ADDRESS_FILE):
        raise FileNotFoundError(f"Retriever address file {RETRIEVER_ADDRESS_FILE} not found.")
    address = json.loads(_jsonnet.evaluate_file(RETRIEVER_ADDRESS_FILE))
    host, port = address["host"], address["port"]
    if isinstance(host, dict):
        host = host[retriever_type]
    if isinstance(port, dict):
        port = port[retriever_type]
    return str(host).rstrip("/") + ":" + str(port) + "/retrieve/"


def find_matching_paragraph_text(corpus_name: str, original_paragraph_text: str, method: str) -> Optional[Dict]:
    """Find the corpus paragraph that matches a dataset paragraph (IRCoT's remapping step).

    Retrieves the top hit for the paragraph text and accepts it if its text contains the
    original almost verbatim (rapidfuzz partial ratio > 95). Returns {"title", "paragraph_text"}
    of the corpus paragraph, or None if the retrieval fails or nothing matches.
    """
    retriever_type = {"retrieve_from_elasticsearch": "bm25", "vector_based_similarity_search": "dense"}[method]
    params = {
        "query_text": original_paragraph_text,
        "retrieval_method": method,
        "max_hits_count": 1,
        "corpus_name": corpus_name,
    }
    result = requests.post(get_retriever_url(retriever_type), json=params, timeout=RETRIEVER_TIMEOUT)
    if not result.ok:
        print(f"WARNING: Retrieval failed ({result.status_code}). Skipping this mapping.", file=sys.stderr)
        return None

    retrieval = result.json()["retrieval"]
    if not retrieval:
        print("WARNING: Nothing retrieved. Skipping this mapping.", file=sys.stderr)
        return None
    assert retrieval[0]["corpus_name"] == corpus_name

    retrieved_paragraph_text = retrieval[0]["paragraph_text"]
    match_ratio = fuzz.partial_ratio(original_paragraph_text, retrieved_paragraph_text)
    if match_ratio > 95:
        return {"title": retrieval[0]["title"], "paragraph_text": retrieved_paragraph_text}
    print(
        f"WARNING: Couldn't map the original paragraph text to the retrieved one ({match_ratio}).\n"
        f"Original: {original_paragraph_text}\n"
        f"Retrieved: {retrieved_paragraph_text}",
        file=sys.stderr,
    )
    return None
