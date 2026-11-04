"""Convert the DPR Natural Questions splits into the processed QA format.

Reads  <raw_data>/nq/biencoder-nq-{train,dev}.json   (download/raw_data.sh)
Writes <processed_data>/nq/{train,dev}.jsonl

Each question keeps all of its DPR positive passages as supporting contexts and adds 5
random negative and 5 random hard-negative passages. All passages come from the DPR
Wikipedia passages, i.e. from the corpus `wiki`. Train is processed before dev because
both draw from one random stream (seed 13370); this order reproduces the files used in
the paper. The NQ test set is a 500-question sample of dev.jsonl, drawn with
`subsample_dataset_and_remap_paras.py nq test 500`.

Loading biencoder-nq-train.json (7.4 GB) needs about 18 GB of RAM.
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List

from tqdm import tqdm

from common import PROCESSED_DATA_ROOT, RAW_DATA_ROOT, read_json


def safe_sample(items: List[Any], count: int) -> List[Any]:
    count = min(count, len(items))
    return random.sample(items, count) if count > 0 else []


def make_contexts(passages: List[Dict], is_supporting: bool, first_idx: int) -> List[Dict]:
    return [
        {
            "idx": first_idx + offset,
            "title": passage["title"].strip(),
            "paragraph_text": passage["text"].strip(),
            "is_supporting": is_supporting,
        }
        for offset, passage in enumerate(passages)
    ]


def write_nq_instances(raw_instances: List[Dict], output_path: str, set_name: str):
    print(f"Writing {len(raw_instances)} questions to {output_path}")
    with open(output_path, "w", encoding="utf8") as output_file:
        for idx, raw_instance in enumerate(tqdm(raw_instances)):
            contexts = make_contexts(raw_instance["positive_ctxs"], True, 0)
            contexts += make_contexts(safe_sample(raw_instance["negative_ctxs"], 5), False, len(contexts))
            contexts += make_contexts(safe_sample(raw_instance["hard_negative_ctxs"], 5), False, len(contexts))
            processed_instance = {
                "dataset": "nq",
                "question_id": f"single_nq_{set_name}_{idx}",
                "question_text": raw_instance["question"],
                "answers_objects": [
                    {"number": "", "date": {"day": "", "month": "", "year": ""}, "spans": raw_instance["answers"]}
                ],
                "contexts": contexts,
            }
            output_file.write(json.dumps(processed_instance) + "\n")


def main():
    argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Paths follow SPARC_RAG_RAW_DATA_ROOT and SPARC_RAG_PROCESSED_DATA_ROOT "
        "(defaults: raw_data, processed_data).",
    ).parse_args()

    output_directory = os.path.join(PROCESSED_DATA_ROOT, "nq")
    os.makedirs(output_directory, exist_ok=True)

    random.seed(13370)  # determines which negative passages are sampled
    for set_name in ("train", "dev"):
        raw_instances = read_json(os.path.join(RAW_DATA_ROOT, "nq", f"biencoder-nq-{set_name}.json"))
        write_nq_instances(raw_instances, os.path.join(output_directory, f"{set_name}.jsonl"), set_name)
        del raw_instances


if __name__ == "__main__":
    main()
