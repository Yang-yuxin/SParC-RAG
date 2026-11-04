"""Build the HotpotQA passage corpus for the dense retriever.

Reads the Wikipedia abstracts in <raw_data>/hotpotqa/wikpedia-paragraphs/*/wiki_*.bz2 and
writes one passage per article to <processed_data>/hotpotqa/corpus.jsonl ({"_id", "title",
"text", "metadata"} per line), the file serve_beir.py and build_beir_index.py load. The id is
the numeric Wikipedia page id and the text is the article's sentences joined with spaces.
"""

import argparse
import bz2
import glob
import json
import os
import shutil
from multiprocessing import Pool, cpu_count
from typing import Dict, List

from tqdm import tqdm

from dataloaders import PROCESSED_DATA_ROOT, RAW_DATA_ROOT


def read_passages(file_path: str) -> List[Dict]:
    passages = []
    with bz2.BZ2File(file_path) as file:
        for line in file:
            article = json.loads(line.strip())
            passages.append(
                {
                    "_id": str(article.get("id")),
                    "title": article.get("title", ""),
                    "text": " ".join(sentence.strip() for sentence in article.get("text", [])),
                    "metadata": {},
                }
            )
    return passages


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Paths follow SPARC_RAG_RAW_DATA_ROOT and SPARC_RAG_PROCESSED_DATA_ROOT "
        "(defaults: raw_data, processed_data).",
    )
    parser.add_argument("--num_workers", type=int, default=min(cpu_count(), 8), help="Parallel workers (default: up to 8).")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing corpus.jsonl.")
    args = parser.parse_args()

    input_files = sorted(glob.glob(os.path.join(RAW_DATA_ROOT, "hotpotqa", "wikpedia-paragraphs", "*", "wiki_*.bz2")))
    if not input_files:
        parser.error(f"No wiki_*.bz2 files under {os.path.join(RAW_DATA_ROOT, 'hotpotqa', 'wikpedia-paragraphs')}.")
    output_directory = os.path.join(PROCESSED_DATA_ROOT, "hotpotqa")
    output_file = os.path.join(output_directory, "corpus.jsonl")
    if os.path.exists(output_file) and not args.force:
        parser.error(f"{output_file} already exists; pass --force to rebuild it.")
    os.makedirs(output_directory, exist_ok=True)

    passage_count = 0
    temp_file = output_file + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as out_file, Pool(processes=args.num_workers) as pool:
        for passages in tqdm(pool.imap(read_passages, input_files), total=len(input_files), desc="Files"):
            for passage in passages:
                out_file.write(json.dumps(passage) + "\n")
            passage_count += len(passages)
    shutil.move(temp_file, output_file)
    print(f"Wrote {passage_count} passages to {output_file}")


if __name__ == "__main__":
    main()
