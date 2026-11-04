"""Build the 2WikiMultiHopQA passage corpus for the dense retriever.

Collects the context paragraphs of <raw_data>/2wikimultihopqa/{train,dev,test}.json in file
order, drops exact (title, text) duplicates and writes them with sequential ids "0", "1", ...
to <processed_data>/2wikimultihopqa/corpus.jsonl ({"_id", "title", "text", "metadata"} per
line), the file serve_beir.py and build_beir_index.py load.
"""

import argparse
import json
import os
import shutil

from tqdm import tqdm

from dataloaders import PROCESSED_DATA_ROOT, RAW_DATA_ROOT


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Paths follow SPARC_RAG_RAW_DATA_ROOT and SPARC_RAG_PROCESSED_DATA_ROOT "
        "(defaults: raw_data, processed_data).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing corpus.jsonl.")
    args = parser.parse_args()

    raw_directory = os.path.join(RAW_DATA_ROOT, "2wikimultihopqa")
    output_directory = os.path.join(PROCESSED_DATA_ROOT, "2wikimultihopqa")
    output_file = os.path.join(output_directory, "corpus.jsonl")
    if os.path.exists(output_file) and not args.force:
        parser.error(f"{output_file} already exists; pass --force to rebuild it.")
    os.makedirs(output_directory, exist_ok=True)

    seen = set()
    temp_file = output_file + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as out_file:
        for split in ("train", "dev", "test"):
            with open(os.path.join(raw_directory, f"{split}.json"), encoding="utf-8") as file:
                instances = json.load(file)
            for instance in tqdm(instances, desc=f"Processing {split}.json"):
                for title, sentences in instance["context"]:
                    paragraph_text = " ".join(sentences)
                    key = f"{title}\n{paragraph_text}"
                    if key in seen:
                        continue
                    passage = {"_id": str(len(seen)), "title": title, "text": paragraph_text, "metadata": {}}
                    seen.add(key)
                    out_file.write(json.dumps(passage) + "\n")
    shutil.move(temp_file, output_file)
    print(f"Wrote {len(seen)} passages to {output_file}")


if __name__ == "__main__":
    main()
