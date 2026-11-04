"""Write the Bamboogle test set in the processed QA format.

Reads the HuggingFace dataset chiayewken/bamboogle (split "test": 125 rows with columns
"Question" and "Answer") and writes <processed_data>/bamboogle/test.jsonl with question
ids bamboogle_1 ... bamboogle_125 in dataset order. Bamboogle has no gold paragraphs, so
`contexts` is empty; questions are answered from the corpus `wiki`.
"""

import argparse
import json
import os

from datasets import load_dataset

from common import PROCESSED_DATA_ROOT

HF_DATASET = "chiayewken/bamboogle"


def main():
    argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="The output path follows SPARC_RAG_PROCESSED_DATA_ROOT (default: processed_data).",
    ).parse_args()

    rows = load_dataset(HF_DATASET, split="test")
    output_directory = os.path.join(PROCESSED_DATA_ROOT, "bamboogle")
    os.makedirs(output_directory, exist_ok=True)
    output_file_path = os.path.join(output_directory, "test.jsonl")
    with open(output_file_path, "w", encoding="utf8") as output_file:
        for idx, row in enumerate(rows, start=1):
            instance = {
                "dataset": "bamboogle",
                "question_id": f"bamboogle_{idx}",
                "question_text": row["Question"],
                "answers_objects": [
                    {"number": "", "date": {"day": "", "month": "", "year": ""}, "spans": [row["Answer"]]}
                ],
                "contexts": [],
            }
            output_file.write(json.dumps(instance) + "\n")
    print(f"Wrote {len(rows)} questions to {output_file_path}")


if __name__ == "__main__":
    main()
