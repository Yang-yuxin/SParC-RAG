"""Merge preference-pair files into the train/test dataset read by train_dpo_asymmetric.py.

The pairs of all input files are concatenated in the order given, shuffled with ``random.Random(seed)``,
and cut at ``int(n_pairs * train_ratio)``: the first part is the train split, the rest the test split.
The order of the input files therefore changes the split. Given the 12 files behind the paper's
checkpoints in the order listed in training/README.md, the defaults (--train_ratio 0.9, --seed 42)
reproduce their split exactly (3,356 pairs -> 3,020 train / 336 test).

Every example keeps ``prompt``, ``chosen`` and ``rejected`` and gets two extra columns: ``role``
(``query_rewrite`` or ``answer_eval``, used for per-role validation metrics) and ``dataset``.

Usage:
    python training/prepare_dpo_dataset.py \
        --inputs training/data/query_rewrite/query_rewrite_dpo_*.jsonl \
                 training/data/answer_eval/answer_eval_dpo_*.jsonl \
        --output_dir training/data/dpo_joint
"""

import argparse
import collections
import json
import os
import random
import sys

from datasets import Dataset, DatasetDict


def read_pairs(path):
    """Read one preference pair per non-empty line of a JSONL file."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pair_role(pair):
    """Tell the module a pair trains from the metadata its generator writes."""
    metadata = pair.get("metadata", {})
    if "is_correct" in metadata:  # gen_dpo_data_answer_eval.py labels the evaluated answer
        return "answer_eval"
    if "dispatch_width" in metadata:  # gen_dpo_data_query_rewrite.py records the number of queries
        return "query_rewrite"
    raise ValueError(f"Cannot tell the role of the pair with metadata keys {sorted(metadata)}")


def to_example(pair):
    return {
        "prompt": pair["prompt"],
        "chosen": pair["chosen"],
        "rejected": pair["rejected"],
        "role": pair_role(pair),
        "dataset": pair["metadata"]["dataset"],
    }


def print_composition(splits):
    counts = collections.Counter(
        (split, ex["role"], ex["dataset"]) for split, examples in splits.items() for ex in examples
    )
    print(f"{'split':<6} {'role':<14} {'dataset':<16} {'pairs':>6}")
    for (split, role, dataset), n in sorted(counts.items()):
        print(f"{split:<6} {role:<14} {dataset:<16} {n:>6}")
    for split, examples in splits.items():
        print(f"{split:<6} {'total':<31} {len(examples):>6}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True, help="Pair files (JSONL), concatenated in this order.")
    parser.add_argument("--output_dir", required=True, help="Where to save the DatasetDict (must not exist).")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Fraction of pairs in the train split.")
    parser.add_argument("--seed", type=int, default=42, help="Seed of the shuffle before the split.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        sys.exit(f"Output directory already exists: {args.output_dir}")

    pairs = []
    for path in args.inputs:
        file_pairs = read_pairs(path)
        print(f"{len(file_pairs):6d} pairs  {path}")
        pairs.extend(file_pairs)

    random.Random(args.seed).shuffle(pairs)
    n_train = int(len(pairs) * args.train_ratio)
    if not 0 < n_train < len(pairs):
        sys.exit(f"--train_ratio {args.train_ratio} leaves an empty split for {len(pairs)} pairs")

    examples = [to_example(pair) for pair in pairs]
    splits = {"train": examples[:n_train], "test": examples[n_train:]}
    DatasetDict({name: Dataset.from_list(rows) for name, rows in splits.items()}).save_to_disk(args.output_dir)

    print_composition(splits)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
