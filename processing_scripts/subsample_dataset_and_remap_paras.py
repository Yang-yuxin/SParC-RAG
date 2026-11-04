"""Sample evaluation questions from <processed_data>/<dataset>/dev.jsonl (IRCoT's protocol).

  test           500 questions, excluding dev_subsampled.jsonl if it exists -> test_subsampled.jsonl
  dev            500 questions                                             -> dev_subsampled.jsonl
  dev_diff_size  SAMPLE_SIZE questions not in test_subsampled.jsonl        -> dev_<SAMPLE_SIZE>_subsampled.jsonl

The paper's NQ test set is `nq test 500`, drawn while no NQ dev_subsampled.jsonl exists. The
test and dev samples of HotpotQA, 2WikiMultiHopQA and MuSiQue come with download/processed_data.sh.

For `test` and `dev` of the multi-hop datasets, every context paragraph is replaced by its
match in the dataset's retrieval corpus, as IRCoT did; this needs the matching retriever
server (.retriever_address.jsonnet). NQ contexts are DPR passages, i.e. already passages of
the corpus `wiki`, and dev_diff_size samples are written as drawn, so these two need no
retriever. This reproduces the NQ test set and the dev_500 samples used in the paper.
"""

import argparse
import os
import random

from tqdm import tqdm

from common import PROCESSED_DATA_ROOT, find_matching_paragraph_text, read_jsonl, write_jsonl

DATASETS = ("hotpotqa", "2wikimultihopqa", "musique", "nq")
DEV_TEST_SAMPLE_SIZE = 500


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog=__doc__.split("\n\n", 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset_name", choices=DATASETS)
    parser.add_argument("set_name", choices=("dev", "test", "dev_diff_size"))
    parser.add_argument("sample_size", type=int, help="Number of questions (must be 500 for dev and test).")
    parser.add_argument(
        "--method",
        default="retrieve_from_elasticsearch",
        choices=("retrieve_from_elasticsearch", "vector_based_similarity_search"),
        help="Retriever used to remap context paragraphs (default: BM25, as in IRCoT).",
    )
    args = parser.parse_args()
    if args.set_name != "dev_diff_size" and args.sample_size != DEV_TEST_SAMPLE_SIZE:
        parser.error(f"{args.set_name} samples always have {DEV_TEST_SAMPLE_SIZE} questions.")

    dataset_directory = os.path.join(PROCESSED_DATA_ROOT, args.dataset_name)
    avoid_file_path = None
    if args.set_name == "test":
        dev_file_path = os.path.join(dataset_directory, "dev_subsampled.jsonl")
        avoid_file_path = dev_file_path if os.path.exists(dev_file_path) else None
        output_file_name = "test_subsampled.jsonl"
    elif args.set_name == "dev":
        output_file_name = "dev_subsampled.jsonl"
    else:
        avoid_file_path = os.path.join(dataset_directory, "test_subsampled.jsonl")
        output_file_name = f"dev_{args.sample_size}_subsampled.jsonl"

    instances = read_jsonl(os.path.join(dataset_directory, "dev.jsonl"))
    if avoid_file_path:
        avoid_ids = {instance["question_id"] for instance in read_jsonl(avoid_file_path)}
        instances = [instance for instance in instances if instance["question_id"] not in avoid_ids]

    random.seed(13370)  # determines which questions are sampled
    instances = random.sample(instances, args.sample_size)

    if args.set_name != "dev_diff_size" and args.dataset_name != "nq":
        for instance in tqdm(instances, desc="Remapping context paragraphs"):
            for context in instance["contexts"]:
                matched = find_matching_paragraph_text(args.dataset_name, context["paragraph_text"], args.method)
                if matched is None:
                    continue
                context["title"] = matched["title"]
                context["paragraph_text"] = matched["paragraph_text"]

    output_file_path = os.path.join(dataset_directory, output_file_name)
    write_jsonl(instances, output_file_path)
    print(f"Wrote {len(instances)} questions to {output_file_path}")


if __name__ == "__main__":
    main()
