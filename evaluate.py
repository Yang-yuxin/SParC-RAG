#!/usr/bin/env python3
"""Score SPARC-RAG predictions.

Reads predictions/<trial_name>/<run_id>/prediction__<eval-file-stem>[_all].json written by
predict.py (falling back to the *_temp files of an unfinished run), loads the ground truth from
--eval_path and reports
  - answer EM, F1, precision and recall (DROP evaluator) and accuracy (an answer string occurs
    in the prediction), and
  - supporting-paragraph EM/F1/precision/recall of the paragraphs retrieved on all branches
    (title_* for exact title match, para_* for fuzzy paragraph match).
Writes evaluation_metrics__<eval-file-stem>.json and ground_truth__<eval-file-stem>.json next
to the predictions.
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict

from tqdm import tqdm

from inference.dataset_readers import MultiParaRCReader
from metrics.drop_answer_em_f1 import DropAnswerEmAndF1
from metrics.support_em_f1 import SupportEmF1Metric


def answer_extractor(potentially_cot: str) -> str:
    """Return the text after "answer is:" if the prediction is a chain of thought.

    Applied to every prediction before scoring; it strips surrounding double quotes too.
    """
    if potentially_cot.startswith('"') and potentially_cot.endswith('"'):
        potentially_cot = potentially_cot[1:-1]

    cot_regex = re.compile(".* answer is:? (.*)\\.?")
    match = cot_regex.match(potentially_cot)
    if match:
        output = match.group(1)
        if output.endswith("."):
            output = output[:-1]
    else:
        output = potentially_cot

    return output


def evaluate_by_dicts(
    id_to_ground_truths: Dict[str, Any],
    id_to_gold_docs: Dict[str, Any],
    id_to_predictions: Dict[str, Any],
    id_to_retrieved_docs: Dict[str, Any],
    save_all: bool = False,
) -> Dict:
    answer_metric = DropAnswerEmAndF1(save_all=save_all)
    support_metric = SupportEmF1Metric(do_normalize_answer=True, save_all=save_all)

    for id_ in tqdm(id_to_ground_truths.keys(), desc="Evaluating predictions"):
        ground_truth = id_to_ground_truths[id_]
        prediction = id_to_predictions[id_]
        if id_ not in id_to_retrieved_docs:
            print(f"WARNING: No retrieved docs found for id {id_}.", file=sys.stderr)
        retrieved_titles_and_docs = id_to_retrieved_docs.get(id_, [])
        gold_titles_and_docs = id_to_gold_docs.get(id_)

        assert isinstance(prediction, (str, list))
        if isinstance(prediction, str):
            # A prediction that looks like a list ("[a, b]") is scored as several answer spans.
            if prediction.strip().startswith("[") or prediction.strip().endswith("]"):
                prediction = [
                    e
                    for e in prediction.replace('"', "")
                    .replace("[", "")
                    .replace("]", "")
                    .split(",")
                ]
            else:
                prediction = [prediction]
        prediction = [answer_extractor(str(e)) for e in prediction]

        # The ground truth is the tuple of answer strings of the question's answer object. The
        # DROP evaluator scores a tuple of several strings as a multi-span answer. NQ lists all
        # answer aliases in one answer object, so for NQ questions with several aliases EM
        # requires predicting every alias, and F1 is the best-matching alias' F1 divided by the
        # number of aliases.
        answer_metric(prediction, [ground_truth])
        support_metric(retrieved_titles_and_docs, gold_titles_and_docs)

    evaluation_results = answer_metric.get_metric()
    evaluation_results.update(support_metric.get_metric())
    return evaluation_results


def load_ground_truths(ground_truth_file_path: str):
    id_to_ground_truths = {}
    id_to_gold_docs = {}
    for example in MultiParaRCReader().read_examples(ground_truth_file_path):
        id_ = example["qid"]
        id_to_ground_truths[id_] = example["answer"]
        id_to_gold_docs[id_] = [
            example["metadata"]["gold_titles"],
            example["metadata"]["gold_paras"],
        ]
    return id_to_ground_truths, id_to_gold_docs


def load_json(file_path: str) -> Dict:
    with open(file_path, "r") as file:
        return json.load(file)


def _existing(path: str, temp_path: str) -> str:
    if os.path.exists(path):
        return path
    print(f"{path} not found; using {temp_path} (unfinished run).")
    return temp_path


def main():
    parser = argparse.ArgumentParser(description="Score SPARC-RAG predictions.")
    parser.add_argument("--eval_path", type=str, required=True, help="evaluation file (.jsonl)")
    parser.add_argument(
        "--trial_name",
        type=str,
        default="default",
        help="Name of the set for predictions (matches predict.py).",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Run ID written by predict.py (e.g. 20250804-170015_ab12cd).",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        default=False,
        help="Score only the questions that have predictions (for unfinished runs).",
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default="predictions",
        help="Base predictions directory (default: predictions).",
    )
    parser.add_argument(
        "--save_all", action="store_true", default=False, help="Also save per-question scores."
    )
    args = parser.parse_args()

    prediction_directory = os.path.join(args.predictions_dir, args.trial_name, args.run_id)
    prediction_file_name = os.path.splitext(os.path.basename(args.eval_path))[0]
    prefix = os.path.join(prediction_directory, "prediction__" + prediction_file_name)
    prediction_file_path = _existing(prefix + ".json", prefix + "_temp.json")
    all_file_path = _existing(prefix + "_all.json", prefix + "_temp_all.json")
    for path in (prediction_file_path, all_file_path):
        if not os.path.exists(path):
            raise SystemExit(f"{path} is not available.")

    id_to_ground_truths, id_to_gold_docs = load_ground_truths(args.eval_path)
    id_to_predictions = load_json(prediction_file_path)
    id_to_retrieved_docs = {
        id_: all_res.get("support_docs", []) for id_, all_res in load_json(all_file_path).items()
    }

    if args.truncate:
        id_to_ground_truths = {
            id_: ground_truth
            for id_, ground_truth in id_to_ground_truths.items()
            if id_ in id_to_predictions
        }
    if set(id_to_ground_truths.keys()) != set(id_to_predictions.keys()):
        raise SystemExit("Ids in input examples and predictions don't match.")

    evaluation_results = evaluate_by_dicts(
        id_to_ground_truths=id_to_ground_truths,
        id_to_gold_docs=id_to_gold_docs,
        id_to_predictions=id_to_predictions,
        id_to_retrieved_docs=id_to_retrieved_docs,
        save_all=args.save_all,
    )
    print(json.dumps(evaluation_results, indent=4))

    save_metrics_path = os.path.join(
        prediction_directory, "evaluation_metrics__" + prediction_file_name + ".json"
    )
    print(f"Saving metrics in {save_metrics_path}")
    with open(save_metrics_path, "w") as file:
        json.dump(evaluation_results, file, indent=4)

    ground_truth_in_dict_file_path = os.path.join(
        prediction_directory, "ground_truth__" + prediction_file_name + ".json"
    )
    with open(ground_truth_in_dict_file_path, "w") as file:
        json.dump(id_to_ground_truths, file, indent=4)


if __name__ == "__main__":
    main()
