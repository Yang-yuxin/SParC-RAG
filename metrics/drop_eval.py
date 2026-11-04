"""Vendored from the official DROP evaluation script (AllenNLP, allennlp/tools/drop_eval.py).

Only ``get_metrics`` and its helpers are kept; ``get_metrics`` also returns precision and recall.
"""

from typing import List, Set, Tuple, Union
import string
import re

import numpy as np
from scipy.optimize import linear_sum_assignment


# From here through _normalize_answer was originally copied from:
# https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
# Then cleaned up and modified a bit.
def _remove_articles(text: str) -> str:
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)


def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


EXCLUDE = set(string.punctuation)


def _remove_punc(text: str) -> str:
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text


def _lower(text: str) -> str:
    return text.lower()


def _tokenize(text: str) -> List[str]:
    return re.split(" |-", text)


def _normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    parts = [
        _white_space_fix(_remove_articles(_normalize_number(_remove_punc(_lower(token))))) for token in _tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    normalized = " ".join(parts).strip()
    return normalized


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    else:
        return text


def _answer_to_bags(answer: Union[str, List[str], Tuple[str, ...]]) -> Tuple[List[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> Tuple[List[float], List[float], List[float]]:
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
    f1s = np.zeros([len(gold), len(predicted)])
    precs = np.zeros([len(gold), len(predicted)])
    recalls = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                f1_prec_recall = _compute_f1(pred_item, gold_item)
                f1s[gold_index, pred_index] = f1_prec_recall[0]
                precs[gold_index, pred_index] = f1_prec_recall[1]
                recalls[gold_index, pred_index] = f1_prec_recall[2]
    row_ind, col_ind = linear_sum_assignment(-f1s)

    max_f1s = np.zeros([max(len(gold), len(predicted))])
    max_precs = np.zeros([max(len(gold), len(predicted))])
    max_recalls = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_f1s[row] = max(max_f1s[row], f1s[row, column])
        max_precs[row] = max(max_precs[row], precs[row, column])
        max_recalls[row] = max(max_recalls[row], recalls[row, column])
    return (max_f1s, max_precs, max_recalls)


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> Tuple[float, float, float]:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (2 * precision * recall) / (precision + recall) if not (precision == 0.0 and recall == 0.0) else 0.0
    return (f1, precision, recall)


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def get_metrics(
    predicted: Union[str, List[str], Tuple[str, ...]], gold: Union[str, List[str], Tuple[str, ...]]
) -> Tuple[float, float, float, float]:
    """
    Takes a predicted answer and a gold answer (that are both either a string or a list of
    strings), and returns exact match, the DROP F1 metric, precision and recall for the prediction.
    """
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)

    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        exact_match = 1.0
    else:
        exact_match = 0.0

    f1_per_bag, prec_per_bag, recall_per_bag = _align_bags(predicted_bags[1], gold_bags[1])

    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)

    prec = np.mean(prec_per_bag)
    prec = round(prec, 2)

    recall = np.mean(recall_per_bag)
    recall = round(recall, 2)

    return exact_match, f1, prec, recall
