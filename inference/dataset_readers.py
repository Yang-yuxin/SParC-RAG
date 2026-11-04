"""Reader for the processed question files (one JSON object per line)."""

import json
from collections import Counter

from tqdm import tqdm


def format_drop_answer(answer_json):
    """Return the answer strings of a DROP-style answer object (number, spans or date)."""
    if answer_json["number"]:
        return [answer_json["number"]]
    if len(answer_json["spans"]):
        return answer_json["spans"]
    date_json = answer_json["date"]
    if not (date_json["day"] or date_json["month"] or date_json["year"]):
        raise ValueError(f"Number, spans or date not set in {answer_json}")
    return [date_json["day"] + "-" + date_json["month"] + "-" + date_json["year"]]


class MultiParaRCReader:
    """Read processed QA files into the inputs of the controller and of evaluate.py.

    ``answer`` is the tuple of answer strings of the question's most common answer object, and
    ``metadata`` holds the gold supporting paragraphs used for the retrieval metrics. The
    paragraph lists start empty: every run starts with retrieval.
    """

    def read_examples(self, file):
        with open(file, "r") as input_fp:
            for line in tqdm(input_fp):
                if not line.strip():
                    continue

                input_instance = json.loads(line)
                question = input_instance["question_text"]
                formatted_answers = [
                    tuple(format_drop_answer(answers_object))
                    for answers_object in input_instance["answers_objects"]
                ]
                answer = Counter(formatted_answers).most_common()[0][0]

                gold_paragraphs = [
                    paragraph
                    for paragraph in input_instance["contexts"]
                    if paragraph["is_supporting"]
                ]
                metadata = {
                    "level": input_instance.get("level", None),
                    "type": input_instance.get("type", None),
                    "answer_type": input_instance.get("answer_type", None),
                    "simplified_answer_type": input_instance.get(
                        "simplified_answer_type", None
                    ),
                    "gold_titles": [paragraph["title"] for paragraph in gold_paragraphs],
                    "gold_paras": [
                        paragraph["paragraph_text"] for paragraph in gold_paragraphs
                    ],
                    "gold_ids": [paragraph.get("id", None) for paragraph in gold_paragraphs],
                }

                yield {
                    "qid": input_instance["question_id"],
                    "query": question,
                    "answer": answer,
                    "question": question,
                    "titles": [],
                    "paras": [],
                    "metadata": metadata,
                }
