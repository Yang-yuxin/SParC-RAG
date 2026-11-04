"""Data carried by a search state: the question and the sequence of inference steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class InferenceStep:
    participant: str
    # Completion (output) tokens of the LLM call that produced this step; 0 for steps that do
    # not call an LLM.
    token_size: int = 0

    def get_name(self):
        return self.__class__.__name__


@dataclass(kw_only=True)
class RetrievalStep(InferenceStep):
    titles: list
    paras: list


@dataclass(kw_only=True)
class QuestionAnsweringStep(InferenceStep):
    answer: str
    reasoning: str = None
    prompt: str = None


@dataclass(kw_only=True)
class AnswerEvaluationStep(InferenceStep):
    answer: str
    evaluation: str
    reasoning: str = None
    prompt: str = None
    context: str = None


@dataclass(kw_only=True)
class QuestionRewritingStep(InferenceStep):
    input_question: str
    output_question: str
    reasoning: str = None
    strategy: str = None
    prompt: str = None
    dispatch_output: str = None
    # Branches of a dispatch (W >= 2) share one LLM call: token_size holds this branch's share of its
    # completion tokens, dispatch_token_size all of them.
    dispatch_token_size: int = None


@dataclass(kw_only=True)
class ContextUpdateStep(InferenceStep):
    old_context: str
    new_context: str = None
    prompt: str = None


@dataclass(kw_only=True)
class DispatchStep(InferenceStep):
    size: int
    next: str


@dataclass(kw_only=True)
class DispatchCombineStep(InferenceStep):
    choices: list
    combined_answer: Any
    reason: str
    prompt: str = None


class StructuredDataInstance(dict):
    """The data of one question: the reader's fields plus the ``inference_seq`` of steps."""

    def __init__(self, input_data):
        super().__init__(input_data)
        self.setdefault("inference_seq", [])

    def add_answer(self, qastep: QuestionAnsweringStep):
        self["inference_seq"].append(qastep)

    def get_last_step(self):
        return self["inference_seq"][-1]

    def get_last_answer(self, reasoning: bool = False):
        for step in reversed(self["inference_seq"]):
            if isinstance(step, QuestionAnsweringStep):
                if reasoning:
                    return (step.answer, step.reasoning)
                return step.answer
        raise ValueError("No last answer! No answer generated yet")

    def get_last_query(self):
        for step in reversed(self["inference_seq"]):
            if isinstance(step, QuestionRewritingStep):
                return step.output_question
        return None

    def get_structured_path(self):
        """Return the inference sequence as JSON-serializable data.

        Each step becomes ``{<step class name>: {<attribute>: <value>, ...}}``.
        """

        def serialize_step(step):
            if isinstance(step, DispatchCombineStep):
                return {
                    "DispatchCombineStep": {
                        "choices": [serialize_step(choice) for choice in step.choices],
                        "combined_answer": serialize_step(step.combined_answer),
                        "reason": step.reason,
                        "participant": step.participant,
                    }
                }
            if isinstance(step, InferenceStep):
                attr_dict = {}
                for k, v in step.__dict__.items():
                    if isinstance(v, InferenceStep):
                        attr_dict[k] = serialize_step(v)
                    elif isinstance(v, list):
                        attr_dict[k] = [
                            serialize_step(i) if isinstance(i, InferenceStep) else i
                            for i in v
                        ]
                    else:
                        attr_dict[k] = v
                return {step.get_name(): attr_dict}
            return step

        return [serialize_step(step) for step in self["inference_seq"]]

    def get_total_token_count(self) -> int:
        """Sum ``token_size`` over the steps of this path (``token_count`` in the predictions).

        That is the completion tokens of the LLM calls on the selected path, with this branch's
        share of each Query Rewriter call that dispatched branches.
        """
        return sum(step.token_size for step in self["inference_seq"])
