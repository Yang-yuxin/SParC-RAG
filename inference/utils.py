"""Text helpers shared by the participants."""

from typing import List

from rapidfuzz import fuzz

from inference.data_instances import (
    AnswerEvaluationStep,
    QuestionAnsweringStep,
    QuestionRewritingStep,
    RetrievalStep,
)

WH_WORDS = {"who", "what", "when", "where", "why", "which", "how", "does", "is"}


def get_history_representation(
    state,
    use_rewritten_question: bool = False,
    use_retrieved_documents: bool = False,
    use_generated_answer: bool = False,
    use_answer_evaluation: bool = False,
    use_context: bool = False,
    use_last_only: bool = False,
    wrapped_in=("[", "]"),
    max_para_num_words: int = 600,
) -> str:
    """Render the steps of ``state`` as text for a prompt.

    The selected kinds of steps are listed oldest first, one per line, each prefixed with its
    index in the inference sequence. With ``use_last_only`` only the most recent step of each
    kind is listed. With ``use_context`` the current note is prepended as "Historical context:".
    ``wrapped_in`` gives the opening and closing brackets (``None`` for none).
    """
    history = wrapped_in[0] if wrapped_in else ""
    if use_context and state.data.get("context"):
        history += "Historical context:" + state.data["context"]

    entries = []
    used = set()
    for i, step in reversed(list(enumerate(state.data["inference_seq"]))):
        if isinstance(step, RetrievalStep):
            if use_retrieved_documents and not (
                use_last_only and "retrieved_documents" in used
            ):
                documents = "\n\n".join(
                    para_to_text(title, para, max_para_num_words)
                    for title, para in zip(step.titles, step.paras)
                )
                entries.append(f"Step {i} Retrieved documents:" + documents + "\n")
                used.add("retrieved_documents")
        if isinstance(step, QuestionAnsweringStep) and not (
            use_last_only and "generated_answer" in used
        ):
            if use_generated_answer:
                entries.append(
                    f"Step {i} Generated answer:"
                    + step.answer
                    + "\n"
                    + "Reason:"
                    + step.reasoning
                    + "\n"
                )
            used.add("generated_answer")
        if isinstance(step, QuestionRewritingStep):
            if use_rewritten_question and not (
                use_last_only and "rewritten_question" in used
            ):
                entries.append(
                    f"Step {i} Rewritten question:" + step.output_question + "\n"
                )
                used.add("rewritten_question")
        if isinstance(step, AnswerEvaluationStep):
            if use_answer_evaluation and not (
                use_last_only and "answer_evaluation" in used
            ):
                entries.append(
                    f"Step {i} Answer evaluation: {step.evaluation}\n Reason: {step.reasoning}\n"
                )
                used.add("answer_evaluation")

    for entry in reversed(entries):
        history += entry.strip() + "\n"

    if wrapped_in:
        history += " " + wrapped_in[1]

    return history


def remove_wh_words(text: str) -> str:
    """Drop question words (and "does", "is") from a retrieval query."""
    return " ".join(
        word for word in text.split(" ") if word.strip().lower() not in WH_WORDS
    )


def is_para_closely_matching(
    existing_titles: List[str],
    existing_paras: List[str],
    new_title: str,
    new_para: str,
    match_threshold: float = 90,
) -> bool:
    """Whether (new_title, new_para) duplicates a selected paragraph (fuzzy ratio >= threshold)."""
    if new_title in existing_titles and new_para in existing_paras:
        return True

    for existing_title, existing_para in zip(existing_titles, existing_paras):
        condition_1 = fuzz.ratio(existing_title, new_title) >= match_threshold
        condition_2 = fuzz.ratio(existing_para, new_para) >= match_threshold
        if condition_1 and condition_2:
            return True
    return False


def para_to_text(title: str, para: str, max_num_words: int) -> str:
    """Format a paragraph as "Wikipedia Title: <title>\\n<text>", keeping its first words."""
    # Split on single spaces (not split()) so that the paragraph's newlines are preserved.
    para = " ".join(para.split(" ")[:max_num_words])
    para = (
        para.strip()
        if para.strip().startswith("Wikipedia Title: ")
        else "Wikipedia Title: " + title + "\n" + para.strip()
    )
    return para
