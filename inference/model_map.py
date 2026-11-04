"""Participant classes by the ``name`` used in the ``models`` blocks of the configs."""

from inference.participant_agentic import (
    AnswerEvaluatorParticipant,
    AnswerGenerationParticipant,
    ContextManagerParticipant,
    NaiveConcatContextParticipant,
    QueryRewriteParticipant,
    RetrieveAndResetParagraphsParticipant,
    StateCombinerParticipant,
)

MODEL_NAME_CLASS = {
    "retrieve_and_reset_paragraphs": RetrieveAndResetParagraphsParticipant,
    "rewrite": QueryRewriteParticipant,
    "context": ContextManagerParticipant,
    "naive_concat_context": NaiveConcatContextParticipant,
    "answer_gen": AnswerGenerationParticipant,
    "answer_eval": AnswerEvaluatorParticipant,
    "state_combiner": StateCombinerParticipant,
}
