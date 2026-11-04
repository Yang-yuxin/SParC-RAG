"""Controller that runs the SPARC-RAG participants as a state machine.

A search state holds the data of one question and the name of the participant to run next
(``state.next``). Whenever a state is about to run the Query Rewriter (``rewrite``), the
controller records a DispatchStep and asks the rewriter for ``width`` branches. The branches
then advance one step each per iteration, one after the other. Once every branch is back
before ``rewrite`` or has reached the end state, the ``state_combiner`` participant keeps one
of them. The search stops when the kept state reaches the end state or when its number of
answering rounds reaches ``max_length``.
"""

import copy

from inference.data_instances import (
    DispatchStep,
    QuestionAnsweringStep,
    RetrievalStep,
    StructuredDataInstance,
)

DISPATCH_AT = "rewrite"
COMBINER = "state_combiner"


def count_sequential_answering_steps(inference_seq):
    """Count answering rounds; consecutive QuestionAnsweringSteps count as one round."""
    count = 0
    in_answering_group = False
    for step in inference_seq:
        if isinstance(step, QuestionAnsweringStep):
            if not in_answering_group:
                count += 1
                in_answering_group = True
        else:
            in_answering_group = False
    return count


def _step_node_data(step, **extra):
    """Graph-log node data of an inference step: its type, participant, tokens and fields."""
    data = {
        "step_type": step.get_name(),
        "participant": step.participant,
        "token_size": step.token_size,
        **extra,
    }
    for name, value in step.__dict__.items():
        if name in ("participant", "token_size"):
            continue
        if isinstance(value, (str, int, float, bool)):
            data[name] = value
        elif isinstance(value, list):
            data[name + "_count"] = len(value)
            data[name] = value
    return data


def _branch_node_data(participant, index, branch_state):
    """Graph-log node data of a dispatched branch."""
    info = {
        "step_type": "dispatch_branch",
        "branch_index": index,
        "participant": participant,
        "dispatch_rank": index,
    }
    inference_seq = branch_state.data["inference_seq"]
    last_step = inference_seq[-1]
    last_step_info = {"type": last_step.__class__.__name__}
    for name, value in last_step.__dict__.items():
        if value is not None:
            last_step_info[name] = value
    info["last_step"] = last_step_info
    info["output_question"] = last_step.output_question
    for step in reversed(inference_seq):
        if hasattr(step, "answer") and not info.get("last_answer"):
            info["last_answer"] = step.answer
        if hasattr(step, "output_question") and not info.get("last_rewritten_query"):
            info["last_rewritten_query"] = step.output_question
    return info


class ModelController:
    """Run the participants on one question at a time (see the module docstring).

    :param participants: participant objects by name; ``next_model`` fields refer to these names.
    :param start_state: name of the first participant.
    :param end_state: value of ``state.next`` that ends the search.
    :param width: number of branches requested from the Query Rewriter (W).
    :param max_length: maximum number of answering rounds (-1: no limit).
    :param logger: optional ``logger.RunLogger`` that records the log graph of every question.
    """

    def __init__(self, participants, start_state, end_state, width, max_length=-1, logger=None):
        self.participants = participants
        self.start_state = start_state
        self.end_state = end_state
        self.width = width
        self.max_length = max_length
        self.graph_logger = logger
        self.all_retrieved_docs = set()

    def answer(self, json_input, debug=False):
        """Run the search for one question; return (qid, final_answer, final_state, support_docs).

        ``support_docs`` is [titles, ["<title> <paragraph>", ...]] over the paragraphs retrieved
        on all branches, including discarded ones, in arbitrary order.
        """
        qid = json_input["qid"]
        init_question = json_input["question"]
        if debug:
            print(f"[START QUERY] : {init_question}")
        init_state = SearchState(StructuredDataInstance(json_input), self.start_state)

        self.all_retrieved_docs = set()
        for participant in self.participants.values():
            if hasattr(participant, "reset_failure_counter"):
                participant.reset_failure_counter()

        if self.graph_logger:
            self.graph_logger.reset_log_graph()
            init_state.current_node_id = self.graph_logger.add_graph_node(
                {
                    "step_type": "initial_question",
                    "participant": "model_controller",
                    "question": init_question,
                    "qid": qid,
                },
                state=init_state,
            )

        current_state = init_state
        while True:
            if isinstance(current_state, list) and len(current_state) == 1:
                current_state = current_state[0]

            if isinstance(current_state, list):
                if all(s.next in (self.end_state, DISPATCH_AT) for s in current_state):
                    if debug:
                        print(
                            "[ModelController] All branches before selection or at the end: "
                            f"{[s.next for s in current_state]}"
                        )
                    for s in current_state:
                        self._accumulate_retrieval_docs(s)
                    current_state = self._combine(current_state)
                else:
                    # Advance every branch by one step; branches never dispatch again.
                    next_states = []
                    for s in current_state:
                        self._accumulate_retrieval_docs(s)
                        next_states.append(self.execute(s, allow_dispatch=False))
                    current_state = next_states
                continue

            if (
                self.max_length > 0
                and count_sequential_answering_steps(current_state.data["inference_seq"])
                >= self.max_length
            ):
                break
            if current_state.next == self.end_state:
                break
            current_state = self.execute(current_state)
            self._accumulate_retrieval_docs(current_state)

        final_state = current_state
        if self.all_retrieved_docs:
            titles, paras = (list(x) for x in zip(*self.all_retrieved_docs))
            support_doc = [
                titles,
                [title + " " + para for title, para in zip(titles, paras)],
            ]
        else:
            support_doc = [[], []]

        final_answer = final_state.data.get_last_answer()
        if self.graph_logger:
            self.graph_logger.save_log_graph(qid)
        return (qid, final_answer, final_state, support_doc)

    def execute(self, state, allow_dispatch=True):
        """Run the participant named by ``state.next``; return its output state(s)."""
        if state.next == self.end_state:
            return state
        if state.next not in self.participants:
            raise ValueError(
                f"No participant named {state.next!r}; check the next_model fields of the config."
            )
        participant = self.participants[state.next]
        num_steps_before = len(state.data["inference_seq"])
        if allow_dispatch and state.next == DISPATCH_AT:
            output = self._dispatch(state, participant)
        else:
            output = participant.query(state)

        if self.graph_logger and not isinstance(output, list):
            for step in output.data["inference_seq"][num_steps_before:]:
                step_node_id = self.graph_logger.add_graph_node(_step_node_data(step), state=output)
                self.graph_logger.add_graph_edge(
                    output.current_node_id, step_node_id, f"creates_{step.get_name()}"
                )
                output.current_node_id = step_node_id
        return output

    def _dispatch(self, state, participant):
        """Ask the Query Rewriter for ``width`` branches."""
        state.data["inference_seq"].append(
            DispatchStep(size=self.width, participant="model_controller", next=state.next)
        )
        dispatch_node_id = None
        if self.graph_logger:
            dispatch_node_id = self.graph_logger.add_graph_node(
                {
                    "step_type": "dispatch",
                    "participant": "model_controller",
                    "dispatch_size": self.width,
                    "next_step": state.next,
                },
                state=state,
            )
            self.graph_logger.add_graph_edge(
                state.current_node_id, dispatch_node_id, f"dispatch_size_{self.width}"
            )

        output = participant.query(state, dispatch_size=self.width)
        if self.graph_logger and isinstance(output, list):
            for index, branch_state in enumerate(output):
                branch_node_id = self.graph_logger.add_graph_node(
                    _branch_node_data(state.next, index, branch_state), state=branch_state
                )
                self.graph_logger.add_graph_edge(dispatch_node_id, branch_node_id, f"branch_{index}")
                branch_state.current_node_id = branch_node_id
        return output

    def _combine(self, states):
        """Hand the paused branches to the StateCombiner; return the kept state."""
        output = self.participants[COMBINER].query(states)
        if self.graph_logger:
            step = output.data.get_last_step()
            selection_node_id = self.graph_logger.add_graph_node(
                _step_node_data(step, input_count=len(states)), state=output
            )
            for index, s in enumerate(states):
                self.graph_logger.add_graph_edge(
                    s.current_node_id, selection_node_id, f"selection_input_{index}"
                )
            output.current_node_id = selection_node_id
        return output

    def _accumulate_retrieval_docs(self, state):
        """Add the paragraphs of every RetrievalStep of ``state`` (or of a list of states)."""
        for s in state if isinstance(state, list) else [state]:
            for step in s.data["inference_seq"]:
                if isinstance(step, RetrievalStep):
                    self.all_retrieved_docs.update(zip(step.titles, step.paras))


class SearchState:
    """A question's data plus the name of the participant to run next."""

    def __init__(self, data, next_participant):
        self.data = data
        self.next = next_participant
        self.current_node_id = None  # node of the latest step in the log graph

    def copy(self):
        """Deep copy of the data; the log-graph position is shared."""
        new_state = SearchState(copy.deepcopy(self.data), self.next)
        new_state.current_node_id = self.current_node_id
        return new_state
