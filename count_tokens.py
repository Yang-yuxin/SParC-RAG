#!/usr/bin/env python3
"""Token usage of SPARC-RAG runs.

Reads the log graph written by ``predict.py`` (``log_graph__<eval-file-stem>.json`` in a run directory)
and reports the average number of tokens per question: prompt plus completion tokens, summed over every
LLM call made for the question, with a breakdown by agent.

* Prompt tokens are counted with tiktoken's ``cl100k_base`` encoding on the filled prompt.
* Completion tokens are the counts returned by the LLM server.

Example::

    python count_tokens.py predictions/my_trial/my_run_id
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import tiktoken

ENCODING = tiktoken.get_encoding("cl100k_base")
# Reasoning that AnswerEvaluatorParticipant (inference/participant_agentic.py) stores for an answer
# rejected under a fixed depth without an LLM call (its filled prompt is stored all the same).
AUTO_REJECTED_REASONING = "Auto-rejected"
AGENT_NAMES = {
    "rewrite": "Query Rewriter",
    "context": "Context Manager",
    "answer_gen": "Answer Generator",
    "answer_eval": "Answer Evaluator",
    "state_combiner": "Branch selection",
}


def _num_tokens(text):
    return len(ENCODING.encode(text)) if text else 0


def _is_auto_rejected(data):
    """Whether an Answer Evaluator step rejected the answer without an LLM call."""
    return not data.get("token_size") and str(data.get("reasoning") or "").startswith(AUTO_REJECTED_REASONING)


def _dispatches(graph):
    """Return the branch nodes of every dispatch of a question, one list per dispatch."""
    nodes = {node["id"]: node for node in graph["nodes"]}
    branches = defaultdict(list)
    for edge in graph["edges"]:
        source, target = nodes[edge["from"]], nodes[edge["to"]]
        if source["step_type"] == "dispatch" and target["step_type"] == "dispatch_branch":
            branches[edge["from"]].append(target)
    return list(branches.values())


def count_question(graph):
    """Return a dict mapping each agent to the tokens of its LLM calls for one question."""
    per_agent = defaultdict(int)
    for node in graph["nodes"]:
        if node["step_type"] == "dispatch_branch":
            continue  # counted once per dispatch below
        data = node["data"]
        agent = node["participant"]
        if agent == "answer_eval" and _is_auto_rejected(data):
            continue
        per_agent[agent] += _num_tokens(data.get("prompt")) + (data.get("token_size") or 0)

    for branch_nodes in _dispatches(graph):
        step = branch_nodes[0]["data"]["last_step"]
        per_agent[branch_nodes[0]["participant"]] += _num_tokens(step["prompt"]) + step["dispatch_token_size"]
    return per_agent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prediction_dir", help="Run directory written by predict.py (contains log_graph__*.json).")
    args = parser.parse_args()

    log_files = sorted(glob.glob(os.path.join(args.prediction_dir, "log_graph__*.json")))
    if not log_files:
        raise SystemExit(f"No log_graph__*.json found in {args.prediction_dir}")

    for log_file in log_files:
        with open(log_file) as handle:
            log_graph = json.load(handle)
        totals = defaultdict(int)
        for graph in log_graph.values():
            for agent, tokens in count_question(graph).items():
                totals[agent] += tokens
        num_questions = len(log_graph)
        print(f"{os.path.basename(log_file)}: {num_questions} questions, "
              f"{sum(totals.values()) / num_questions / 1000:.1f}K tokens per question")
        for agent, tokens in sorted(totals.items(), key=lambda item: -item[1]):
            if tokens:
                print(f"  {AGENT_NAMES.get(agent, agent):<20s} {tokens / num_questions / 1000:6.2f}K")


if __name__ == "__main__":
    main()
