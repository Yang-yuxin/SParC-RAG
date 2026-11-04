"""Generate Answer-Evaluator preference pairs (chosen / rejected judgement) for DPO.

Input: the full traces (prediction__<split>_all.json) of SPARC-RAG rollouts on training questions, run with
the base model and models.answer_eval.always_reject=true so that every round reaches the Answer Evaluator
(see training/scripts/gen_answer_eval_pairs.sh). For each question:
  1. Pick one Answer-Evaluator step at random and rebuild its prompt with --prompt_file from the question,
     the Context-Manager note of the preceding context update and the candidate answer.
  2. Label the candidate answer correct (SQuAD F1 > 0.7 or exact match against a gold answer) or incorrect.
  3. Sample --num_generations judgements and read the verdict of each, "yes" (accept: STOP) or "no"
     (reject: CONTINUE), as the Answer Evaluator does at inference. Skip the question if all judgements agree.
  4. Let a judge LLM pick the best "yes" and the best "no" judgement. chosen = the one that matches the
     label ("yes" for a correct answer, "no" for an incorrect one), rejected = the other.

The generator LLM is served by llm_server/serve.py at $LLM_SERVER_HOST:$LLM_SERVER_PORT. By default the
judge is the same model at the same endpoint; a --judge_model starting with "gpt" is queried through the
OpenAI API instead (requires the openai package and $OPENAI_API_KEY; the paper's data used gpt-4o).

Usage (from the repository root), on the rollouts of training/scripts/gen_answer_eval_pairs.sh:
    python training/gen_dpo_data_answer_eval.py --dataset hotpotqa \
        --predictions_path predictions/dpo_rollouts_hotpotqa/<run_id>/prediction__train_dpo_rollouts_all.json \
        --ground_truth_path processed_data/hotpotqa/train_dpo_rollouts.jsonl
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime

from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from inference.participant_agentic import parse_verdict  # noqa: E402
from lib import read_jsonl  # noqa: E402
from metrics.squad_answer_em_f1 import compute_exact, compute_f1, metric_max_over_ground_truths  # noqa: E402
from models.llm_client_generator import LLMClientGenerator  # noqa: E402

GENERATION_MAX_TOKENS = 512
JUDGE_TOKENS_LIMIT = 32768
JUDGE_PROMPT = """You are evaluating multiple responses to an answer evaluation task. Select the BEST response based on:
1. Clarity and coherence of reasoning
2. Quality of explanation
3. Logical soundness of the evaluation

Original evaluation prompt:
{original_prompt}

Candidate responses:
{responses}

Which response number (1-{num_responses}) provides the BEST evaluation? Respond with ONLY the number (1-{num_responses}).
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["hotpotqa", "musique", "2wikimultihopqa"], required=True)
    parser.add_argument("--predictions_path", required=True, help="Rollout traces (prediction__<split>_all.json).")
    parser.add_argument("--ground_truth_path", required=True, help="The questions the rollouts ran on (JSONL).")
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Use a random subset of this many questions (-1: all).")
    parser.add_argument("--num_generations", type=int, default=9, help="Judgements sampled per question.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature of the judgements (1.0 produced the paper's data).")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p of the judgements (0.95 produced the paper's data).")
    parser.add_argument("--gen_model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model served at the LLM endpoint.")
    parser.add_argument("--judge_model", default=None,
                        help="Tie-breaking judge: default --gen_model at the LLM endpoint; gpt-* models go to "
                             "the OpenAI API (the paper's data used gpt-4o).")
    parser.add_argument("--model_tokens_limit", type=int, default=32768,
                        help="Token limit of the LLM client for prompt plus generated tokens (as for the Answer "
                             "Evaluator in configs/).")
    parser.add_argument("--prompt_file", default="prompts/answer_evaluator",
                        help="Answer-Evaluator prompt used to rebuild the prompts of the trace.")
    parser.add_argument("--refill_prompts", action=argparse.BooleanOptionalAction, default=True,
                        help="Rebuild the prompts with --prompt_file (otherwise use the prompts of the trace).")
    parser.add_argument("--output_dir", default="training/data/answer_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="Print the judgements of every question.")
    args = parser.parse_args()
    if args.judge_model is None:
        args.judge_model = args.gen_model
    return args


def check_llm_endpoint():
    host, port = os.environ.get("LLM_SERVER_HOST"), os.environ.get("LLM_SERVER_PORT")
    if not host or not port or not host.startswith(("http://", "https://")):
        sys.exit("Set LLM_SERVER_HOST (with http:// or https://) and LLM_SERVER_PORT, "
                 "e.g. export LLM_SERVER_HOST=http://localhost LLM_SERVER_PORT=8010")


def extract_answer_eval_steps(prediction):
    """Answer-Evaluator steps of a trace, each with the note of the preceding non-empty context update."""
    steps, context = [], ""
    for step in prediction.get("paths", []):
        if not isinstance(step, dict):
            continue
        if "ContextUpdateStep" in step:
            context = step["ContextUpdateStep"].get("new_context", "") or context
        elif "AnswerEvaluationStep" in step and "answer" in step["AnswerEvaluationStep"]:
            data = step["AnswerEvaluationStep"]
            steps.append({
                "prompt": data.get("prompt", ""),
                "answer": data.get("answer", ""),
                "evaluation": data.get("evaluation", ""),
                "context": context,
            })
    return steps


def refill_prompt(step, template):
    """Rebuild an Answer-Evaluator prompt: question parsed from the trace's prompt, note, candidate answer."""
    match = re.search(r"- Question:\s*(.+?)(?:\n-|\n\n)", step["prompt"], re.DOTALL)
    question = match.group(1).strip() if match else ""
    return template.format(query=question, context=step["context"], answer=step["answer"])


def is_answer_correct(answer, ground_truth):
    """Correct if SQuAD F1 > 0.7 or exact match against any gold answer span."""
    gold = [span for obj in ground_truth.get("answers_objects", []) for span in obj.get("spans", []) if span]
    if not gold:
        return False
    exact = metric_max_over_ground_truths(compute_exact, answer, gold)
    f1 = metric_max_over_ground_truths(compute_f1, answer, gold)
    return f1 > 0.7 or exact == 1.0


def sample_judgements(llm, prompt, num_generations):
    """Sample judgements one request at a time (stop at [END]); a failed request raises."""
    outputs = []
    for _ in range(num_generations):
        output = llm.generate_text_sequence(prompt, stop=["[END]"])[0][0][0].strip()
        outputs.append({"output": output, "evaluation": parse_verdict(output)[0]})
    return outputs


class Judge:
    """Pick the best of several judgements with an LLM; fall back to a random pick if that fails."""

    def __init__(self, model_name, model_tokens_limit):
        self.model_name = model_name
        self.fallbacks = Counter()
        if model_name.lower().startswith("gpt"):
            from openai import OpenAI  # optional dependency, only needed for an OpenAI judge

            if not os.environ.get("OPENAI_API_KEY"):
                sys.exit(f"--judge_model {model_name} needs OPENAI_API_KEY (and optionally OPENAI_BASE_URL)")
            self.openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_BASE_URL"))
        else:
            self.openai = None
            self.llm = LLMClientGenerator(model_name=model_name, temperature=0.0,
                                          max_length=GENERATION_MAX_TOKENS, model_tokens_limit=model_tokens_limit)

    def ask(self, prompt):
        if self.openai is not None:
            completion = self.openai.chat.completions.create(
                model=self.model_name, temperature=0.0, max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            return completion.choices[0].message.content.strip()
        return self.llm.generate_text_sequence(prompt, stop=None)[0][0][0].strip()

    def best(self, evaluation_prompt, candidates):
        if len(candidates) == 1:
            return candidates[0]["output"]
        responses = "".join(f"\n--- Response {i} ---\n{c['output']}\n" for i, c in enumerate(candidates, 1))
        prompt = JUDGE_PROMPT.format(original_prompt=evaluation_prompt, responses=responses,
                                     num_responses=len(candidates))
        try:
            numbers = re.findall(r"\d+", self.ask(prompt).split("\n")[0])
        except Exception as e:
            print(f"WARNING: judge request failed ({e}); picking at random", file=sys.stderr)
            self.fallbacks["request failed"] += 1
            return random.choice(candidates)["output"]
        if numbers and 1 <= int(numbers[0]) <= len(candidates):
            return candidates[int(numbers[0]) - 1]["output"]
        self.fallbacks["unparseable answer"] += 1
        return random.choice(candidates)["output"]


def main():
    args = parse_args()
    check_llm_endpoint()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    llm = LLMClientGenerator(model_name=args.gen_model, temperature=args.temperature, top_p=args.top_p,
                             max_length=GENERATION_MAX_TOKENS, model_tokens_limit=args.model_tokens_limit)
    judge = Judge(args.judge_model, JUDGE_TOKENS_LIMIT)
    with open(args.prompt_file, encoding="utf-8") as f:
        template = f.read()

    ground_truth = {item["question_id"]: item for item in read_jsonl(args.ground_truth_path)}
    with open(args.predictions_path) as f:
        predictions = json.load(f)
    print(f"{len(predictions)} traces, {len(ground_truth)} questions with gold answers")

    instances, skipped_empty_context = [], 0
    for qid, prediction in predictions.items():
        steps = extract_answer_eval_steps(prediction)
        if not steps or qid not in ground_truth:
            continue
        step = random.choice(steps)
        if args.refill_prompts:
            if not step["context"]:
                skipped_empty_context += 1
                continue
            step["prompt"] = refill_prompt(step, template)
        instances.append({
            "qid": qid,
            "prompt": step["prompt"],
            "answer": step["answer"],
            "original_evaluation": step["evaluation"],
            "is_correct": is_answer_correct(step["answer"], ground_truth[qid]),
            "dataset": args.dataset,
        })
    print(f"{len(instances)} Answer-Evaluator steps (one per question); "
          f"{skipped_empty_context} skipped because the note was empty")
    if 0 < args.num_samples < len(instances):
        instances = random.sample(instances, args.num_samples)

    pairs, logs = [], []
    stats = Counter()
    for instance in tqdm(instances, desc="Generating pairs"):
        if not instance["prompt"]:
            stats["empty prompt"] += 1
            continue
        try:
            generations = sample_judgements(llm, instance["prompt"], args.num_generations)
        except Exception as e:
            print(f"Error on question {instance['qid']}: {e}")
            stats["generation failed"] += 1
            continue
        accepts = [g for g in generations if g["evaluation"] == "yes"]
        rejects = [g for g in generations if g["evaluation"] == "no"]
        if args.debug:
            print(f"[{instance['qid']}] answer={instance['answer']!r} correct={instance['is_correct']} "
                  f"yes={len(accepts)} no={len(rejects)}")
        if not accepts or not rejects:
            stats["all judgements agree"] += 1
            continue
        best_accept = judge.best(instance["prompt"], accepts)
        best_reject = judge.best(instance["prompt"], rejects)
        chosen, rejected = (best_accept, best_reject) if instance["is_correct"] else (best_reject, best_accept)
        pairs.append({
            "prompt": instance["prompt"],
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "qid": instance["qid"],
                "dataset": instance["dataset"],
                "answer": instance["answer"],
                "is_correct": instance["is_correct"],
                "num_generations": len(generations),
                "num_accepts": len(accepts),
                "num_rejects": len(rejects),
                "accept_ratio": len(accepts) / len(generations),
            },
        })
        logs.append({key: instance[key] for key in ("qid", "prompt", "answer", "is_correct")}
                    | {"generations": generations, "chosen": chosen, "rejected": rejected})

    print(f"{len(pairs)} pairs from {len(instances)} questions; skipped: {dict(stats)}; "
          f"judge fallbacks to a random pick: {dict(judge.fallbacks)}")
    if not pairs:
        sys.exit("No pairs generated")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{args.dataset}_{len(instances)}samples_{timestamp}"
    with open(os.path.join(args.output_dir, f"answer_eval_dpo_{stem}.jsonl"), "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    with open(os.path.join(args.output_dir, f"answer_eval_generation_log_{stem}.json"), "w") as f:
        json.dump(logs, f, indent=2)
    with open(os.path.join(args.output_dir, f"answer_eval_metadata_{stem}.json"), "w") as f:
        json.dump({
            "args": vars(args),
            "num_eval_instances": len(instances),
            "num_dpo_instances": len(pairs),
            "skipped": dict(stats),
            "judge_fallbacks": dict(judge.fallbacks),
            "truncated_prompts": llm.num_truncated_prompts,
            "correct_answers": sum(instance["is_correct"] for instance in instances),
            "incorrect_answers": sum(not instance["is_correct"] for instance in instances),
            "timestamp": timestamp,
        }, f, indent=2)
    print(f"Saved to {os.path.join(args.output_dir, f'answer_eval_dpo_{stem}.jsonl')}")


if __name__ == "__main__":
    main()
