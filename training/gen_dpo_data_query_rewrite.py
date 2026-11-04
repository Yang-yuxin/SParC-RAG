"""Generate Query-Rewriter preference pairs (chosen / rejected rewrite) for DPO.

For each training question, in a seeded random order:
  1. Retrieve the top-k paragraphs for the question with the dense retriever. Skip the question if they
     already contain all of its supporting paragraphs.
  2. Ask the LLM for a note on these paragraphs.
  3. Sample one candidate rewrite per (temperature, top-p) setting of a 3 x 3 grid over {0.1, 0.5, 0.9}
     with the Query-Rewriter prompt. Each candidate proposes --queries_per_rewrite queries and picks a
     retriever (bm25 or dense) for each.
  4. Read the queries of each candidate as the Query Rewriter does at inference (its first
     --queries_per_rewrite items with both a strategy and a query), run every query through its
     retriever (top-k) and compute the recall of the supporting paragraphs over the union of the
     initial and the new paragraphs.
  5. chosen = a candidate with the highest recall, rejected = one with the lowest, ties broken at random.
     A question whose candidates all have the same recall is skipped. No LLM judge is involved.

The LLM is served by llm_server/serve.py at $LLM_SERVER_HOST:$LLM_SERVER_PORT; the retrievers are the
sparse (BM25) and dense servers of retriever_server/.

Usage (from the repository root):
    python training/gen_dpo_data_query_rewrite.py --dataset hotpotqa --queries_per_rewrite 2 --num_samples 500
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from inference.participant_agentic import parse_dispatch_output  # noqa: E402
from lib import read_jsonl  # noqa: E402
from models.llm_client_generator import LLMClientGenerator  # noqa: E402

REWRITE_PROMPT_FILE = os.path.join(
    REPO_ROOT, "prompts", "query_rewriter_dispatch"
)
SAMPLING_GRID = [(temperature, top_p) for temperature in (0.1, 0.5, 0.9) for top_p in (0.1, 0.5, 0.9)]
NOTE_TEMPERATURE, NOTE_MAX_TOKENS = 0.7, 512
REWRITE_MAX_TOKENS = 1024


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["hotpotqa", "musique", "2wikimultihopqa"], required=True,
                        help="Also the corpus name on the retrieval servers.")
    parser.add_argument("--input_data_path", default=None,
                        help="Training questions (default: <dataset>/train.jsonl in $SPARC_RAG_PROCESSED_DATA_ROOT, "
                             "default processed_data).")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of preference pairs to generate.")
    parser.add_argument("--num_rewrites", type=int, default=9,
                        help="Candidates per question, drawn from the 3 x 3 sampling grid (at most 9).")
    parser.add_argument("--queries_per_rewrite", type=int, default=2,
                        help="Queries per candidate (the width W; the paper's data uses 2, 3 and 4).")
    parser.add_argument("--gen_model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model served at the LLM endpoint.")
    parser.add_argument("--model_tokens_limit", type=int, default=32768,
                        help="Token limit of the LLM client for prompt plus generated tokens (as for the Query "
                             "Rewriter in configs/).")
    parser.add_argument("--sparse_url", default="http://localhost:8020", help="BM25 retrieval server.")
    parser.add_argument("--dense_url", default="http://localhost:8000", help="Dense retrieval server.")
    parser.add_argument("--retrieval_top_k", type=int, default=6, help="Paragraphs per retrieval.")
    parser.add_argument("--output_dir", default="training/data/query_rewrite")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="Print the candidates and recalls of every question.")
    args = parser.parse_args()
    if args.input_data_path is None:
        data_root = os.environ.get("SPARC_RAG_PROCESSED_DATA_ROOT", "processed_data")
        args.input_data_path = os.path.join(data_root, args.dataset, "train.jsonl")
    if not 1 <= args.num_rewrites <= len(SAMPLING_GRID):
        parser.error(f"--num_rewrites must be between 1 and {len(SAMPLING_GRID)}")
    return args


def check_llm_endpoint():
    host, port = os.environ.get("LLM_SERVER_HOST"), os.environ.get("LLM_SERVER_PORT")
    if not host or not port or not host.startswith(("http://", "https://")):
        sys.exit("Set LLM_SERVER_HOST (with http:// or https://) and LLM_SERVER_PORT, "
                 "e.g. export LLM_SERVER_HOST=http://localhost LLM_SERVER_PORT=8010")


def generate(llm, prompt, temperature, max_tokens, top_p=1.0):
    """Sample one completion. A failed request yields an empty string."""
    llm.temperature, llm.top_p, llm.max_length = temperature, top_p, max_tokens
    try:
        result = llm.generate_text_sequence(prompt, stop=None)
    except Exception as e:
        print(f"LLM request failed: {e}")
        return ""
    return result[0][0][0]


def retrieve(query, corpus, top_k, strategy, args):
    """Top-k paragraphs from the BM25 server for strategy "bm25", from the dense server otherwise."""
    if strategy and strategy.lower() == "bm25":
        url = args.sparse_url
        payload = {"retrieval_method": "retrieve_from_elasticsearch", "document_type": "title_paragraph_text"}
    else:
        url = args.dense_url
        payload = {"retrieval_method": "vector_based_similarity_search"}
    payload.update(corpus_name=corpus, query_text=query, max_hits_count=top_k)
    try:
        response = requests.post(f"{url}/retrieve/", json=payload, timeout=30)
        response.raise_for_status()
        return response.json().get("retrieval", [])
    except Exception as e:
        print(f"Retrieval failed for {query[:50]!r}: {e}")
        return []


def format_documents(documents):
    return "".join(
        f"[{i}] Title: {doc.get('title', 'Untitled')}\nText: {doc.get('paragraph_text', doc.get('text', ''))}\n\n"
        for i, doc in enumerate(documents, 1)
    )


def note_prompt(question, documents):
    return f"""You are a helpful research assistant. Based on the provided document content, write a concise note that integrates all relevant information to help answer the question. Please ensure that the note includes all original text information useful for answering the question.

Question: {question}

Document content:
{format_documents(documents)}

Write your note below:

"""


def add_new_paragraphs(documents, new_documents):
    """Append the paragraphs whose text is not already present."""
    merged = list(documents)
    seen = {doc.get("paragraph_text", "") for doc in documents}
    for doc in new_documents:
        text = doc.get("paragraph_text", "")
        if text and text not in seen:
            seen.add(text)
            merged.append(doc)
    return merged


def recall(documents, supporting_paragraphs):
    if not supporting_paragraphs:
        return 0.0
    retrieved = {doc.get("paragraph_text", "") for doc in documents}
    return sum(paragraph in retrieved for paragraph in supporting_paragraphs) / len(supporting_paragraphs)


def select_pair(candidates):
    """Indices of a highest-recall and a lowest-recall candidate (random among ties), or None if all tie."""
    groups = defaultdict(list)
    for candidate in candidates:
        groups[candidate["recall"]].append(candidate)
    recalls = sorted(groups, reverse=True)
    if len(recalls) < 2:
        return None
    best = random.choice(groups[recalls[0]])["rewrite_idx"]
    worst = random.choice(groups[recalls[-1]])["rewrite_idx"]
    return best, worst


def save_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    check_llm_endpoint()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    llm = LLMClientGenerator(
        model_name=args.gen_model, temperature=NOTE_TEMPERATURE, max_length=NOTE_MAX_TOKENS,
        model_tokens_limit=args.model_tokens_limit,
    )
    with open(REWRITE_PROMPT_FILE, encoding="utf-8") as f:
        rewrite_template = f.read()

    questions = read_jsonl(args.input_data_path)
    random.shuffle(questions)
    print(f"{len(questions)} questions in {args.input_data_path}; target: {args.num_samples} pairs")

    tag = f"{args.dataset}_width{args.queries_per_rewrite}"
    partial_file = os.path.join(args.output_dir, f"query_rewrite_dpo_{tag}.partial.jsonl")
    pairs, logs = [], []
    skipped = defaultdict(int)
    processed = 0
    for idx, sample in enumerate(questions):
        if len(pairs) >= args.num_samples:
            break
        processed += 1
        question = sample["question_text"]
        question_id = sample.get("question_id", f"sample_{idx}")
        supporting = [ctx.get("paragraph_text", "") for ctx in sample.get("contexts", []) if ctx.get("is_supporting")]

        initial_docs = retrieve(question, args.dataset, args.retrieval_top_k, None, args)
        if not initial_docs:
            skipped["no paragraphs retrieved"] += 1
            continue
        initial_recall = recall(initial_docs, supporting)
        if initial_recall >= 1.0:
            skipped["initial recall already 1.0"] += 1
            continue

        note = generate(llm, note_prompt(question, initial_docs), NOTE_TEMPERATURE, NOTE_MAX_TOKENS).strip()
        if not note:
            skipped["note generation failed"] += 1
            continue

        prompt = rewrite_template.format(query=question, current_query=question, context=note,
                                         N=args.queries_per_rewrite)
        grid = SAMPLING_GRID
        if args.num_rewrites < len(SAMPLING_GRID):
            grid = random.sample(SAMPLING_GRID, args.num_rewrites)
        rewrites = [
            {"text": generate(llm, prompt, temperature, REWRITE_MAX_TOKENS, top_p).strip(),
             "temperature": temperature, "top_p": top_p}
            for temperature, top_p in grid
        ]

        candidates = []
        for i, rewrite in enumerate(rewrites):
            queries, _ = parse_dispatch_output(rewrite["text"], args.queries_per_rewrite)
            new_docs = []
            for strategy, query in queries:
                new_docs = add_new_paragraphs(
                    new_docs, retrieve(query, args.dataset, args.retrieval_top_k, strategy, args)
                )
                time.sleep(0.1)  # be gentle with the retrieval servers
            docs = add_new_paragraphs(initial_docs, new_docs)
            candidates.append({
                "rewrite_idx": i,
                "queries": [{"strategy": strategy, "query": query} for strategy, query in queries],
                "num_new_docs": len(new_docs),
                "num_total_docs": len(docs),
                "recall": round(recall(docs, supporting), 4),
                "is_chosen": False,
                "is_rejected": False,
                "temperature": rewrite["temperature"],
                "top_p": rewrite["top_p"],
            })

        selection = select_pair(candidates)
        if args.debug:
            print(f"[{question_id}] {question}\n  initial recall {initial_recall:.3f}; candidate recalls "
                  f"{[c['recall'] for c in candidates]}; selected {selection}")
            for i, rewrite in enumerate(rewrites):
                print(f"  [{i}] T={rewrite['temperature']} p={rewrite['top_p']}: {rewrite['text'][:300]!r}")
        if selection is None:
            skipped["all candidates have the same recall"] += 1
            continue
        best, worst = selection
        candidates[best]["is_chosen"] = True
        candidates[worst]["is_rejected"] = True
        delta = candidates[best]["recall"] - candidates[worst]["recall"]

        pairs.append({
            "prompt": prompt,
            "chosen": rewrites[best]["text"],
            "rejected": rewrites[worst]["text"],
            "all_rewrites": rewrites,
            "metadata": {
                "question_id": question_id,
                "dataset": args.dataset,
                "dispatch_width": args.queries_per_rewrite,
                "best_idx": best,
                "worst_idx": worst,
                "margin": delta,
                "num_rewrites_generated": len(rewrites),
                "initial_recall": round(initial_recall, 4),
                "chosen_recall": candidates[best]["recall"],
                "rejected_recall": candidates[worst]["recall"],
                "delta_recall": round(delta, 4),
                "chosen_num_docs": candidates[best]["num_total_docs"],
                "rejected_num_docs": candidates[worst]["num_total_docs"],
                "num_ground_truth": len(supporting),
                "selection_method": "recall_only",
                "all_group_recalls": candidates,
                "chosen_temp": rewrites[best]["temperature"],
                "chosen_top_p": rewrites[best]["top_p"],
                "rejected_temp": rewrites[worst]["temperature"],
                "rejected_top_p": rewrites[worst]["top_p"],
            },
        })
        logs.append({
            "question_id": question_id,
            "question": question,
            "note": note,
            "rewrites": rewrites,
            "best_idx": best,
            "worst_idx": worst,
            "recall_metrics": {"initial": round(initial_recall, 4), "chosen": candidates[best]["recall"],
                               "rejected": candidates[worst]["recall"], "delta": round(delta, 4)},
            "all_group_recalls": candidates,
        })
        print(f"{len(pairs)}/{args.num_samples} pairs after {processed} questions "
              f"(recall initial {initial_recall:.3f}, chosen {candidates[best]['recall']:.3f}, "
              f"rejected {candidates[worst]['recall']:.3f})")
        if len(pairs) % 100 == 0:
            save_jsonl(partial_file, pairs)

    print(f"Processed {processed} questions, generated {len(pairs)} pairs; skipped: {dict(skipped)}")
    if len(pairs) < args.num_samples:
        print(f"WARNING: ran out of questions before reaching {args.num_samples} pairs", file=sys.stderr)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{tag}_{len(pairs)}samples_{timestamp}"
    save_jsonl(os.path.join(args.output_dir, f"query_rewrite_dpo_{stem}.jsonl"), pairs)
    with open(os.path.join(args.output_dir, f"generation_log_{stem}.json"), "w") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, f"metadata_{stem}.json"), "w") as f:
        json.dump({"args": vars(args), "num_instances": len(pairs), "dispatch_width": args.queries_per_rewrite,
                   "timestamp": timestamp, "gen_model": args.gen_model, "skipped": dict(skipped),
                   "truncated_prompts": llm.num_truncated_prompts}, f, indent=2)
    if os.path.exists(partial_file):
        os.remove(partial_file)
    print(f"Saved {len(pairs)} pairs to {os.path.join(args.output_dir, f'query_rewrite_dpo_{stem}.jsonl')}")


if __name__ == "__main__":
    main()
