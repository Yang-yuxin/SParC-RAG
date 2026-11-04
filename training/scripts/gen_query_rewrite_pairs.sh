#!/usr/bin/env bash
# Step 1: Query-Rewriter preference pairs for HotpotQA, MuSiQue and 2WikiMultiHopQA at widths 2, 3 and 4.
#
# Needs the base model at the LLM endpoint and the BM25 and dense retrieval servers (see training/README.md).
# By default every (dataset, width) asks for as many pairs as the data behind the paper's checkpoints has
# (2,854 in total); set NUM_SAMPLES to use one target for all.
#
# Environment (defaults in brackets):
#   LLM_SERVER_HOST [http://localhost]   LLM_SERVER_PORT [8010]
#   SPARSE_URL [http://localhost:8020]   DENSE_URL [http://localhost:8000]
#   OUTPUT_DIR [training/data/query_rewrite]   NUM_SAMPLES [per dataset and width, see below]
#   SPARC_RAG_PROCESSED_DATA_ROOT [processed_data] (the questions are <dataset>/train.jsonl there)
#
# Usage (from anywhere): bash training/scripts/gen_query_rewrite_pairs.sh [extra generator arguments]
set -euo pipefail
cd "$(dirname "$0")/../.."

export LLM_SERVER_HOST="${LLM_SERVER_HOST:-http://localhost}"
export LLM_SERVER_PORT="${LLM_SERVER_PORT:-8010}"
SPARSE_URL="${SPARSE_URL:-http://localhost:8020}"
DENSE_URL="${DENSE_URL:-http://localhost:8000}"
OUTPUT_DIR="${OUTPUT_DIR:-training/data/query_rewrite}"

paper_pairs() {  # pairs per dataset and width in the data behind the paper's checkpoints
  case "$1_$2" in
    musique_2) echo 279 ;; musique_3) echo 267 ;; musique_4) echo 263 ;;
    hotpotqa_2) echo 353 ;; hotpotqa_3) echo 348 ;; hotpotqa_4) echo 300 ;;
    2wikimultihopqa_2) echo 321 ;; 2wikimultihopqa_3) echo 355 ;; 2wikimultihopqa_4) echo 368 ;;
  esac
}

for dataset in musique hotpotqa 2wikimultihopqa; do
  for width in 2 3 4; do
    num_samples="${NUM_SAMPLES:-$(paper_pairs "$dataset" "$width")}"
    echo "== ${dataset}, width ${width}: ${num_samples} pairs"
    python training/gen_dpo_data_query_rewrite.py \
      --dataset "$dataset" \
      --queries_per_rewrite "$width" \
      --num_samples "$num_samples" \
      --num_rewrites 9 \
      --retrieval_top_k 6 \
      --sparse_url "$SPARSE_URL" \
      --dense_url "$DENSE_URL" \
      --output_dir "$OUTPUT_DIR" \
      --seed 42 \
      "$@"
  done
done
