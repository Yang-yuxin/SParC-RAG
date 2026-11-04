#!/usr/bin/env bash
# Step 2: Answer-Evaluator preference pairs for HotpotQA, MuSiQue and 2WikiMultiHopQA.
#
#   2a. Roll out SPARC-RAG with the base model on a slice of the training questions: width 2, depth 2,
#       and models.answer_eval.always_reject=true so that every round reaches the Answer Evaluator.
#   2b. Turn one Answer-Evaluator step per question into a preference pair (gen_dpo_data_answer_eval.py).
#
# Needs the base model at the LLM endpoint and the retrieval servers (see training/README.md). By default
# the slices are those of the paper's data: questions 1-500 of MuSiQue and 501-1000 of HotpotQA and
# 2WikiMultiHopQA (<dataset>/train.jsonl in SPARC_RAG_PROCESSED_DATA_ROOT).
#
# Environment (defaults in brackets):
#   LLM_SERVER_HOST [http://localhost]   LLM_SERVER_PORT [8010]   GEN_MODEL [Qwen/Qwen2.5-7B-Instruct]
#   JUDGE_MODEL [GEN_MODEL; e.g. gpt-4o, as for the paper's data, needs OPENAI_API_KEY]
#   NUM_QUESTIONS [500]   OUTPUT_DIR [training/data/answer_eval]   TRIAL_PREFIX [dpo_rollouts]
#   SPARC_RAG_PROCESSED_DATA_ROOT [processed_data]
#
# Usage (from anywhere): bash training/scripts/gen_answer_eval_pairs.sh [extra generator arguments]
set -euo pipefail
cd "$(dirname "$0")/../.."

export LLM_SERVER_HOST="${LLM_SERVER_HOST:-http://localhost}"
export LLM_SERVER_PORT="${LLM_SERVER_PORT:-8010}"
GEN_MODEL="${GEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
JUDGE_MODEL="${JUDGE_MODEL:-$GEN_MODEL}"
NUM_QUESTIONS="${NUM_QUESTIONS:-500}"
OUTPUT_DIR="${OUTPUT_DIR:-training/data/answer_eval}"
TRIAL_PREFIX="${TRIAL_PREFIX:-dpo_rollouts}"
DATA_ROOT="${SPARC_RAG_PROCESSED_DATA_ROOT:-processed_data}"

first_question() {  # 0-based start of each slice in the data behind the paper's checkpoints
  case "$1" in
    musique) echo 0 ;;
    hotpotqa | 2wikimultihopqa) echo 500 ;;
  esac
}

for dataset in musique hotpotqa 2wikimultihopqa; do
  first="$(first_question "$dataset")"
  questions="${DATA_ROOT}/${dataset}/train_dpo_rollouts.jsonl"
  sed -n "$((first + 1)),$((first + NUM_QUESTIONS))p" "${DATA_ROOT}/${dataset}/train.jsonl" > "$questions"
  trial="${TRIAL_PREFIX}_${dataset}"
  echo "== ${dataset}: rollouts on questions $((first + 1))-$((first + NUM_QUESTIONS))"
  python exp_helper.py \
    --base-jsonnet "configs/qwen2.5_7b_dpo_${dataset}.jsonnet" \
    --trial "$trial" \
    --eval-path "$questions" \
    --llm-host "$LLM_SERVER_HOST" \
    --llm-port "$LLM_SERVER_PORT" \
    --llm-model-name "${GEN_MODEL##*/}" \
    --set controller.fixed_length=2 \
    --set controller.fixed_width=2 \
    --set models.answer_eval.always_reject=true \
    -y

  run_id="$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['run_id'])" \
    "predictions/${trial}/run_manifest.latest.json")"
  echo "== ${dataset}: preference pairs from run ${run_id}"
  python training/gen_dpo_data_answer_eval.py \
    --dataset "$dataset" \
    --predictions_path "predictions/${trial}/${run_id}/prediction__train_dpo_rollouts_all.json" \
    --ground_truth_path "$questions" \
    --gen_model "$GEN_MODEL" \
    --judge_model "$JUDGE_MODEL" \
    --num_generations 9 \
    --output_dir "$OUTPUT_DIR" \
    --seed 42 \
    "$@"
done
