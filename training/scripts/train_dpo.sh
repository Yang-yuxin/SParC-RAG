#!/usr/bin/env bash
# Steps 3-5: build the joint preference dataset, train one LoRA adapter with asymmetric DPO, merge it.
#
# The pair files are merged in a fixed order (Query Rewriter: MuSiQue, HotpotQA, 2WikiMultiHopQA at
# widths 2, 3, 4; then Answer Evaluator: MuSiQue, HotpotQA, 2WikiMultiHopQA), the order of the paper's
# data; each pattern must match exactly one file. Stop the LLM server first to free the GPUs.
#
# Environment (defaults in brackets):
#   MODEL [Qwen/Qwen2.5-7B-Instruct]
#   NUM_GPUS [2]: effective batch = NUM_GPUS x 1 x 4 (2 GPUs for the paper's checkpoints)
#   QR_DIR [training/data/query_rewrite]   AE_DIR [training/data/answer_eval]
#   DATA_DIR [training/data/dpo_joint]: built from QR_DIR and AE_DIR unless it exists
#   ADAPTER_DIR [training/checkpoints/dpo_<model name>]   MASTER_PORT [29500]
#
# Extra arguments go to train_dpo_asymmetric.py. Settings of the paper's checkpoints:
#   bash training/scripts/train_dpo.sh
#   MODEL=Qwen/Qwen3-32B bash training/scripts/train_dpo.sh \
#     --deepspeed training/ds_config_zero2.json --max_length 4608 --max_prompt_length 4096
set -euo pipefail
shopt -s nullglob
cd "$(dirname "$0")/../.."

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
NUM_GPUS="${NUM_GPUS:-2}"
QR_DIR="${QR_DIR:-training/data/query_rewrite}"
AE_DIR="${AE_DIR:-training/data/answer_eval}"
DATA_DIR="${DATA_DIR:-training/data/dpo_joint}"
model_name="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')"
ADAPTER_DIR="${ADAPTER_DIR:-training/checkpoints/dpo_${model_name}}"

inputs=()
add_input() {
  local matches=($1)
  if [ "${#matches[@]}" -ne 1 ]; then
    echo "Expected exactly one file matching $1, found ${#matches[@]}." \
         "Move the extra files away or call training/prepare_dpo_dataset.py directly." >&2
    exit 1
  fi
  inputs+=("${matches[0]}")
}

if [ ! -d "$DATA_DIR" ]; then
  for dataset in musique hotpotqa 2wikimultihopqa; do
    for width in 2 3 4; do
      add_input "${QR_DIR}/query_rewrite_dpo_${dataset}_width${width}_*samples_*.jsonl"
    done
  done
  for dataset in musique hotpotqa 2wikimultihopqa; do
    add_input "${AE_DIR}/answer_eval_dpo_${dataset}_*samples_*.jsonl"
  done
  python training/prepare_dpo_dataset.py --inputs "${inputs[@]}" --output_dir "$DATA_DIR"
fi

torchrun --nproc_per_node "$NUM_GPUS" --master_port "${MASTER_PORT:-29500}" training/train_dpo_asymmetric.py \
  --model_name_or_path "$MODEL" \
  --dataset_path "$DATA_DIR" \
  --output_dir "$ADAPTER_DIR" \
  "$@"

python training/merge_lora_adapter.py --adapter_path "$ADAPTER_DIR"
