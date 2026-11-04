#!/usr/bin/env bash
# Download IRCoT's processed HotpotQA, 2WikiMultiHopQA and MuSiQue files, including the
# 500-question test subsamples (test_subsampled.jsonl) the paper evaluates on.
#
# Run from the repository root:
#   bash download/processed_data.sh
#
# Files go to $SPARC_RAG_PROCESSED_DATA_ROOT (default: processed_data). Needs unzip and
# gdown (pip install gdown). If gdown fails, download the zip from the URL below by hand.
set -euo pipefail

PROCESSED_DATA_ROOT="${SPARC_RAG_PROCESSED_DATA_ROOT:-processed_data}"
TEMP_DIR=".temp"

command -v gdown >/dev/null || { echo "gdown not found; install it with: pip install gdown" >&2; exit 1; }
mkdir -p "$TEMP_DIR" "$PROCESSED_DATA_ROOT"

# URL: https://drive.google.com/file/d/1t2BjJtsejSIUZI54PKObMFG6_wMMG3bC/view?usp=sharing
gdown "1t2BjJtsejSIUZI54PKObMFG6_wMMG3bC&confirm=t" -O "$TEMP_DIR/processed_data.zip"
unzip -o "$TEMP_DIR/processed_data.zip" -d "$TEMP_DIR" \
    "processed_data/hotpotqa/*" "processed_data/2wikimultihopqa/*" "processed_data/musique/*"
cp -r "$TEMP_DIR/processed_data/." "$PROCESSED_DATA_ROOT/"

rm -rf "$TEMP_DIR"

# Resulting layout of $SPARC_RAG_PROCESSED_DATA_ROOT, for each of hotpotqa, 2wikimultihopqa
# and musique:
# ├── annotated_only_train.jsonl
# ├── dev.jsonl
# ├── dev_subsampled.jsonl
# ├── test_subsampled.jsonl
# └── train.jsonl
