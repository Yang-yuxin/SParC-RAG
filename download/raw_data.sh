#!/usr/bin/env bash
# Download the raw data that the retrieval corpora and the NQ questions are built from.
#
# Run from the repository root:
#   bash download/raw_data.sh
#
# Files go to $SPARC_RAG_RAW_DATA_ROOT (default: raw_data). Steps whose output already
# exists are skipped, so the script can be re-run after a failed download. If a URL is
# unavailable, download the file by hand and place it as in the layout at the end.
# Needs wget, gzip, tar (with bzip2), unzip and gdown (pip install gdown), and about 30 GB
# of free disk (24 GB remain after the temporary archives are removed).
set -euo pipefail

RAW_DATA_ROOT="${SPARC_RAG_RAW_DATA_ROOT:-raw_data}"
TEMP_DIR=".temp"

command -v gdown >/dev/null || { echo "gdown not found; install it with: pip install gdown" >&2; exit 1; }
mkdir -p "$TEMP_DIR"

# Natural Questions (DPR retriever splits) and the DPR Wikipedia passages, which form the
# corpus `wiki` used for NQ and Bamboogle.
mkdir -p "$RAW_DATA_ROOT/nq"
for path in wikipedia_split/psgs_w100.tsv data/retriever/biencoder-nq-train.json data/retriever/biencoder-nq-dev.json; do
    target="$RAW_DATA_ROOT/nq/$(basename "$path")"
    if [ ! -f "$target" ]; then
        wget -O "$target.gz" "https://dl.fbaipublicfiles.com/dpr/$path.gz"
        gzip -d "$target.gz"
    fi
done

# HotpotQA: introductory paragraphs of the 2017-10-01 English Wikipedia (corpus `hotpotqa`).
# The directory name `wikpedia-paragraphs` is the one IRCoT uses.
if [ ! -d "$RAW_DATA_ROOT/hotpotqa/wikpedia-paragraphs" ]; then
    mkdir -p "$RAW_DATA_ROOT/hotpotqa"
    wget -O "$TEMP_DIR/hotpotqa_wikipedia.tar.bz2" \
        https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2
    tar -xjf "$TEMP_DIR/hotpotqa_wikipedia.tar.bz2" -C "$TEMP_DIR"
    mv "$TEMP_DIR/enwiki-20171001-pages-meta-current-withlinks-abstracts" "$RAW_DATA_ROOT/hotpotqa/wikpedia-paragraphs"
fi

# 2WikiMultiHopQA: the context paragraphs of all splits form the corpus `2wikimultihopqa`.
if [ ! -f "$RAW_DATA_ROOT/2wikimultihopqa/train.json" ]; then
    wget -O "$TEMP_DIR/2wikimultihopqa.zip" "https://www.dropbox.com/s/7ep3h8unu2njfxv/data_ids.zip?dl=1"
    unzip -jo "$TEMP_DIR/2wikimultihopqa.zip" -d "$RAW_DATA_ROOT/2wikimultihopqa" -x "*.DS_Store" "__MACOSX/*"
fi

# MuSiQue v1.0: the paragraphs of all splits form the corpus `musique`.
# URL: https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view?usp=sharing
if [ ! -f "$RAW_DATA_ROOT/musique/musique_full_v1.0_train.jsonl" ]; then
    gdown "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h&confirm=t" -O "$TEMP_DIR/musique_v1.0.zip"
    unzip -jo "$TEMP_DIR/musique_v1.0.zip" -d "$RAW_DATA_ROOT/musique" -x "*.DS_Store" "__MACOSX/*"
fi

rm -rf "$TEMP_DIR"

# Resulting layout of $SPARC_RAG_RAW_DATA_ROOT:
# ├── 2wikimultihopqa
# │   ├── dev.json
# │   ├── id_aliases.json
# │   ├── test.json
# │   └── train.json
# ├── hotpotqa
# │   └── wikpedia-paragraphs/{AA,AB,...}/wiki_*.bz2
# ├── musique
# │   ├── dev_test_singlehop_questions_v1.0.json
# │   ├── musique_ans_v1.0_{train,dev,test}.jsonl
# │   └── musique_full_v1.0_{train,dev,test}.jsonl
# └── nq
#     ├── biencoder-nq-dev.json
#     ├── biencoder-nq-train.json
#     └── psgs_w100.tsv
