# Fine-tuning: preference data and asymmetric DPO

SPARC-RAG fine-tunes one LoRA adapter that is shared by the Query Rewriter (QR) and the Answer
Evaluator (AE). The adapter is trained with DPO on preference pairs of both modules; the prompt tells
the two roles apart. Pairs in which the AE should continue but the rejected output stops
(wrong-stop pairs) get a larger loss weight λ. See the paper, Section 3 ("Scaling-oriented
Fine-tuning") and Appendix "Implementation Details".

The preference data used in the paper is not included; this directory contains the code that generates
it. The fine-tuned configurations use the model names `Qwen2.5-7B-Instruct-SPARC-DPO` and
`Qwen3-32B-SPARC-DPO`: one checkpoint per backbone, trained on the joint QR + AE data.

## Overview

| Step | Command | Output |
|---|---|---|
| 1. Query-Rewriter pairs | `bash training/scripts/gen_query_rewrite_pairs.sh` | `training/data/query_rewrite/query_rewrite_dpo_<dataset>_width<W>_<N>samples_<time>.jsonl` |
| 2. Answer-Evaluator pairs | `bash training/scripts/gen_answer_eval_pairs.sh` | `training/data/answer_eval/answer_eval_dpo_<dataset>_<N>samples_<time>.jsonl` |
| 3–5. Dataset, training, merge | `bash training/scripts/train_dpo.sh` | `training/data/dpo_joint/`, `training/checkpoints/dpo_<model>/`, `training/models/dpo_<model>_merged/` |

The shell scripts run from the repository root (they change to it themselves) and are configured
through the environment variables listed at their top; extra arguments are passed to the Python script
they wrap. In the file names, `<N>` is the number of pairs for the QR files and the number of
questions with an AE step for the AE files.

| File | Purpose |
|---|---|
| `gen_dpo_data_query_rewrite.py` | QR pairs for one dataset and width |
| `gen_dpo_data_answer_eval.py` | AE pairs from rollout traces of one dataset |
| `prepare_dpo_dataset.py` | Merges pair files and splits them 90/10 into a `DatasetDict` |
| `train_dpo_asymmetric.py` | Joint DPO with the wrong-stop weight λ (TRL) |
| `ds_config_zero2.json` | DeepSpeed ZeRO-2 with CPU optimizer offload, bf16 (used for the 32B model) |
| `merge_lora_adapter.py` | Merges the adapter into the base model |
| `scripts/*.sh` | The three drivers above |

## Requirements

- The packages in `requirements.txt`. `openai` is needed only for an OpenAI judge in step 2 and DeepSpeed
  only for `--deepspeed`.
- Steps 1–2 use the services of the inference pipeline (main README): `llm_server/serve.py` serving the
  **base** model Qwen2.5-7B-Instruct at `LLM_SERVER_HOST=http://localhost`, `LLM_SERVER_PORT=8010`, the
  BM25 retrieval server on port 8020 and the dense one on port 8000, and `<dataset>/train.jsonl` for
  HotpotQA, MuSiQue and 2WikiMultiHopQA in `processed_data/` (or `SPARC_RAG_PROCESSED_DATA_ROOT`). NQ and
  Bamboogle are not used for training. The data of both of the paper's checkpoints, including the 32B one,
  was generated with Qwen2.5-7B-Instruct. Step 1 takes the retriever URLs from
  `--sparse_url` and `--dense_url` (driver: `SPARSE_URL`, `DENSE_URL`); the rollouts of step 2 use the
  endpoints of the configurations.
- Steps 3–5 need GPUs; stop the LLM server first. The paper's checkpoints were trained on one node
  with 2 NVIDIA H100 (80 GB): the 7B adapter with plain DDP in about 1 hour, the 32B adapter with
  DeepSpeed ZeRO-2 in about 3.8 hours. The base weights are loaded in fp32 by default, as in the
  paper's runs; for 32B this needs about 130 GB of host memory per process. DeepSpeed casts the model
  to bf16 when it starts, so with `--deepspeed` the option `--base_dtype bfloat16` halves this memory.

## Step 1: Query-Rewriter pairs

```bash
bash training/scripts/gen_query_rewrite_pairs.sh
```

For each dataset and width W ∈ {2, 3, 4} (queries per rewrite), `gen_dpo_data_query_rewrite.py` goes
through the training questions in a seeded random order. For each question it

1. retrieves the dense top-6 paragraphs and skips the question if they contain all supporting paragraphs;
2. asks the LLM for a note on these paragraphs (temperature 0.7, at most 512 tokens);
3. samples 9 candidate rewrites with the QR prompt
   (`prompts/query_rewriter_dispatch`), one per
   temperature × top-p setting in {0.1, 0.5, 0.9}²; each candidate proposes W queries and a retriever
   (bm25 or dense) for each;
4. reads the queries of each candidate as the QR does at inference (its first W items with both a
   strategy and a query), retrieves the top-6 paragraphs per query and computes the recall of the
   supporting paragraphs over the union of the initial and the new paragraphs;
5. takes a highest-recall candidate as chosen and a lowest-recall one as rejected (random among ties);
   questions whose candidates all have the same recall are skipped. No LLM judge is involved.

It stops at `--num_samples` pairs. The driver asks for the counts of the paper's data: MuSiQue
279/267/263, HotpotQA 353/348/300 and 2WikiMultiHopQA 321/355/368 pairs for W = 2/3/4 (2,854 in total).
Generation is slow: the HotpotQA W = 4 file of the paper's data took about 27 hours.

## Step 2: Answer-Evaluator pairs

```bash
bash training/scripts/gen_answer_eval_pairs.sh                    # judge: the base model
JUDGE_MODEL=gpt-4o bash training/scripts/gen_answer_eval_pairs.sh # judge of the paper's data (needs OPENAI_API_KEY)
```

**2a. Rollouts.** `exp_helper.py` runs SPARC-RAG with `configs/qwen2.5_7b_dpo_<dataset>.jsonnet`, every
module switched to the base model, width 2, depth 2 (`controller.fixed_length=2`) and
`models.answer_eval.always_reject=true`, so that every round reaches the AE and the loop never stops
early. The driver uses the question slices of the paper's data: questions 1–500 of MuSiQue and
501–1000 of HotpotQA and 2WikiMultiHopQA (`NUM_QUESTIONS` sets the size). The traces are written to
`predictions/dpo_rollouts_<dataset>/<run_id>/prediction__train_dpo_rollouts_all.json`.

**2b. Pairs.** For each question, `gen_dpo_data_answer_eval.py`

1. picks one AE step of the trace at random and rebuilds its prompt with
   `prompts/answer_evaluator` from the question, the note of the preceding
   Context-Manager update and the candidate answer (questions with an empty note are skipped);
2. labels the candidate answer correct if its SQuAD F1 against a gold answer exceeds 0.7 or it matches
   one exactly;
3. samples 9 judgements (temperature 1.0, top-p 0.95, stop at `[END]`) and reads the verdict of each,
   `yes` (accept: STOP) or `no` (reject: CONTINUE), as the AE does at inference; questions whose
   judgements all agree are skipped;
4. lets the judge pick the best `yes` and the best `no` judgement (a random pick if the judge request
   fails or its answer cannot be parsed; such fallbacks are counted in the metadata file);
5. uses the judgement that matches the label as chosen and the other as rejected.

For the paper's data, 477 (MuSiQue), 471 (HotpotQA) and 476 (2WikiMultiHopQA) questions had an AE step
and 141, 160 and 201 of them produced a pair (502 in total).

## Steps 3–5: dataset, training, merge

```bash
bash training/scripts/train_dpo.sh                                    # Qwen2.5-7B-Instruct
MODEL=Qwen/Qwen3-32B bash training/scripts/train_dpo.sh \
  --deepspeed training/ds_config_zero2.json --max_length 4608 --max_prompt_length 4096
```

**Step 3.** `prepare_dpo_dataset.py` concatenates the 12 pair files in a fixed order (QR: MuSiQue,
HotpotQA, 2WikiMultiHopQA, each at W = 2, 3, 4; then AE: MuSiQue, HotpotQA, 2WikiMultiHopQA), shuffles the
pairs with `random.Random(42)` and keeps the first 90% for training and the rest for validation. Given
the 12 files of the paper's data in this order, it reproduces their split exactly: 3,020 training
pairs (2,573 QR + 447 AE, of which 227 are wrong-stop pairs) and 336 validation pairs (281 QR + 55 AE).
It adds `role` and `dataset` columns and prints the composition of both splits.

**Step 4.** `train_dpo_asymmetric.py` trains the adapter with these defaults:

| Setting | Default |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` (`--model_name_or_path`) |
| LoRA | r 16, α 32, dropout 0.05 on `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| DPO | sigmoid loss, β 0.1, wrong-stop weight λ 2.0, reference = base model (adapter disabled) |
| Optimisation | AdamW, learning rate 5e-5 with linear decay and no warm-up, gradient clipping 1.0, 3 epochs |
| Batch | 1 per GPU × 4 accumulation steps; 2 GPUs in the driver (effective batch 8, 1,134 steps) |
| Length | 8,192 tokens, prompt 8,000 (the 32B run used 4,608 / 4,096) |
| Precision | bf16 mixed precision, base weights loaded in fp32; gradient checkpointing |
| Other | seed 42; validation on the test split after every epoch; TensorBoard logs |

The weight of each pair is computed from its text before TRL tokenizes it: λ if the verdict of the
chosen output is `no` and that of the rejected output is `yes`, 1 otherwise. Verdicts are read as the
AE reads them at inference (`parse_verdict` in `inference/participant_agentic.py`: the first of the first
two lines that starts with yes/accept or no/reject decides); QR outputs start with a tag, so QR pairs
keep weight 1. The weight multiplies the pair's DPO loss before the batch mean,
`loss = mean_i w_i · (−log σ(β · margin_i))`; λ = 1 gives standard DPO. At start-up the script logs how
many pairs get weight λ in each split and role.

The script writes to `--output_dir`: the adapter, the tokenizer, `training_args.json`,
`trainer_state.json`, checkpoints every 500 steps and TensorBoard logs in `runs/`. Besides TRL's
metrics, `train/loss_weight` is the mean weight of the pairs seen since the last log, and the
validation metrics are reported per role (`eval/query_rewrite_*`, `eval/answer_eval_*`) when the
dataset has a `role` column.

**Step 5.** `merge_lora_adapter.py` merges the adapter into the base model named in its
`adapter_config.json` and saves the result, with the tokenizer and a `merge_metadata.json`, to
`training/models/<adapter directory>_merged`. Serve that directory with `llm_server/serve.py` and run
the `configs/*_dpo_*.jsonnet` configurations with it as described in the main README.
