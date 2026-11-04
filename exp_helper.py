#!/usr/bin/env python3
"""Materialise a config, then run predict.py (and optionally evaluate.py) on it.

1. Evaluates the base .jsonnet config and merges an overlay built from the shared LLM flags,
   the retriever flags and any --set overrides.
2. Writes the result to predictions/<trial>/<run_id>/config__<eval-file-stem>.jsonnet (the file
   predict.py keeps as the run's config) and records the run in predictions/<trial>/runs.json
   (run id, config hash, dataset, model name).
3. Runs predict.py on it and, with --run-evaluate, evaluate.py. Their console output, errors
   included, is appended to predictions/<trial>/predict.log and evaluate.log.

Examples
--------
# Predict and evaluate one paper configuration:
python exp_helper.py \
  --base-jsonnet configs/qwen2.5_7b_dpo_musique.jsonnet \
  --trial dpo_musique \
  --eval-path processed_data/musique/test_subsampled.jsonl \
  --run-evaluate -y

# Override fields of the config:
python exp_helper.py \
  --base-jsonnet configs/qwen2.5_7b_musique.jsonnet \
  --trial musique_w1 \
  --eval-path processed_data/musique/test_subsampled.jsonl \
  --set controller.fixed_width=1 \
  --set models.document_retriever.retrieval_count=8 \
  --run-evaluate -y

Values passed to --set are parsed as JSON when possible (true, null, 10, 0.2, "text") and used as
plain strings otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import _jsonnet

# Models whose shared LLM parameters the --llm-* flags override.
MODELS_WITH_LLM = [
    "answer_gen",
    "answer_eval",
    "context",
    "rewrite",
    "state_combiner",
]


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def canonical_json(o: Any) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"))


def _gen_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "_" + uuid.uuid4().hex[:6]


def short_hash(*parts: str, n: int = 10) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()[:n]


def evaluate_config(base_path: Path) -> Dict[str, Any]:
    """Evaluate a .jsonnet config."""
    return json.loads(_jsonnet.evaluate_file(str(base_path)))


def parse_kv_overrides(pairs: List[str]) -> Dict[str, Any]:
    def cast_val(v: str) -> Any:
        try:
            return json.loads(v)
        except Exception:
            return v

    root: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set entry must be key=value, got: {pair}")
        k, v = pair.split("=", 1)
        v = cast_val(v)
        parts = k.split(".")
        cur = root
        for p in parts[:-1]:
            nxt = cur.get(p)
            if nxt is None:
                cur[p] = {}
                nxt = cur[p]
            if not isinstance(nxt, dict):
                raise ValueError(
                    f"Override path collides with non-dict at '{p}' in {pair}"
                )
            cur = nxt
        cur[parts[-1]] = v
    return root


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def build_llm_overlay(
    models: List[str],
    model_name: Optional[str],
    temperature: Optional[float],
    max_length: Optional[int],
    stopword: Optional[str],
    tokens_limit: Optional[int],
) -> Dict[str, Any]:
    """Overlay that sets the given shared LLM parameters in every model of ``models``."""
    cfg: Dict[str, Any] = {}
    if model_name is not None:
        cfg["model_name"] = model_name
    if temperature is not None:
        cfg["temperature"] = temperature
    if max_length is not None:
        cfg["max_length"] = max_length
    if stopword is not None:
        cfg["stopword"] = stopword
    if tokens_limit is not None:
        cfg["model_tokens_limit"] = tokens_limit
    if not cfg:
        return {}
    return {"models": {m: dict(cfg) for m in models}}


def build_retriever_overlay(
    host: Optional[str], sparse_port: Optional[int], dense_port: Optional[int]
) -> Dict[str, Any]:
    """Overlay for the retriever endpoints of models.document_retriever."""
    retriever: Dict[str, Any] = {}
    if host is not None:
        retriever["retriever_host"] = {"bm25": host, "dense": host}
    ports = {k: v for k, v in (("bm25", sparse_port), ("dense", dense_port)) if v is not None}
    if ports:
        retriever["retriever_port"] = ports
    return {"models": {"document_retriever": retriever}} if retriever else {}


def ensure_trial_dir(trial: str, predictions_dir: str = "predictions") -> Path:
    trial_dir = Path(predictions_dir) / trial
    trial_dir.mkdir(parents=True, exist_ok=True)
    return trial_dir


def append_run_log(trial_dir: Path, entry: Dict[str, Any]) -> None:
    runs_path = trial_dir / "runs.json"
    data = json.loads(load_text(runs_path)) if runs_path.exists() else []
    data.append(entry)
    save_text(runs_path, json.dumps(data, indent=2))


def confirm(prompt: str) -> bool:
    try:
        ans = input(f"{prompt} [y/N]: ").strip().lower()
        return ans in ("y", "yes")
    except EOFError:
        return False


def run_subprocess(cmd: List[str], log_path: Path, env: Optional[Dict[str, str]] = None) -> int:
    """Run ``cmd`` with its stdout and stderr appended to ``log_path``; return its exit code."""
    print("$ " + " ".join(cmd))
    print(f"  console output -> {log_path} (follow it with: tail -f {log_path})", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write("$ " + " ".join(cmd) + "\n")
        fp.flush()
        return subprocess.call(cmd, stdout=fp, stderr=fp, env=env)


def main():
    p = argparse.ArgumentParser(description="Materialise a config and run predict.py (and evaluate.py).")

    # Base / Output
    p.add_argument(
        "--base-jsonnet", required=True, help="Path to the base .jsonnet config"
    )
    p.add_argument(
        "--trial", required=True, help="Trial name; creates <predictions-dir>/<trial>"
    )
    p.add_argument(
        "--eval-path",
        required=True,
        help="Path to evaluation jsonl for predict.py --eval_path",
    )
    p.add_argument(
        "--predictions-dir",
        default="predictions",
        help="Base predictions directory (default: predictions)",
    )

    # Shared LLM params
    p.add_argument(
        "--llm-model-name", default=None, help="Shared model_name for LLM modules"
    )
    p.add_argument(
        "--llm-temperature", type=float, default=None, help="Shared temperature"
    )
    p.add_argument("--llm-max-length", type=int, default=None, help="Shared max_length")
    p.add_argument("--llm-stopword", default=None, help="Shared stopword")
    p.add_argument(
        "--llm-tokens-limit", type=int, default=None, help="Shared model_tokens_limit"
    )

    # Arbitrary overrides
    p.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        help="Override any field via dot-path key=value, e.g. --set models.document_retriever.retrieval_count=8",
    )

    # Scripts
    p.add_argument("--predict-py", default="predict.py", help="Path to predict.py")
    p.add_argument(
        "--evaluate-py", default="evaluate.py", help="Path to evaluate.py"
    )
    p.add_argument(
        "--run-evaluate", action="store_true", help="Run evaluate.py after predict.py"
    )

    # Server configuration
    p.add_argument("--llm-host", default=None, help="Override LLM server host (e.g., http://localhost)")
    p.add_argument("--llm-port", type=int, default=None, help="Override LLM server port")
    p.add_argument(
        "--retriever-host",
        default=None,
        help="Override the host of both retrievers (models.document_retriever.retriever_host)",
    )
    p.add_argument("--sparse-retrieval-port", type=int, default=None, help="Override sparse (BM25) retrieval port")
    p.add_argument("--dense-retrieval-port", type=int, default=None, help="Override dense retrieval port")

    # Control
    p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompts")

    args = p.parse_args()

    base_path = Path(args.base_jsonnet).resolve()
    if not base_path.exists():
        print(f"[ERROR] Base jsonnet not found: {base_path}")
        sys.exit(2)

    trial_dir = ensure_trial_dir(args.trial, args.predictions_dir)

    # predict.py reads these (default: .llm_server_address.jsonnet).
    if args.llm_host is not None:
        os.environ["LLM_SERVER_HOST"] = str(args.llm_host)
    if args.llm_port is not None:
        os.environ["LLM_SERVER_PORT"] = str(args.llm_port)

    base_cfg = evaluate_config(base_path)
    existing_models = [m for m in MODELS_WITH_LLM if m in base_cfg.get("models", {})]

    llm_overlay = build_llm_overlay(
        existing_models,
        args.llm_model_name,
        args.llm_temperature,
        args.llm_max_length,
        args.llm_stopword,
        args.llm_tokens_limit,
    )
    retriever_overlay = build_retriever_overlay(
        args.retriever_host, args.sparse_retrieval_port, args.dense_retrieval_port
    )
    if retriever_overlay and "document_retriever" not in base_cfg.get("models", {}):
        print("[ERROR] The retriever flags need models.document_retriever in the base config.")
        sys.exit(2)
    set_overrides = parse_kv_overrides(args.set_overrides) if args.set_overrides else {}
    overlay = deep_merge(deep_merge(llm_overlay, retriever_overlay), set_overrides)
    total_cfg = deep_merge(base_cfg, overlay)

    # Every run gets its own generated config, so concurrent runs of a trial do not collide.
    run_id = _gen_run_id()
    eval_file_stem = os.path.splitext(os.path.basename(args.eval_path))[0]
    gen_cfg_path = trial_dir / run_id / f"config__{eval_file_stem}.jsonnet"
    save_text(gen_cfg_path, json.dumps(total_cfg, indent=2))

    h = short_hash(load_text(base_path), canonical_json(overlay), n=10)
    model_name = next(
        (
            total_cfg["models"][m]["model_name"]
            for m in MODELS_WITH_LLM
            if "model_name" in total_cfg.get("models", {}).get(m, {})
        ),
        "unknown",
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    dataset = os.path.basename(os.path.dirname(args.eval_path))

    manifest = {
        "run_id": run_id,
        "hash": h,
        "trial": args.trial,
        "dataset": dataset,
        "model_name": model_name,
        "generated_jsonnet": str(gen_cfg_path),
        "eval_path": args.eval_path,
        "command": " ".join(sys.argv),
        "timestamp": timestamp,
    }
    save_text(trial_dir / "run_manifest.latest.json", json.dumps(manifest, indent=2))
    append_run_log(trial_dir, manifest)

    print(f"[OK] Generated config: {gen_cfg_path}")
    print(f"[OK] Trial dir:       {trial_dir}")

    controller = total_cfg.get("controller", {})
    if controller.get("fixed_length", -1) == -1 and controller.get("max_length", -1) == -1:
        print(
            "WARNING: Adaptive depth (controller.fixed_length=-1) without controller.max_length: "
            "the number of rounds is unbounded. Add e.g. --set controller.max_length=8.",
            file=sys.stderr,
        )

    if not args.yes and not confirm("Proceed to run predict.py now?"):
        print("Aborted before running predict.py.")
        sys.exit(0)

    predict_cmd = [
        sys.executable,
        args.predict_py,
        "--config_path",
        str(gen_cfg_path),
        "--eval_path",
        args.eval_path,
        "--trial_name",
        args.trial,
        "--run_id",
        run_id,
        "--predictions_dir",
        args.predictions_dir,
    ]
    env = os.environ.copy()
    predict_log = trial_dir / "predict.log"
    code_pred = run_subprocess(predict_cmd, predict_log, env=env)
    if code_pred != 0:
        print(f"[ERROR] predict.py failed with exit code {code_pred}; see the end of {predict_log}")
        sys.exit(code_pred)

    if args.run_evaluate:
        if not args.yes and not confirm("Run evaluate.py as well?"):
            print("Skipped evaluate.py.")
            sys.exit(0)

        eval_cmd = [
            sys.executable,
            args.evaluate_py,
            "--eval_path",
            args.eval_path,
            "--trial_name",
            args.trial,
            "--run_id",
            run_id,
            "--predictions_dir",
            args.predictions_dir,
        ]
        evaluate_log = trial_dir / "evaluate.log"
        code_eval = run_subprocess(eval_cmd, evaluate_log, env=env)
        if code_eval != 0:
            print(f"[ERROR] evaluate.py failed with exit code {code_eval}; see the end of {evaluate_log}")
            sys.exit(code_eval)

    print("[DONE] All steps finished successfully.")


if __name__ == "__main__":
    main()
