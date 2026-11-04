"""Merge a LoRA adapter trained by train_dpo_asymmetric.py into its base model.

The base model is read from the adapter's adapter_config.json unless --base_model is given. The merged
model and the base model's tokenizer are written to --output_path, by default
training/models/<adapter directory name>_merged.

Usage (from the repository root):
    python training/merge_lora_adapter.py --adapter_path training/checkpoints/dpo_qwen2.5-7b-instruct
"""

import argparse
import json
import os
from datetime import datetime

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter_path", required=True, help="Directory with adapter_config.json.")
    parser.add_argument("--base_model", default=None,
                        help="Base model (default: base_model_name_or_path of adapter_config.json).")
    parser.add_argument("--output_path", default=None,
                        help="Default: training/models/<adapter directory name>_merged.")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto",
                        help="dtype of the merged weights (auto: the base checkpoint's dtype).")
    parser.add_argument("--device_map", default="auto")
    args = parser.parse_args()

    adapter_path = args.adapter_path.rstrip("/")
    with open(os.path.join(adapter_path, "adapter_config.json")) as f:
        base_model = args.base_model or json.load(f)["base_model_name_or_path"]
    adapter_name = os.path.basename(adapter_path)
    output_path = args.output_path or os.path.join("training", "models", f"{adapter_name}_merged")
    dtype = args.dtype if args.dtype == "auto" else getattr(torch, args.dtype)

    print(f"Merging {adapter_path} into {base_model}")
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype, device_map=args.device_map)
    model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(output_path)
    with open(os.path.join(output_path, "merge_metadata.json"), "w") as f:
        json.dump({"base_model": base_model, "adapter": adapter_name,
                   "merge_date": datetime.now().isoformat(timespec="seconds")}, f, indent=2)
    print(f"Saved the merged model to {output_path}")


if __name__ == "__main__":
    main()
