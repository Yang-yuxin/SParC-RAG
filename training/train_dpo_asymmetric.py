"""Joint DPO training of the Query Rewriter and the Answer Evaluator with an asymmetric pair weight.

One LoRA adapter is trained on the preference pairs of both modules, as built by
prepare_dpo_dataset.py; the prompt tells the two roles apart. The Answer Evaluator ends the retrieval
loop when it answers "yes" (STOP: accept the candidate answer) and starts another round when it answers
"no" (CONTINUE). Stopping when the loop should continue, i.e. accepting an incorrect answer, is the
costlier error. The DPO loss of every pair whose chosen output says "no" and whose rejected output says
"yes" (a wrong-stop pair) is therefore multiplied by lambda (--wrong_stop_multiplier):

    loss = mean_i( w_i * -log sigmoid(beta * margin_i) ),  w_i = lambda for wrong-stop pairs, 1 otherwise.

Verdicts are read as at inference (inference.participant_agentic.parse_verdict). Query-Rewriter pairs and
Answer-Evaluator pairs that prefer "yes" keep weight 1. The defaults are listed in training/README.md
(step 4).

Usage (from the repository root):
    torchrun --nproc_per_node 2 training/train_dpo_asymmetric.py \
        --dataset_path training/data/dpo_joint \
        --output_dir training/checkpoints/dpo_qwen2.5-7b-instruct
"""

import argparse
import collections
import json
import os
import sys

import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from inference.participant_agentic import parse_verdict  # noqa: E402


def is_wrong_stop_pair(chosen, rejected):
    """True if the pair prefers CONTINUE ("no") over STOP ("yes")."""
    return parse_verdict(chosen)[0] == "no" and parse_verdict(rejected)[0] == "yes"


def add_loss_weights(dataset, wrong_stop_multiplier):
    """Add the per-pair ``loss_weight`` column (computed from the raw text, before TRL tokenizes it)."""
    weights = [
        wrong_stop_multiplier if is_wrong_stop_pair(chosen, rejected) else 1.0
        for chosen, rejected in zip(dataset["chosen"], dataset["rejected"])
    ]
    return dataset.add_column("loss_weight", weights)


class LossWeightCollator:
    """Wrap TRL's preference collator so that the ``loss_weight`` of each pair is batched too."""

    def __init__(self, collator):
        self.collator = collator

    def __call__(self, examples):
        batch = self.collator(examples)
        batch["loss_weight"] = torch.tensor([example["loss_weight"] for example in examples], dtype=torch.float32)
        return batch


class AsymmetricDPOTrainer(DPOTrainer):
    """DPOTrainer that multiplies the DPO loss of each pair by its ``loss_weight`` before the batch mean.

    The ``loss_weight`` column is kept through TRL's preprocessing (it is a signature column), batched by
    ``LossWeightCollator`` and applied in ``dpo_loss``, which returns one loss per pair.

    Validation metrics are named after the evaluation set, e.g. ``eval_answer_eval_rewards/accuracies``
    when the trainer is given one evaluation set per role.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_collator = LossWeightCollator(self.data_collator)
        self._loss_weights = None
        self._metric_key_prefix = "eval"

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        if "loss_weight" not in self._signature_columns:
            self._signature_columns.append("loss_weight")

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        weights = batch.pop("loss_weight")
        self._loss_weights = weights
        loss, metrics = super().get_batch_loss_metrics(model, batch, train_eval)
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}loss_weight"] = self.accelerator.gather_for_metrics(weights).mean().item()
        return loss, metrics

    def dpo_loss(self, *args, **kwargs):
        losses, chosen_rewards, rejected_rewards = super().dpo_loss(*args, **kwargs)
        return losses * self._loss_weights.to(losses.device, losses.dtype), chosen_rewards, rejected_rewards

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None,
                        metric_key_prefix="eval"):
        self._metric_key_prefix = metric_key_prefix
        return super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)

    def store_metrics(self, metrics, train_eval="train"):
        if train_eval == "eval":  # "eval_rewards/margins" -> "eval_answer_eval_rewards/margins"
            metrics = {self._metric_key_prefix + key[len("eval"):]: value for key, value in metrics.items()}
        super().store_metrics(metrics, train_eval)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset_path", required=True,
                        help="DatasetDict saved by prepare_dpo_dataset.py (train split, optional test split).")
    parser.add_argument("--output_dir", required=True,
                        help="Directory for checkpoints and the final adapter.")

    parser.add_argument("--wrong_stop_multiplier", type=float, default=2.0,
                        help="lambda, the loss weight of wrong-stop pairs (1.0 gives standard DPO).")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta.")

    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="5e-5 for LoRA; the paper gives 5e-7 for full fine-tuning (--no-use_lora).")
    parser.add_argument("--max_length", type=int, default=8192, help="Maximum prompt + completion tokens.")
    parser.add_argument("--max_prompt_length", type=int, default=8000, help="Maximum prompt tokens.")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True,
                        help="Train with bf16 mixed precision.")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--base_dtype", choices=["float32", "bfloat16"], default="float32",
                        help="dtype in which the model weights are loaded (float32 as in the paper's runs; "
                             "with --deepspeed, bfloat16 halves host memory).")
    parser.add_argument("--deepspeed", default=None,
                        help="DeepSpeed config, e.g. training/ds_config_zero2.json (default: no DeepSpeed).")

    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj",
                        help="Comma-separated module names.")

    parser.add_argument("--eval_strategy", choices=["no", "steps", "epoch"], default="epoch",
                        help="When to compute validation metrics on the test split, per role if it has a "
                             "role column.")
    parser.add_argument("--eval_steps", type=int, default=500, help="Used with --eval_strategy steps.")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--report_to", default="tensorboard", help="tensorboard, wandb or none.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--local_rank", type=int, default=-1, help=argparse.SUPPRESS)  # set by the deepspeed launcher
    return parser.parse_args()


def count_wrong_stop_pairs(name, dataset, wrong_stop_multiplier, log):
    """Log how many pairs are wrong-stop pairs (weight lambda), per role if the dataset has a role column."""
    roles = dataset["role"] if "role" in dataset.column_names else ["all"] * len(dataset)
    pairs, wrong_stop = collections.Counter(roles), collections.Counter()
    for role, chosen, rejected in zip(roles, dataset["chosen"], dataset["rejected"]):
        wrong_stop[role] += is_wrong_stop_pair(chosen, rejected)
    for role in sorted(pairs):
        log(f"{name} {role}: {wrong_stop[role]} of {pairs[role]} pairs are wrong-stop pairs "
            f"(weight {wrong_stop_multiplier})")


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset = load_from_disk(args.dataset_path)
    train_dataset = add_loss_weights(dataset["train"], args.wrong_stop_multiplier)
    eval_dataset = dataset.get("test")
    if eval_dataset is not None:
        eval_dataset = add_loss_weights(eval_dataset, args.wrong_stop_multiplier)

    training_args = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False if args.use_lora else None,
        eval_strategy=args.eval_strategy if eval_dataset is not None else "no",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        seed=args.seed,
    )
    is_main_process = training_args.process_index == 0

    def log(message):
        if is_main_process:
            print(message, flush=True)

    count_wrong_stop_pairs("train", train_dataset, args.wrong_stop_multiplier, log)
    if eval_dataset is not None:
        count_wrong_stop_pairs("test", eval_dataset, args.wrong_stop_multiplier, log)
        if "role" in eval_dataset.column_names:
            roles = eval_dataset["role"]
            eval_dataset = {
                role: eval_dataset.select([i for i, r in enumerate(roles) if r == role]) for role in sorted(set(roles))
            }

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, dtype=getattr(torch, args.base_dtype), trust_remote_code=args.trust_remote_code
    )
    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules.split(","),
            bias="none",
            task_type="CAUSAL_LM",
        )

    trainer = AsymmetricDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    if args.use_lora and is_main_process:
        trainer.model.print_trainable_parameters()
    if trainer.is_world_process_zero():
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    trainer.train()
    trainer.save_model(args.output_dir)
    trainer.save_state()
    log(f"Saved the final model to {args.output_dir}")


if __name__ == "__main__":
    main()
