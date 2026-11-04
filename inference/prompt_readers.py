"""Prompt files and client-side prompt-length control."""

import os
import sys
from functools import lru_cache
from typing import Tuple

# Tokenizers used to measure prompts on the client side, by the short model names that the
# configs use. Fine-tuned checkpoints share the tokenizer of their base model.
TOKENIZERS = {
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen3-32B": "Qwen/Qwen3-32B",
    "Qwen2.5-7B-Instruct-SPARC-DPO": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen3-32B-SPARC-DPO": "Qwen/Qwen3-32B",
}

# Fallback for other short names: the base tokenizer of the model family the name starts with.
FAMILY_TOKENIZERS = {
    "Qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen3": "Qwen/Qwen3-32B",
}


def resolve_tokenizer_name(model_name: str) -> str:
    """Return the Hugging Face id or local path of the tokenizer for ``model_name``.

    Names containing "/" (Hugging Face ids such as ``Qwen/Qwen3-32B``, or local paths) are used
    as they are. Other names are looked up in TOKENIZERS, then by model family, and finally
    fall back to $MODEL_PATH, the checkpoint served under a custom short name.
    """
    if "/" in model_name:
        return model_name
    if model_name in TOKENIZERS:
        return TOKENIZERS[model_name]
    for prefix, tokenizer_name in FAMILY_TOKENIZERS.items():
        if model_name.startswith(prefix):
            return tokenizer_name
    if os.environ.get("MODEL_PATH"):
        return os.environ["MODEL_PATH"]
    raise ValueError(
        f"No tokenizer known for model {model_name!r}. Use a Hugging Face id or a path as "
        "model_name, or set MODEL_PATH to the checkpoint served under this name."
    )


@lru_cache(maxsize=15)
def get_tokenizer(model_name):
    from transformers import AutoTokenizer

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return AutoTokenizer.from_pretrained(resolve_tokenizer_name(model_name))


def read_prompt(file_path: str) -> str:
    """Return the prompt template stored in ``file_path``, without surrounding whitespace."""
    with open(file_path, "r") as file:
        return file.read().strip()


def fit_prompt_into_given_limit(
    original_prompt: str,
    model_length_limit: int,
    estimated_generation_length: int,
    tokenizer_model_name: str,
    block_delimiter: str = "\n\n\n",
) -> Tuple[str, bool]:
    """Shorten a prompt so that prompt + generation fits ``model_length_limit`` tokens.

    Every prompt sent to the LLM goes through this function. The prompt is split into blocks at
    ``block_delimiter``; each block is stripped and empty blocks are dropped, and the blocks are
    joined again with the delimiter. While prompt + ``estimated_generation_length`` reaches the
    limit, the first block is dropped (the last block is always kept). If prompt + generation
    still exceeds the limit, lines are dropped from the top of the prompt, always at least one,
    until the remaining text alone fits the limit (the generation budget is not reserved here).

    Returns the prompt and whether any block or line was dropped.
    """
    blocks = [
        block.strip()
        for block in original_prompt.strip().split(block_delimiter)
        if block.strip()
    ]
    tokenizer = get_tokenizer(tokenizer_model_name)
    block_sizes = [len(tokenizer.tokenize(block)) for block in blocks]

    if blocks:
        last_block = blocks.pop(-1)
        last_block_size = block_sizes.pop(-1)
    else:
        last_block = ""
        last_block_size = 0

    num_dropped_blocks = 0
    while True:
        updated_length = (
            sum(block_sizes) + last_block_size + estimated_generation_length
        )
        if updated_length < model_length_limit or not block_sizes:
            break
        blocks.pop(0)
        block_sizes.pop(0)
        num_dropped_blocks += 1

    if num_dropped_blocks:
        print(
            f"WARNING: Dropped {num_dropped_blocks} leading block(s) of the prompt to fit "
            f"the model length limit ({model_length_limit} tokens).",
            file=sys.stderr,
        )

    updated_prompt = block_delimiter.join(blocks + ([last_block] if last_block else []))

    trimmed_lines = False
    if updated_length > model_length_limit:
        print(
            "WARNING: The updated prompt is still longer than the model length limit. Trimming it further.",
            file=sys.stderr,
        )
        updated_lines = updated_prompt.split("\n")
        while updated_lines:
            updated_lines.pop(0)
            trimmed_lines = True
            if len(tokenizer.tokenize("\n".join(updated_lines))) <= model_length_limit:
                break
        updated_prompt = "\n".join(updated_lines)

    return updated_prompt, bool(num_dropped_blocks) or trimmed_lines
