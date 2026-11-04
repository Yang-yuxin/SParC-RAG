"""vLLM server for SPARC-RAG (GET /generate/), used by models/llm_client_generator.py.

Start it with, for example,
    MODEL_NAME=Qwen2.5-7B-Instruct uvicorn serve:app --app-dir llm_server --port 8010
uvicorn listens on 127.0.0.1; add --host 0.0.0.0 only to serve other machines (the server has no
authentication).

MODEL_NAME is the short name the configs use as ``model_name`` (the client drops an "org/"
prefix such as "Qwen/"). The models of the paper are listed in MODELS with the vLLM settings and
the prompt format of the paper's runs:
  - Qwen2.5-7B-Instruct and Qwen3-32B load from Hugging Face;
  - the fine-tuned Qwen2.5-7B-Instruct-SPARC-DPO and Qwen3-32B-SPARC-DPO load from MODEL_PATH,
    a merged checkpoint directory (see training/).
MODEL_PATH also replaces the weights of a listed model, and serves any other checkpoint under
any other MODEL_NAME, with the default vLLM settings; for such models CHAT_TEMPLATE=1/0 selects
the prompt format (default: chat template iff MODEL_NAME contains "Qwen3").

Other environment variables: TP_SIZE (tensor-parallel size; default: all visible GPUs) and
ENGINE_ARGS, a JSON object of vLLM engine arguments that override the model's settings, e.g.
ENGINE_ARGS='{"gpu_memory_utilization": 0.8, "max_model_len": 16384}'.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import List, Optional

import torch
from fastapi import FastAPI, Query
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'llm_server_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

TP_SIZE = int(os.environ.get("TP_SIZE", torch.cuda.device_count()))

# vLLM engine settings of the paper runs.
DEFAULT_ENGINE_ARGS = dict(dtype="auto", max_model_len=32768)
QWEN3_32B_ENGINE_ARGS = dict(
    dtype="bfloat16",
    max_model_len=16384,
    gpu_memory_utilization=0.92,
    enforce_eager=False,
    enable_prefix_caching=True,
    max_num_seqs=8,
    swap_space=4,
    enable_chunked_prefill=True,
)


@dataclass(frozen=True)
class ModelSpec:
    path: Optional[str]  # Hugging Face id; None: the weights must come from MODEL_PATH
    # True: the prompt goes through the tokenizer's chat template as one user message (llm.chat).
    # False: the raw prompt is completed (llm.generate).
    chat_template: bool
    engine_args: dict


MODELS = {
    # Qwen2.5 models were prompted with raw text, without a chat template.
    "Qwen2.5-7B-Instruct": ModelSpec("Qwen/Qwen2.5-7B-Instruct", False, DEFAULT_ENGINE_ARGS),
    "Qwen2.5-7B-Instruct-SPARC-DPO": ModelSpec(None, False, DEFAULT_ENGINE_ARGS),
    # Qwen3 models use the chat template; thinking follows each request's enable_thinking
    # (false in all configs).
    "Qwen3-32B": ModelSpec("Qwen/Qwen3-32B", True, QWEN3_32B_ENGINE_ARGS),
    # The paper's runs served the fine-tuned 32B model with the default engine settings.
    "Qwen3-32B-SPARC-DPO": ModelSpec(None, True, DEFAULT_ENGINE_ARGS),
}

# generate() truncates prompts longer than PROMPT_GUARD_MAX_MODEL_LEN - max_length tokens. The
# paper's runs used this bound for every model.
PROMPT_GUARD_MAX_MODEL_LEN = 32768


def resolve_model(model_shortname: str):
    """Return (weights path or HF id, chat_template, engine_args) for MODEL_NAME."""
    model_path = os.environ.get("MODEL_PATH")
    spec = MODELS.get(model_shortname)
    if spec is None:
        if not model_path:
            raise ValueError(
                f"Unknown MODEL_NAME {model_shortname!r}. Known: {sorted(MODELS)}; "
                "set MODEL_PATH to serve another checkpoint under this name."
            )
        chat_template = os.environ.get("CHAT_TEMPLATE")
        if chat_template is None:
            use_chat_template = "Qwen3" in model_shortname
        else:
            use_chat_template = chat_template == "1"
        return model_path, use_chat_template, DEFAULT_ENGINE_ARGS
    if model_path is None and spec.path is None:
        raise ValueError(
            f"'{model_shortname}' is a fine-tuned checkpoint; set MODEL_PATH to the merged "
            "checkpoint directory (see training/) to serve it."
        )
    return model_path or spec.path, spec.chat_template, spec.engine_args


MODEL_NAME = os.environ.get("MODEL_NAME")
if not MODEL_NAME:
    raise RuntimeError("Set MODEL_NAME to the short name of the model to serve.")
MODEL_PATH, USE_CHAT_TEMPLATE, ENGINE_ARGS = resolve_model(MODEL_NAME)
ENGINE_ARGS = {**ENGINE_ARGS, **json.loads(os.environ.get("ENGINE_ARGS", "{}"))}

request_counter = 0


@lru_cache(maxsize=None)
def get_model_and_tokenizer():
    """Load the model with vLLM (tensor parallelism over TP_SIZE GPUs) and its tokenizer."""
    logger.info(f"Loading {MODEL_PATH} with tensor parallelism size {TP_SIZE} and {ENGINE_ARGS}")
    llm = LLM(model=MODEL_PATH, tensor_parallel_size=TP_SIZE, **ENGINE_ARGS)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    logger.info(f"Successfully loaded model: {MODEL_PATH}")
    return llm, tokenizer


app = FastAPI()


@app.get("/")
async def index():
    return {"message": f"Hello! This is a vLLM server for {MODEL_NAME}. Use /generate/ for text generation."}


@app.get("/generate/")
async def generate(
    prompt: str,
    max_length: int = 200,
    min_length: int = 1,
    temperature: float = 0.0,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    stop: List[str] = Query(default=None),
    enable_thinking: bool = False,
):
    """Generate one completion of ``prompt`` (``max_length`` = maximum new tokens).

    Returns the generated text with the numbers of prompt and completion tokens.
    """
    global request_counter

    request_id = f"req_{request_counter}_{time.time()}"
    request_counter += 1
    start_time = time.time()

    llm, tokenizer = get_model_and_tokenizer()

    # Keep the beginning of over-long prompts.
    encoded = tokenizer.encode(prompt)
    if len(encoded) > PROMPT_GUARD_MAX_MODEL_LEN - max_length:
        max_prompt_tokens = PROMPT_GUARD_MAX_MODEL_LEN - max_length - 100
        logger.warning(f"[{request_id}] Truncated a prompt of {len(encoded)} tokens to {max_prompt_tokens}")
        prompt = tokenizer.decode(encoded[:max_prompt_tokens])

    sampling_params = SamplingParams(
        max_tokens=max_length,
        min_tokens=min_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop=stop,
    )

    if USE_CHAT_TEMPLATE:
        outputs = llm.chat(
            [{"role": "user", "content": prompt}],
            sampling_params=sampling_params,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )
    else:
        outputs = llm.generate([prompt], sampling_params)

    generated_texts = [output.outputs[0].text for output in outputs]
    input_token_counts = [len(output.prompt_token_ids) for output in outputs]
    output_token_counts = [len(output.outputs[0].token_ids) for output in outputs]

    run_time = time.time() - start_time
    logger.info(
        f"[{request_id}] Completed in {run_time:.2f}s; "
        f"tokens: in={input_token_counts[0]}, out={output_token_counts[0]}"
    )

    return {
        "generated_texts": generated_texts,
        "input_token_counts": input_token_counts,
        "output_token_counts": output_token_counts,
        "run_time_in_seconds": run_time,
        "model_name": MODEL_NAME,
        "request_id": request_id,
    }


logger.info("Loading model and tokenizer with vLLM...")
get_model_and_tokenizer()
logger.info("Model and tokenizer loaded successfully.")
