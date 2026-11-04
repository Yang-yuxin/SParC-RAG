"""HTTP client for the vLLM server in llm_server/serve.py (GET /generate/).

The server address comes from LLM_SERVER_HOST (e.g. http://localhost) and LLM_SERVER_PORT;
predict.py sets both from .llm_server_address.jsonnet unless they are already set. To send one
model to its own server, set <KEY>_LLM_SERVER_HOST and/or <KEY>_LLM_SERVER_PORT, where KEY is
the short model name with every character other than a letter or digit replaced by "_", e.g.
Qwen3_32B_LLM_SERVER_PORT=8011.
"""

import os
import re

import requests

from inference.prompt_readers import fit_prompt_into_given_limit

# Seconds to wait for a generation.
REQUEST_TIMEOUT = 1800


def short_model_name(model_name: str) -> str:
    """The server-side name of a model: an "org/" prefix (e.g. "Qwen/") is dropped."""
    return model_name.split("/")[1] if "/" in model_name else model_name


def server_address(model_name: str):
    """Return the (host, port) of the LLM server that serves ``model_name``."""
    key = re.sub(r"[^0-9A-Za-z]", "_", short_model_name(model_name))
    host = os.environ.get(f"{key}_LLM_SERVER_HOST", os.environ.get("LLM_SERVER_HOST"))
    port = os.environ.get(f"{key}_LLM_SERVER_PORT", os.environ.get("LLM_SERVER_PORT"))
    return host, port


class LLMClientGenerator:
    """Generate text with the model ``model_name`` served by llm_server/serve.py.

    Every prompt is first fitted into ``model_tokens_limit`` tokens together with ``max_length``
    generated tokens (see ``inference.prompt_readers.fit_prompt_into_given_limit``);
    ``num_truncated_prompts`` counts the prompts that had to be shortened. The server samples
    whenever ``temperature`` > 0.
    """

    def __init__(
        self,
        model_name,
        max_length=100,
        min_length=1,
        temperature=0.0,
        top_k=50,
        top_p=1.0,
        repetition_penalty=1.0,
        model_tokens_limit=2000,
        enable_thinking=False,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.min_length = min_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.model_tokens_limit = model_tokens_limit
        self.enable_thinking = enable_thinking
        self.num_truncated_prompts = 0

    def generate_text_sequence(self, prompt, stop=None):
        """Generate one completion of ``prompt``, stopping at ``stop`` (a string or a list).

        Returns ``[((text, num_output_tokens), score)]``. ``num_output_tokens`` counts the
        completion tokens only; ``score`` is 1.0.
        """
        prompt, truncated = fit_prompt_into_given_limit(
            original_prompt=prompt.rstrip(),
            model_length_limit=self.model_tokens_limit,
            estimated_generation_length=self.max_length,
            tokenizer_model_name=self.model_name,
        )
        self.num_truncated_prompts += truncated

        params = {
            "prompt": prompt,
            "max_length": self.max_length,
            "min_length": self.min_length,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "enable_thinking": self.enable_thinking,
        }
        if stop is not None:
            params["stop"] = [stop] if isinstance(stop, str) else stop

        model_name = short_model_name(self.model_name)
        host, port = server_address(self.model_name)
        response = requests.get(
            host + ":" + str(port) + "/generate/", params=params, timeout=REQUEST_TIMEOUT
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"LLM generation request failed ({response.status_code}): {response.text[:500]}"
            )
        result = response.json()
        if result.get("model_name") != model_name:
            raise RuntimeError(
                f"The LLM server serves {result.get('model_name')!r}, not {model_name!r}; "
                "check the MODEL_NAME it was started with."
            )

        text = result["generated_texts"][0]
        if text.startswith(prompt):
            text = text[len(prompt) :]
        return [((text, result["output_token_counts"][0]), 1.0)]
