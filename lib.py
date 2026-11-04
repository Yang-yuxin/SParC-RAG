"""Small helpers shared by predict.py and the training scripts."""

import json
import os
from typing import Dict, List

import _jsonnet


def get_llm_server_address() -> Dict[str, str]:
    """Return {"host": ..., "port": ...} of the LLM server as strings.

    LLM_SERVER_HOST and LLM_SERVER_PORT take precedence over .llm_server_address.jsonnet.
    """
    address = {
        "host": os.environ.get("LLM_SERVER_HOST"),
        "port": os.environ.get("LLM_SERVER_PORT"),
    }
    if address["host"] is None or address["port"] is None:
        config_path = ".llm_server_address.jsonnet"
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"LLM server address file ({config_path}) not available and "
                "LLM_SERVER_HOST/LLM_SERVER_PORT environment variables not set."
            )
        config = json.loads(_jsonnet.evaluate_file(config_path))
        for key in ("host", "port"):
            if address[key] is None:
                address[key] = config[key]
    return {key: str(value) for key, value in address.items()}


def read_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r") as file:
        instances = [json.loads(line.strip()) for line in file.readlines() if line.strip()]
    return instances
