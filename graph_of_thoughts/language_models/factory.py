import json
import os

from .chatgpt import ChatGPT
from .local_hf_model import LocalHFModel


def _load_model_block(config_path: str, model_name: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if model_name not in config:
        raise KeyError(f"Model '{model_name}' not found in {config_path}")
    block = config[model_name]
    if not isinstance(block, dict):
        raise ValueError(f"Model config '{model_name}' should be a JSON object")
    return block


def create_language_model(
    config_path: str = "",
    model_name: str = "chatgpt",
    cache: bool = False,
):
    """
    Build language model instance based on config backend.
    Supported backends:
    - remote_openai_compatible
    - local_hf
    """
    if config_path == "":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.json")

    block = _load_model_block(config_path, model_name)
    backend = str(block.get("backend", "remote_openai_compatible")).strip().lower()

    if backend in {"remote_openai_compatible", "openai_compatible", "chatgpt"}:
        return ChatGPT(config_path=config_path, model_name=model_name, cache=cache)
    if backend in {"local_hf", "hf_local", "qwen_local"}:
        return LocalHFModel(config_path=config_path, model_name=model_name, cache=cache)

    raise ValueError(
        f"Unsupported backend '{backend}' for model '{model_name}'. "
        "Supported: remote_openai_compatible, local_hf."
    )
