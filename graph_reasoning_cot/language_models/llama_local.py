"""
language_models/llama_local.py
===============================
GoT-compatible LM adapter for local LLaMA-2-7B-Chat.

ROOT CAUSE FIX
--------------
GoT's AbstractLanguageModel.__init__ unconditionally calls self.load_config(config_path),
which tries to open config_path as a JSON file. Passing "" or a non-existent path causes
FileNotFoundError. 

Fix: We do NOT call super().__init__() at all. Instead we manually set every attribute
that GoT's Controller reads from a language model:
    - self.config         (dict)
    - self.model_name     (str)
    - self.cache          (bool)
We implement the full AbstractLanguageModel interface ourselves.

GPU Setup (from nvidia-smi)
----------------------------
  GPU 0: Tesla V100-SXM2-32GB  →  used for I/O baseline
  GPU 1: Tesla V100-SXM2-32GB  →  reserved for CoT
  Both V100s: Compute Capability 7.0, NO BF16 support → must use FP16

config.json keys
----------------
{
    "model_path":       "/home/wangzq28/GoT-Experiment/models/LLaMA2-7B-chat",
    "backend":          "vllm",       // "vllm" | "hf"
    "max_new_tokens":   256,
    "temperature":      0.0,
    "top_p":            1.0,
    "gpu_device":       0             // which GPU to use (0 for I/O, 1 for CoT)
}
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── LLaMA-2-Chat prompt template ──────────────────────────────────────────
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS,  E_SYS  = "<<SYS>>\n", "\n<</SYS>>\n\n"
_SYSTEM = (
    "You are a helpful assistant that solves graph reasoning problems. "
    "Answer concisely and correctly."
)

def format_llama2_chat(user_msg: str, system: str = _SYSTEM) -> str:
    return f"{B_INST} {B_SYS}{system}{E_SYS}{user_msg.strip()} {E_INST}"


class LlamaLocal:
    """
    Local LLaMA-2-7B-Chat adapter.

    Intentionally does NOT call super().__init__() to avoid GoT's
    AbstractLanguageModel.load_config() which requires a valid JSON file path.
    Instead we implement the full interface manually.

    Compatible with GoT Controller: implements query() and get_response_texts().
    """

    DEFAULT_MODEL_PATH = "/home/wangzq28/GoT-Experiment/models/LLaMA2-7B-chat"

    def __init__(self, config_path: str = ""):
        # ── Manually set attributes GoT Controller reads ──────────────────
        self.model_name = "llama2-7b-chat"
        self.cache      = False
        self.config: Dict = {}

        # ── Token counters ────────────────────────────────────────────────
        self.prompt_tokens     = 0
        self.completion_tokens = 0

        # ── Engine (lazy-loaded on first query) ───────────────────────────
        self._engine = None

        # ── Load config ───────────────────────────────────────────────────
        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)
        else:
            if config_path and not os.path.exists(config_path):
                logger.warning(
                    "Config file not found: %s — using built-in defaults.", config_path
                )
            self._set_defaults()

    # ── Config ────────────────────────────────────────────────────────────

    def _set_defaults(self):
        self.model_path     = self.DEFAULT_MODEL_PATH
        self.backend        = "vllm"
        self.max_new_tokens = 256
        self.temperature    = 0.0
        self.top_p          = 1.0
        self.gpu_device     = 0       # GPU 0 for I/O baseline
        logger.info("Using built-in defaults | model=%s | gpu=%d",
                    self.model_path, self.gpu_device)

    def _load_from_file(self, config_path: str):
        with open(config_path) as f:
            cfg = json.load(f)
        self.config         = cfg
        self.model_path     = cfg.get("model_path",     self.DEFAULT_MODEL_PATH)
        self.backend        = cfg.get("backend",        "vllm").lower()
        self.max_new_tokens = cfg.get("max_new_tokens", 256)
        self.temperature    = cfg.get("temperature",    0.0)
        self.top_p          = cfg.get("top_p",          1.0)
        self.gpu_device     = cfg.get("gpu_device",     0)
        logger.info("Config loaded: %s | backend=%s | gpu=%d",
                    config_path, self.backend, self.gpu_device)

    # ── Lazy engine initialisation ─────────────────────────────────────────

    def _init_engine(self):
        if self._engine is not None:
            return
        if self.backend == "vllm":
            self._init_vllm()
        elif self.backend == "hf":
            self._init_hf()
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}. Use 'vllm' or 'hf'.")

    def _init_vllm(self):
        try:
            from vllm import LLM, SamplingParams          # type: ignore
        except ImportError as e:
            raise ImportError(
                "vLLM not installed. Run: pip install vllm==0.4.3"
            ) from e

        # Pin to specific GPU via CUDA_VISIBLE_DEVICES
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_device)
        logger.info("Initialising vLLM on GPU %d (FP16, V100) …", self.gpu_device)

        self._engine = LLM(
            model=self.model_path,
            dtype="float16",              # V100 CC7.0 — no BF16
            max_model_len=4096,
            gpu_memory_utilization=0.90,
        )
        self._SamplingParams = SamplingParams
        logger.info("vLLM engine ready on GPU %d.", self.gpu_device)

    def _init_hf(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError as e:
            raise ImportError("transformers not installed.") from e

        logger.info("Initialising HuggingFace pipeline on GPU %d …", self.gpu_device)
        device = f"cuda:{self.gpu_device}" if self.gpu_device >= 0 else "cpu"
        tok = AutoTokenizer.from_pretrained(self.model_path)
        mdl = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=__import__("torch").float16,
            device_map={"": device},
        )
        self._engine        = pipeline("text-generation", model=mdl, tokenizer=tok)
        self._hf_tokenizer  = tok
        logger.info("HuggingFace pipeline ready.")

    # ── Core query (GoT Controller calls this) ────────────────────────────

    def query(self, query: str, num_responses: int = 1) -> List[str]:
        """
        Send `query` to LLaMA-2-Chat and return `num_responses` strings.
        Wraps the raw prompt in the LLaMA-2-Chat [INST] template.
        """
        self._init_engine()
        prompt = format_llama2_chat(query)
        t0 = time.time()
        if self.backend == "vllm":
            responses = self._query_vllm(prompt, num_responses)
        else:
            responses = self._query_hf(prompt, num_responses)
        logger.debug("query %.2fs | n=%d", time.time() - t0, num_responses)
        return responses

    def _query_vllm(self, prompt: str, n: int) -> List[str]:
        temp = self.temperature if n == 1 else max(self.temperature, 0.7)
        sp   = self._SamplingParams(
            n=n, temperature=temp, top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )
        out = self._engine.generate([prompt], sp)[0]
        self.prompt_tokens     += len(out.prompt_token_ids)
        self.completion_tokens += sum(len(o.token_ids) for o in out.outputs)
        return [o.text.strip() for o in out.outputs]

    def _query_hf(self, prompt: str, n: int) -> List[str]:
        temp      = self.temperature if n == 1 else max(self.temperature, 0.7)
        do_sample = temp > 0.0
        outs = self._engine(
            prompt,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=n,
            temperature=temp if do_sample else 1.0,
            top_p=self.top_p  if do_sample else 1.0,
            do_sample=do_sample,
            return_full_text=False,
        )
        enc = self._hf_tokenizer(prompt, return_tensors="pt")
        self.prompt_tokens += enc["input_ids"].shape[-1] * n
        texts = [o["generated_text"].strip() for o in outs]
        for t in texts:
            self.completion_tokens += len(self._hf_tokenizer(t)["input_ids"])
        return texts

    # ── GoT interface ─────────────────────────────────────────────────────

    def get_response_texts(self, responses: List[str]) -> List[str]:
        """Identity — query() already returns plain strings."""
        return responses

    # ── Token accounting ──────────────────────────────────────────────────

    def token_usage(self) -> Dict[str, int]:
        return {
            "prompt_tokens":     self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens":      self.prompt_tokens + self.completion_tokens,
        }

    def reset_token_counters(self):
        self.prompt_tokens     = 0
        self.completion_tokens = 0