import os
from typing import Any, Dict, List, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .abstract_language_model import AbstractLanguageModel


def _resolve_torch_dtype(dtype_name: str):
    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = str(dtype_name or "auto").strip().lower()
    return mapping.get(key, "auto")


class LocalHFModel(AbstractLanguageModel):
    """
    Local HuggingFace CausalLM backend (Qwen-compatible).
    """

    def __init__(
        self, config_path: str = "", model_name: str = "qwen_local_4b", cache: bool = False
    ) -> None:
        super().__init__(config_path, model_name, cache)
        self.config: Dict[str, Any] = self.config[model_name]

        self.model_path: str = self.config.get("model_path", "")
        if not self.model_path:
            raise ValueError(f"{model_name}.model_path is required for local_hf backend")

        self.tokenizer_path: str = self.config.get("tokenizer_path", self.model_path)
        self.device_map: Union[str, Dict[str, Any]] = self.config.get("device_map", "auto")
        self.torch_dtype = _resolve_torch_dtype(self.config.get("torch_dtype", "auto"))
        self.max_new_tokens: int = int(self.config.get("max_new_tokens", 512))
        self.temperature: float = float(self.config.get("temperature", 0.7))
        self.top_p: float = float(self.config.get("top_p", 0.9))
        self.do_sample: bool = bool(self.config.get("do_sample", True))
        self.stop: Union[str, List[str], None] = self.config.get("stop")

        self.prompt_token_cost: float = float(self.config.get("prompt_token_cost", 0.0))
        self.response_token_cost: float = float(self.config.get("response_token_cost", 0.0))

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=True,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            use_safetensors=True,
        )
        self.model.eval()

    def _apply_chat_template(self, query: str) -> str:
        messages = [{"role": "user", "content": query}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return query

    def _truncate_at_stop(self, text: str) -> str:
        if not self.stop:
            return text
        stops = [self.stop] if isinstance(self.stop, str) else list(self.stop)
        end = len(text)
        for stop_token in stops:
            idx = text.find(stop_token)
            if idx != -1:
                end = min(end, idx)
        return text[:end]

    def query(self, query: str, num_responses: int = 1) -> Dict[str, Any]:
        if self.cache and query in self.response_cache:
            cached = self.response_cache[query]
            self.record_query_event(
                prompt=query,
                responses=self.get_response_texts(cached),
                num_responses=num_responses,
                prompt_tokens_delta=0,
                completion_tokens_delta=0,
                cost_delta=0.0,
                meta={"cache_hit": True, "backend": "local_hf"},
            )
            return cached

        prompt_text = self._apply_chat_template(query)
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_token_count = int(inputs["input_ids"].shape[-1])

        before_prompt_tokens = self.prompt_tokens
        before_completion_tokens = self.completion_tokens
        before_cost = self.cost

        generations: List[str] = []
        for _ in range(int(num_responses)):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            generated_ids = outputs[0][prompt_token_count:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generations.append(self._truncate_at_stop(text))

            completion_tokens = int(generated_ids.shape[-1])
            self.prompt_tokens += prompt_token_count
            self.completion_tokens += completion_tokens

        self.cost = (
            self.prompt_token_cost * (float(self.prompt_tokens) / 1000.0)
            + self.response_token_cost * (float(self.completion_tokens) / 1000.0)
        )

        result = {
            "prompt": query,
            "responses": generations,
        }
        if self.cache:
            self.response_cache[query] = result

        self.record_query_event(
            prompt=query,
            responses=generations,
            num_responses=num_responses,
            prompt_tokens_delta=max(0, self.prompt_tokens - before_prompt_tokens),
            completion_tokens_delta=max(0, self.completion_tokens - before_completion_tokens),
            cost_delta=max(0.0, self.cost - before_cost),
            meta={
                "cache_hit": False,
                "backend": "local_hf",
                "model_path": os.path.abspath(self.model_path),
            },
        )
        return result

    def get_response_texts(
        self, query_responses: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> List[str]:
        if isinstance(query_responses, dict):
            return [str(x) for x in query_responses.get("responses", [])]
        texts: List[str] = []
        for item in query_responses:
            texts.extend([str(x) for x in item.get("responses", [])])
        return texts
