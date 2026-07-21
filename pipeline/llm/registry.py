"""
pipeline/llm/registry.py — Model registry and client factory.

MODELS is the authoritative list of supported models.
get_client(model_name, **kwargs) instantiates the right LLMClient subclass.

Design rule: enable_thinking is ALWAYS False for every model in this pipeline.
It is enforced here at the registry level — no tool or stage should override it.
"""

from __future__ import annotations

from pipeline.llm.base import LLMClient


MODELS: dict[str, dict] = {
    "qwen3.5-9b": {
        "client":             "vllm",
        "model_id":           "Qwen/Qwen3.5-9B",
        "enable_thinking":    False,   # MUST remain False — thinking tokens break downstream parsing
        "max_tokens_default": 1024,
        # Qwen3 requires this vLLM extension to suppress thinking-mode preambles.
        "vllm_extra_payload": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    "gemma4-e4b": {
        "client":             "vllm",
        "model_id":           "google/gemma-4-E4B-it",
        "enable_thinking":    False,
        "max_tokens_default": 1024,
        # -it model has a built-in chat template and correct EOS handling; no extra payload needed.
        "vllm_extra_payload": {},
    },
    # Add new models here following the same pattern.
    # "my-new-model": {
    #     "client":  "vllm",          # "vllm" | "hf"
    #     "model_id": "org/model",
    #     "enable_thinking": False,
    #     "max_tokens_default": 1024,
    #     "vllm_extra_payload": {},   # optional: extra fields merged into every request payload
    # },
}


def get_client(model_name: str, **kwargs) -> LLMClient:
    """
    Instantiate an LLMClient for the given model name.

    Args:
        model_name: Key in MODELS dict (e.g. "qwen3.5-9b").
        **kwargs:   Forwarded to the client constructor (e.g. base_url, use_4bit).

    Returns:
        An LLMClient instance ready to call .generate().

    Raises:
        ValueError: If model_name is not in MODELS.
    """
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model: {model_name!r}. Available: {sorted(MODELS)}"
        )

    config      = MODELS[model_name]
    client_type = config["client"]
    model_id    = config["model_id"]

    if client_type == "vllm":
        from pipeline.llm.vllm_client import VLLMClient
        extra = config.get("vllm_extra_payload", {})
        return VLLMClient(model_id=model_id, extra_payload=extra, **kwargs)

    if client_type == "hf":
        from pipeline.llm.hf_client import HFClient
        return HFClient(model_id=model_id, **kwargs)

    raise ValueError(f"Unknown client type: {client_type!r} for model {model_name!r}")
