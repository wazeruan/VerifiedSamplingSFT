"""Backend interface shared by the HF and vLLM implementations.

The same backend (one set of weights) plays three roles: proposal sampler, student scorer
and verifier. Prompts are passed as fully rendered strings so that a partial response can
simply be appended after the assistant header and continued.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Generation:
    text: str
    finished: bool  # True if the model emitted EOS before max_tokens


def render_chat(tokenizer, messages: list[dict[str, str]], assistant_prefix: str = "") -> str:
    """Chat template + generation header (+ partial reply). Shared by backends and SFT."""
    if getattr(tokenizer, "chat_template", None):
        # enable_thinking=False keeps reasoning templates (e.g. Qwen3) from opening a
        # <think> block; templates that do not use the variable ignore it.
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    else:  # plain base model without a chat template
        parts = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
        text = "\n\n".join(parts) + "\n\nAssistant: "
    return text + assistant_prefix


class Backend(ABC):
    tokenizer: object

    def render(self, messages: list[dict[str, str]], assistant_prefix: str = "") -> str:
        """Render chat messages with the generation header, then append a partial reply."""
        return render_chat(self.tokenizer, messages, assistant_prefix)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        n: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> list[list[Generation]]:
        """Return `n` sampled continuations per prompt (temperature 0 = greedy)."""

    @abstractmethod
    def token_logprobs(self, prompts: list[str], responses: list[str]) -> list[list[float]]:
        """Per-token log p(response_t | prompt, response_<t) for each pair."""


def mean_logprob(logprobs: list[float]) -> float:
    return sum(logprobs) / len(logprobs) if logprobs else float("-inf")


def build_backend(model_cfg, backend_cfg) -> Backend:
    if backend_cfg.kind == "hf":
        from vsft.backends.hf import HFBackend

        return HFBackend(model_cfg, backend_cfg)
    if backend_cfg.kind == "vllm":
        from vsft.backends.vllm_backend import VLLMBackend

        return VLLMBackend(model_cfg, backend_cfg)
    raise ValueError(f"unknown backend {backend_cfg.kind!r}")
