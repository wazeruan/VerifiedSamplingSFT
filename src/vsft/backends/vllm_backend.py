"""vLLM backend (Linux + CUDA). Install with `uv sync --extra vllm`."""

from __future__ import annotations

import dataclasses
import os

from vsft.backends.base import Backend, Generation


def _hf_config(model_cfg):
    try:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(
            model_cfg.name_or_path, trust_remote_code=model_cfg.trust_remote_code
        )
    except Exception:  # noqa: BLE001 - unknown config: callers assume the plain case
        return None


def _is_multimodal(model_cfg, cfg=None) -> bool:
    cfg = cfg if cfg is not None else _hf_config(model_cfg)
    if cfg is None:
        return False
    archs = getattr(cfg, "architectures", None) or []
    return any("ConditionalGeneration" in a for a in archs) or hasattr(cfg, "vision_config")


def _has_linear_attention(cfg) -> bool:
    """True for hybrid Gated-DeltaNet models such as Qwen3.5."""
    if cfg is None:
        return False
    text = getattr(cfg, "text_config", None) or cfg
    return "linear_attention" in (getattr(text, "layer_types", None) or [])


def _engine_arg_supported(name: str) -> bool:
    try:
        from vllm.engine.arg_utils import EngineArgs

        return name in {f.name for f in dataclasses.fields(EngineArgs)}
    except Exception:  # noqa: BLE001
        return False


class VLLMBackend(Backend):
    def __init__(self, model_cfg, backend_cfg):
        if not backend_cfg.jit_kernels:
            # No nvcc on compute nodes: keep vLLM off kernels FlashInfer JIT-compiles.
            # Set before vllm is imported; the engine-core subprocess inherits it.
            os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        from vllm import LLM

        hf_cfg = _hf_config(model_cfg)
        kwargs = {}
        if (
            not backend_cfg.jit_kernels
            and _has_linear_attention(hf_cfg)
            and _engine_arg_supported("gdn_prefill_backend")
        ):
            kwargs["gdn_prefill_backend"] = "triton"  # FlashInfer GDN prefill needs nvcc
        if _is_multimodal(model_cfg, hf_cfg):
            # Text-only use of a multimodal checkpoint (e.g. Qwen3.5): no image/video inputs,
            # so vLLM does not profile or reserve memory for the vision encoder.
            kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        self.llm = LLM(
            model=model_cfg.name_or_path,
            dtype=model_cfg.dtype,
            max_model_len=model_cfg.max_model_len,
            gpu_memory_utilization=backend_cfg.gpu_memory_utilization,
            tensor_parallel_size=backend_cfg.tensor_parallel_size,
            # V1 chunks long prefills, so this may be below max_model_len; it bounds the
            # per-step prompt_logprobs logits (tokens x 248k vocab for Qwen3.5).
            max_num_batched_tokens=backend_cfg.max_num_batched_tokens,
            max_num_seqs=backend_cfg.max_num_seqs,
            trust_remote_code=model_cfg.trust_remote_code,
            seed=backend_cfg.seed,
            **kwargs,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.max_len = model_cfg.max_model_len

    @staticmethod
    def _prompt(ids: list[int]):
        from vllm.inputs import TokensPrompt

        return TokensPrompt(prompt_token_ids=ids)

    def generate(self, prompts, n, temperature, top_p, max_tokens):
        from vllm import SamplingParams

        out: list[list[Generation] | None] = [None] * len(prompts)
        reqs, params, idx = [], [], []
        for i, p in enumerate(prompts):
            ids = self.tokenizer(p, add_special_tokens=False)["input_ids"]
            if len(ids) >= self.max_len:
                # No room to generate: empty, unfinished outputs instead of a batch crash.
                out[i] = [Generation("", False) for _ in range(n)]
                continue
            reqs.append(self._prompt(ids))
            params.append(
                SamplingParams(
                    n=n,
                    temperature=temperature,
                    top_p=top_p if temperature > 0 else 1.0,
                    top_k=-1,
                    repetition_penalty=1.0,
                    max_tokens=min(max_tokens, self.max_len - len(ids)),
                )
            )
            idx.append(i)
        if reqs:
            outs = self.llm.generate(reqs, params, use_tqdm=False)
            for i, o in zip(idx, outs, strict=True):
                out[i] = [Generation(c.text, c.finish_reason == "stop") for c in o.outputs]
        return out

    def token_logprobs(self, prompts, responses):
        from vllm import SamplingParams

        results: list[list[float]] = [[] for _ in prompts]
        reqs, spans, idx = [], [], []
        for i, (p, r) in enumerate(zip(prompts, responses, strict=True)):
            p_ids = self.tokenizer(p, add_special_tokens=False)["input_ids"]
            r_ids = self.tokenizer(r, add_special_tokens=False)["input_ids"]
            # vLLM needs one free slot for the (discarded) generated token.
            r_ids = r_ids[: max(0, self.max_len - 1 - len(p_ids))]
            if not r_ids or not p_ids:
                continue
            reqs.append(self._prompt(p_ids + r_ids))
            spans.append((len(p_ids), r_ids))
            idx.append(i)
        if reqs:
            params = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
            outs = self.llm.generate(reqs, params, use_tqdm=False)
            for i, o, (start, r_ids) in zip(idx, outs, spans, strict=True):
                plp = o.prompt_logprobs
                results[i] = [plp[start + t][tid].logprob for t, tid in enumerate(r_ids)]
        return results
