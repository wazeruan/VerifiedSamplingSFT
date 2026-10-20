"""VLLMBackend JIT-kernel gating: FlashInfer sampler env var, GDN prefill backend, and
limit_mm_per_prompt. vllm is not installed here, so a fake package is injected into
sys.modules; the HF config lookup is monkeypatched to return fake configs."""

from __future__ import annotations

import dataclasses
import os
import sys
import types

import pytest

from vsft.backends import vllm_backend
from vsft.backends.vllm_backend import VLLMBackend
from vsft.config import BackendConfig, ModelConfig, load_config

ENV = "VLLM_USE_FLASHINFER_SAMPLER"

HYBRID_CFG = types.SimpleNamespace(
    architectures=["Qwen3_5ForConditionalGeneration"],
    vision_config=object(),
    text_config=types.SimpleNamespace(layer_types=["linear_attention", "full_attention"]),
)
PLAIN_CFG = types.SimpleNamespace(architectures=["Qwen2ForCausalLM"], layer_types=None)


@dataclasses.dataclass
class EngineArgsWithGDN:
    model: str = ""
    gdn_prefill_backend: str = "auto"


@dataclasses.dataclass
class EngineArgsWithoutGDN:
    model: str = ""


def _install_fake_vllm(monkeypatch, engine_args_cls):
    """Install fake vllm packages; returns the list that LLM() calls are recorded into."""
    calls: list[dict] = []

    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1]}

    class LLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def get_tokenizer(self):
            return FakeTokenizer()

    vllm = types.ModuleType("vllm")
    vllm.LLM = LLM
    engine = types.ModuleType("vllm.engine")
    arg_utils = types.ModuleType("vllm.engine.arg_utils")
    arg_utils.EngineArgs = engine_args_cls
    engine.arg_utils = arg_utils
    vllm.engine = engine
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.engine", engine)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", arg_utils)
    return calls


def _configs(jit: bool):
    model = ModelConfig(name_or_path="Qwen/Qwen3.5-9B", max_model_len=4096)
    backend = BackendConfig(kind="vllm", jit_kernels=jit)
    return model, backend


def _build(monkeypatch, hf_cfg, *, jit: bool, engine_args_cls=EngineArgsWithGDN):
    # Record the env var through monkeypatch so the backend's direct os.environ write is
    # undone at teardown (setenv then delenv records the original state, even when absent).
    monkeypatch.setenv(ENV, "sentinel")
    monkeypatch.delenv(ENV)
    calls = _install_fake_vllm(monkeypatch, engine_args_cls)
    monkeypatch.setattr(vllm_backend, "_hf_config", lambda model_cfg: hf_cfg)
    model, backend = _configs(jit)
    VLLMBackend(model, backend)
    assert len(calls) == 1
    return calls[0]


# ----------------------------------------------------------------- jit_kernels=False
def test_hybrid_model_without_jit_gets_triton_gdn_and_sampler_off(monkeypatch):
    kwargs = _build(monkeypatch, HYBRID_CFG, jit=False)
    assert os.environ[ENV] == "0"
    assert kwargs["gdn_prefill_backend"] == "triton"
    assert kwargs["limit_mm_per_prompt"] == {"image": 0, "video": 0}


def test_plain_model_without_jit_has_no_gdn_kwarg(monkeypatch):
    kwargs = _build(monkeypatch, PLAIN_CFG, jit=False)
    assert os.environ[ENV] == "0"
    assert "gdn_prefill_backend" not in kwargs
    assert "limit_mm_per_prompt" not in kwargs


def test_unknown_config_without_jit_is_text_only(monkeypatch):
    kwargs = _build(monkeypatch, None, jit=False)
    assert os.environ[ENV] == "0"
    assert "gdn_prefill_backend" not in kwargs
    assert "limit_mm_per_prompt" not in kwargs


def test_hybrid_model_engine_args_without_field_skips_gdn_kwarg(monkeypatch):
    kwargs = _build(monkeypatch, HYBRID_CFG, jit=False, engine_args_cls=EngineArgsWithoutGDN)
    assert os.environ[ENV] == "0"
    assert "gdn_prefill_backend" not in kwargs
    # multimodal gating is independent of the engine-arg check
    assert kwargs["limit_mm_per_prompt"] == {"image": 0, "video": 0}


# ----------------------------------------------------------------- jit_kernels=True
def test_hybrid_model_with_jit_sets_neither_env_nor_gdn_kwarg(monkeypatch):
    kwargs = _build(monkeypatch, HYBRID_CFG, jit=True)
    assert ENV not in os.environ
    assert "gdn_prefill_backend" not in kwargs
    assert kwargs["limit_mm_per_prompt"] == {"image": 0, "video": 0}


def test_plain_model_with_jit_sets_nothing(monkeypatch):
    kwargs = _build(monkeypatch, PLAIN_CFG, jit=True)
    assert ENV not in os.environ
    assert "gdn_prefill_backend" not in kwargs
    assert "limit_mm_per_prompt" not in kwargs


# ----------------------------------------------------------------- helpers
def test_has_linear_attention_helper():
    assert vllm_backend._has_linear_attention(HYBRID_CFG) is True
    assert vllm_backend._has_linear_attention(
        types.SimpleNamespace(layer_types=["linear_attention"])) is True
    assert vllm_backend._has_linear_attention(PLAIN_CFG) is False
    assert vllm_backend._has_linear_attention(None) is False


def test_is_multimodal_helper_uses_given_config():
    assert vllm_backend._is_multimodal(None, HYBRID_CFG) is True
    assert vllm_backend._is_multimodal(None, PLAIN_CFG) is False
    assert vllm_backend._is_multimodal(None, None) is False


def test_engine_arg_supported(monkeypatch):
    _install_fake_vllm(monkeypatch, EngineArgsWithGDN)
    assert vllm_backend._engine_arg_supported("gdn_prefill_backend") is True
    assert vllm_backend._engine_arg_supported("no_such_field") is False
    _install_fake_vllm(monkeypatch, EngineArgsWithoutGDN)
    assert vllm_backend._engine_arg_supported("gdn_prefill_backend") is False


# ----------------------------------------------------------------- config validation
@pytest.mark.parametrize("value,expected", [("true", True), ("false", False)])
def test_backend_jit_kernels_config(value, expected):
    assert load_config(None, [f"backend.jit_kernels={value}"]).backend.jit_kernels is expected


def test_backend_jit_kernels_defaults_false():
    assert load_config().backend.jit_kernels is False
