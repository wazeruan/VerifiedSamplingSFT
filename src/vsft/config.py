"""Typed configuration loaded from YAML.

Every section is a dataclass with defaults, so a config file only needs to list the
values it changes. Unknown keys are rejected so typos fail loudly.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "auto"  # auto | bfloat16 | float16 | float32
    device: str = "auto"  # auto | cuda | mps | cpu (HF backend only)
    trust_remote_code: bool = False
    max_model_len: int = 4096


@dataclass
class BackendConfig:
    kind: str = "hf"  # hf | vllm
    gen_batch_size: int = 8  # prompts per HF generate call (each yields n samples)
    score_batch_size: int = 4  # sequences per HF scoring forward pass
    gpu_memory_utilization: float = 0.85  # vLLM only
    tensor_parallel_size: int = 1  # vLLM only
    seed: int = 0


@dataclass
class BoostConfig:
    num_candidates: int = 4  # K candidates regenerated per round
    max_rounds: int = 6  # verifier/regenerate iterations per example
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 1024
    # Winner selection: argmax of mean token log-prob under the student; every candidate
    # within `tie_epsilon` (nats/token) of the best is tied, and the tie is broken by
    # token-F1 overlap with the expert trace.
    tie_epsilon: float = 0.02
    # Mean log-prob over the full candidate ("full": prefix + continuation) or only the
    # newly generated continuation ("continuation"; ties are less diluted by long prefixes).
    score_scope: str = "full"
    drop_truncated: bool = True  # prefer candidates that emitted EOS over truncated ones
    include_expert_candidate: bool = False  # add the raw expert trace to round-0 candidates
    on_failure: str = "expert"  # expert | drop | last  (what to emit when rounds run out)
    proposal_prompt: str = "generic"  # key in prompts.PROPOSAL_PROMPTS
    examples_per_batch: int = 16  # examples advanced together per round (throughput)


@dataclass
class VerifierConfig:
    # The verifier is the same local model as the student (shared backend).
    max_new_tokens: int = 256
    temperature: float = 0.0
    use_answer_check: bool = True  # also require the extracted final answer to match
    # If the verifier reply cannot be parsed but the final answer matches, accept anyway
    # (useful with small verifiers that rarely emit valid JSON).
    accept_unparsed_if_answer_ok: bool = False
    max_step_chars: int = 400  # long lines are split into sentence-level steps


@dataclass
class TrainConfig:
    learning_rate: float = 1e-5
    num_epochs: int = 2
    per_device_batch_size: int = 4
    grad_accum_steps: int = 4
    max_seq_len: int = 2048
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    save_every_steps: int = 0  # 0 = only at epoch ends
    keep_last_states: int = 2  # resumable state-* dirs kept on disk (0 = keep all)
    log_every_steps: int = 10
    seed: int = 0
    # Precision: master weights dtype and autocast mode (recorded in the run manifest).
    # auto = fp32 master weights for full FT, bf16 frozen base for LoRA on CUDA
    param_dtype: str = "auto"  # auto | float32 | bfloat16
    save_dtype: str = "bfloat16"  # dtype of exported checkpoints: bfloat16 | float32
    mixed_precision: str = "auto"  # auto (bf16 on CUDA, else no) | bf16 | fp16 | no
    # LoRA (0 = full-parameter finetuning).
    lora_r: int = 0
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Which text to train on: boosted `response` (Sampling SFT) or raw `expert` (SFT baseline).
    source: str = "response"
    include_fallbacks: bool = True  # keep rows whose response fell back to the expert trace


@dataclass
class EvalConfig:
    max_new_tokens: int = 1024
    temperature: float = 0.0
    batch_size: int = 16


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    boost: BoostConfig = field(default_factory=BoostConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _coerce_field(value: Any, type_name: str, path: str) -> Any:
    """Coerce YAML/CLI scalars to the field type (PyYAML reads `1e-5` as a string)."""
    t = type_name.split("|")[0].strip()
    try:
        if t == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.lower() in {"true", "false"}:
                return value.lower() == "true"
            raise ValueError
        if t == "int":
            if isinstance(value, bool):
                raise ValueError
            f = float(value)
            if not f.is_integer():
                raise ValueError
            return int(f)
        if t == "float":
            if isinstance(value, bool):
                raise ValueError
            return float(value)
        if t == "str":
            if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                raise ValueError
            return str(value)
    except (TypeError, ValueError):
        raise ValueError(f"config '{path}' expects {t}, got {value!r}") from None
    return value


def _build(cls: type, data: dict[str, Any] | None, path: str) -> Any:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"config section '{path}' must be a mapping")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(f"unknown config keys in '{path or 'root'}': {sorted(unknown)}")
    kwargs = {}
    for key, value in data.items():
        default = getattr(cls(), key)
        where = f"{path}.{key}" if path else key
        if dataclasses.is_dataclass(default):
            kwargs[key] = _build(type(default), value, where)
        else:
            kwargs[key] = _coerce_field(value, str(fields[key].type), where)
    return cls(**kwargs)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load YAML (optional) then apply `section.key=value` overrides."""
    data: dict[str, Any] = {}
    if path is not None:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    for item in overrides or []:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        dotted, value = item.split("=", 1)
        section, key = dotted.split(".", 1)
        if data.get(section) is None:
            data[section] = {}
        elif not isinstance(data[section], dict):
            raise ValueError(f"config section '{section}' must be a mapping")
        data[section][key] = yaml.safe_load(value)
    cfg = _build(Config, data, "")
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    from vsft.prompts import PROPOSAL_PROMPTS

    b, v, m, be = cfg.boost, cfg.verifier, cfg.model, cfg.backend
    if be.kind not in {"hf", "vllm"}:
        raise ValueError("backend.kind must be 'hf' or 'vllm'")
    if m.dtype not in {"auto", "bfloat16", "float16", "float32"}:
        raise ValueError("model.dtype must be auto | bfloat16 | float16 | float32")
    if m.max_model_len < 64:
        raise ValueError("model.max_model_len must be >= 64")
    if min(be.gen_batch_size, be.score_batch_size, be.tensor_parallel_size) < 1:
        raise ValueError("backend batch sizes and tensor_parallel_size must be >= 1")
    if not 0 < be.gpu_memory_utilization <= 1:
        raise ValueError("backend.gpu_memory_utilization must be in (0, 1]")
    if b.examples_per_batch < 1:
        raise ValueError("boost.examples_per_batch must be >= 1")
    if b.num_candidates > 1 and b.temperature == 0:
        raise ValueError("boost.temperature must be > 0 when num_candidates > 1")
    if b.proposal_prompt not in PROPOSAL_PROMPTS:
        raise ValueError(f"boost.proposal_prompt must be one of {sorted(PROPOSAL_PROMPTS)}")
    if b.score_scope not in {"full", "continuation"}:
        raise ValueError("boost.score_scope must be full | continuation")
    if b.max_new_tokens >= m.max_model_len or v.max_new_tokens >= m.max_model_len:
        raise ValueError("max_new_tokens must be < model.max_model_len")
    if v.temperature < 0 or cfg.eval.temperature < 0:
        raise ValueError("verifier.temperature and eval.temperature must be >= 0")
    if cfg.eval.batch_size < 1 or cfg.eval.max_new_tokens < 1:
        raise ValueError("eval.batch_size and eval.max_new_tokens must be >= 1")
    if b.num_candidates < 1 or b.max_rounds < 1 or b.max_new_tokens < 1:
        raise ValueError("boost.num_candidates, max_rounds, max_new_tokens must be >= 1")
    if b.tie_epsilon < 0:
        raise ValueError("boost.tie_epsilon must be >= 0")
    if b.on_failure not in {"expert", "drop", "last"}:
        raise ValueError("boost.on_failure must be expert | drop | last")
    if not (0 < b.top_p <= 1) or b.temperature < 0:
        raise ValueError("boost.top_p must be in (0, 1] and temperature >= 0")
    if v.max_new_tokens < 1 or v.max_step_chars < 20:
        raise ValueError("verifier.max_new_tokens >= 1 and max_step_chars >= 20 required")
    t = cfg.train
    if t.learning_rate <= 0 or t.num_epochs < 1:
        raise ValueError("train.learning_rate > 0 and num_epochs >= 1 required")
    if t.per_device_batch_size < 1 or t.grad_accum_steps < 1 or t.max_seq_len < 16:
        raise ValueError("train batch size / grad_accum_steps >= 1 and max_seq_len >= 16")
    if not 0 <= t.warmup_ratio < 1:
        raise ValueError("train.warmup_ratio must be in [0, 1)")
    if t.param_dtype not in {"auto", "float32", "bfloat16"}:
        raise ValueError("train.param_dtype must be auto | float32 | bfloat16")
    if t.save_dtype not in {"float32", "bfloat16"}:
        raise ValueError("train.save_dtype must be float32 | bfloat16")
    if t.mixed_precision not in {"auto", "bf16", "fp16", "no"}:
        raise ValueError("train.mixed_precision must be auto | bf16 | fp16 | no")
    if t.source not in {"response", "expert"}:
        raise ValueError("train.source must be response or expert")
    if t.keep_last_states < 0 or t.save_every_steps < 0:
        raise ValueError("train.keep_last_states and save_every_steps must be >= 0")
    if t.lora_r < 0:
        raise ValueError("train.lora_r must be >= 0")
