import pytest

from vsft.config import Config, load_config


def test_defaults():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.boost.num_candidates == 4 and cfg.backend.kind == "hf"
    d = cfg.to_dict()
    assert d["boost"]["on_failure"] == "expert"


def test_yaml_and_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("boost:\n  num_candidates: 8\n  on_failure: drop\nmodel:\n  dtype: float32\n")
    cfg = load_config(
        p,
        [
            "boost.num_candidates=2",
            "boost.include_expert_candidate=true",
            "model.name_or_path=x/y",
        ],
    )
    assert cfg.boost.num_candidates == 2
    assert cfg.boost.on_failure == "drop"
    assert cfg.boost.include_expert_candidate is True
    assert cfg.model.dtype == "float32" and cfg.model.name_or_path == "x/y"
    assert cfg.boost.max_rounds == 6  # untouched default


def test_empty_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("")
    assert load_config(p).boost.num_candidates == 4


@pytest.mark.parametrize(
    "overrides,match",
    [
        (["boost.num_candidatez=3"], "unknown config keys in 'boost'"),
        (["bost.num_candidates=3"], "unknown config keys"),
        (["boost=3"], "section.key=value"),
        (["boost.num_candidates"], "section.key=value"),
        (["boost.num_candidates=0"], "num_candidates"),
        (["boost.max_rounds=0"], "max_rounds"),
        (["boost.tie_epsilon=-0.1"], "tie_epsilon"),
        (["boost.on_failure=retry"], "on_failure"),
        (["boost.top_p=0"], "top_p"),
        (["boost.top_p=1.5"], "top_p"),
        (["boost.temperature=-1"], "temperature"),
        (["backend.kind=tgi"], "backend.kind"),
        (["verifier.max_step_chars=5"], "max_step_chars"),
        (["verifier.max_new_tokens=0"], "max_new_tokens"),
        (["train.learning_rate=0"], "learning_rate"),
    ],
)
def test_validation_errors(overrides, match):
    with pytest.raises(ValueError, match=match):
        load_config(None, overrides)


def test_unknown_yaml_key(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("verifier:\n  use_answer_chek: false\n")
    with pytest.raises(ValueError, match="use_answer_chek"):
        load_config(p)
    p.write_text("boost: [1, 2]\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_config(p)


@pytest.mark.parametrize(
    "ov,attr,want",
    [
        ("boost.tie_epsilon=1e-3", ("boost", "tie_epsilon"), 1e-3),
        ("train.learning_rate=1e-5", ("train", "learning_rate"), 1e-5),
        ("boost.temperature=1", ("boost", "temperature"), 1.0),
        ("boost.num_candidates=8.0", ("boost", "num_candidates"), 8),
        ("boost.max_new_tokens=1e3", ("boost", "max_new_tokens"), 1000),
        ("model.name_or_path=123", ("model", "name_or_path"), "123"),
        ("boost.drop_truncated=false", ("boost", "drop_truncated"), False),
    ],
)
def test_coerce_field(ov, attr, want):
    v = getattr(getattr(load_config(None, [ov]), attr[0]), attr[1])
    assert v == want and type(v) is type(want)


def test_coerce_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("train:\n  learning_rate: 2e-5\n  num_epochs: '3'\n")
    cfg = load_config(p)
    assert cfg.train.learning_rate == pytest.approx(2e-5) and cfg.train.num_epochs == 3


@pytest.mark.parametrize(
    "ov",
    [
        "boost.num_candidates=2.5",
        "boost.num_candidates=many",
        "boost.num_candidates=true",
        "boost.tie_epsilon=abc",
        "boost.tie_epsilon=",
        "boost.drop_truncated=maybe",
        "boost.drop_truncated=1",
        "model.name_or_path=[a]",
        "model.name_or_path=null",
    ],
)
def test_coerce_rejects(ov):
    with pytest.raises(ValueError, match="expects"):
        load_config(None, [ov])


def test_override_into_null_yaml_section(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("boost:\n")
    assert load_config(p, ["boost.num_candidates=2"]).boost.num_candidates == 2


@pytest.mark.parametrize(
    "ovs,match",
    [
        (["boost.examples_per_batch=0"], "examples_per_batch"),
        (["boost.proposal_prompt=nope"], "proposal_prompt"),
        (["boost.temperature=0"], "num_candidates > 1"),
        (["boost.max_new_tokens=4096"], "max_model_len"),
        (["model.max_model_len=512", "verifier.max_new_tokens=512"], "max_model_len"),
        (["model.dtype=fp8"], "model.dtype"),
        (["boost.score_scope=prefix"], "score_scope"),
        (["model.max_model_len=32"], "max_model_len"),
        (["backend.gen_batch_size=0"], "batch"),
        (["backend.gpu_memory_utilization=1.5"], "gpu_memory_utilization"),
        (["verifier.temperature=-0.1"], "temperature"),
        (["eval.batch_size=0"], "eval"),
        (["train.warmup_ratio=1"], "warmup_ratio"),
        (["train.param_dtype=float16"], "param_dtype"),
        (["train.mixed_precision=fp8"], "mixed_precision"),
        (["train.source=both"], "train.source"),
        (["train.lora_r=-1"], "lora_r"),
        (["train.max_seq_len=8"], "max_seq_len"),
    ],
)
def test_new_validation_rules(ovs, match):
    with pytest.raises(ValueError, match=match):
        load_config(None, ovs)


def test_greedy_single_candidate_ok():
    cfg = load_config(None, ["boost.temperature=0", "boost.num_candidates=1",
                             "boost.score_scope=continuation"])
    assert cfg.boost.temperature == 0.0 and cfg.boost.score_scope == "continuation"


@pytest.mark.parametrize("section", ["[1, 2]", "5", "0", "false", "''", "text"])
def test_override_does_not_mask_bad_section(tmp_path, section):
    p = tmp_path / "c.yaml"
    p.write_text(f"boost: {section}\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_config(p, ["boost.num_candidates=2"])
    with pytest.raises(ValueError, match="must be a mapping"):
        load_config(p)


# ------------------------------------------------------------- Qwen3.5 + separate verifier


def test_qwen35_defaults():
    cfg = load_config()
    assert cfg.model.name_or_path == "Qwen/Qwen3.5-4B"
    assert cfg.verifier.model == ""  # opt-in: the student grades unless a config says otherwise
    assert cfg.verifier.gpu_memory_utilization == pytest.approx(0.35)


@pytest.mark.parametrize("v", ["0", "1", "1.0", "-0.1", "1.5"])
def test_verifier_gpu_memory_utilization_range(v):
    with pytest.raises(ValueError, match="verifier.gpu_memory_utilization"):
        load_config(None, [f"verifier.gpu_memory_utilization={v}"])


def test_verifier_model_overrides():
    assert load_config(None, ["verifier.model=''"]).verifier.model == ""
    assert load_config(None, ["verifier.model=a/b"]).verifier.model == "a/b"
    assert load_config(None, ["verifier.gpu_memory_utilization=0.5"]).verifier \
        .gpu_memory_utilization == 0.5


def test_empty_override_value_is_not_empty_string():
    # `--set verifier.model=` parses as YAML null, not "": use verifier.model='' instead
    with pytest.raises(ValueError, match="expects str"):
        load_config(None, ["verifier.model="])


@pytest.mark.parametrize("name", ["smoke.yaml", "h100/math_qwen35.yaml",
                                  "h100/math_qwen35_full_fsdp.yaml"])
def test_shipped_configs_load(name):
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "configs" / name
    cfg = load_config(p)
    assert cfg.model.name_or_path.startswith("Qwen/Qwen3.5-")
    if name == "smoke.yaml":
        assert cfg.model.name_or_path == "Qwen/Qwen3.5-0.8B" and cfg.verifier.model == ""
    else:
        assert cfg.verifier.model == "Qwen/Qwen3.5-9B"
        if cfg.backend.kind == "vllm" and cfg.verifier.model:
            assert (cfg.backend.gpu_memory_utilization + cfg.verifier.gpu_memory_utilization
                    <= 0.95)


def test_old_config_names_gone():
    from pathlib import Path

    h100 = Path(__file__).resolve().parents[1] / "configs" / "h100"
    assert not list(h100.glob("*qwen3b*")) and not list(h100.glob("*full80g*"))


def test_backend_vllm_batching_fields():
    cfg = load_config()
    assert cfg.backend.max_num_batched_tokens == 4096 and cfg.backend.max_num_seqs == 256
    cfg = load_config(None, ["backend.max_num_batched_tokens=8192", "backend.max_num_seqs=64"])
    assert (cfg.backend.max_num_batched_tokens, cfg.backend.max_num_seqs) == (8192, 64)
    for ov in ("backend.max_num_batched_tokens=0", "backend.max_num_seqs=0",
               "backend.max_num_seqs=-3"):
        with pytest.raises(ValueError, match="max_num_batched_tokens and max_num_seqs"):
            load_config(None, [ov])


def test_lora_target_modules_default_and_overrides(tmp_path):
    t = load_config().train
    assert t.lora_target_modules == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                     "up_proj", "down_proj", "in_proj_qkv", "in_proj_z",
                                     "out_proj"]
    assert "in_proj_a" not in t.lora_target_modules and "in_proj_b" not in t.lora_target_modules
    # defaults are not shared between instances
    a, b = load_config().train, load_config().train
    a.lora_target_modules.append("x")
    assert "x" not in b.lora_target_modules
    assert load_config(None, ["train.lora_target_modules=all-linear"]).train \
        .lora_target_modules == "all-linear"
    assert load_config(None, ["train.lora_target_modules=[q_proj, v_proj]"]).train \
        .lora_target_modules == ["q_proj", "v_proj"]
    p = tmp_path / "c.yaml"
    p.write_text("train:\n  lora_target_modules: [all-linear]\n")
    assert load_config(p).train.lora_target_modules == ["all-linear"]


def test_lora_target_modules_single_string_wrapped():
    from vsft.train import lora_targets

    t = load_config(None, ["train.lora_target_modules=q_proj"]).train
    assert lora_targets(t.lora_target_modules) == ["q_proj"]


@pytest.mark.parametrize(
    "value,match",
    [("''", "must not be empty"), ("'  '", "must not be empty"), ("[]", "all-linear"),
     ("[q_proj, 3]", "all-linear"), ("[q_proj, '']", "all-linear"), ("[[q_proj]]", "all-linear"),
     ("5", "all-linear")],
)
def test_lora_target_modules_validation(value, match):
    with pytest.raises(ValueError, match=match):
        load_config(None, [f"train.lora_target_modules={value}"])
