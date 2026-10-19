import json
import math
import re
from pathlib import Path

import pytest
import torch

from vsft.train import IGNORE, collate, completion_text, load_sft_pairs, tokenize_pair

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_math.jsonl"


class WsTok:
    """Whitespace tokenizer with integer ids, no chat template; '<eos>' is its own token."""

    chat_template = None
    eos_token = "<eos>"
    eos_token_id = 1
    pad_token_id = 0

    def __init__(self):
        self.vocab = {"<eos>": 1}

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self.vocab.setdefault(w, len(self.vocab) + 10)
                              for w in re.findall(r"<eos>|[^\s<]+|<", text)]}


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def read_hist(out):
    return [json.loads(x) for x in (out / "train_history.jsonl").read_text().splitlines()]


BOOSTED = [
    {"id": "1", "question": "q1", "expert": "e1", "response": "r1", "status": "accepted"},
    {"id": "2", "question": "q2", "expert": "e2", "response": "e2", "status": "fallback_expert"},
    {"id": "3", "question": "q3", "expert": "e3", "response": "w3", "status": "fallback_last"},
    {"id": "4", "question": "q4", "expert": "e4", "response": None, "status": "dropped"},
    {"id": "5", "question": "q5", "expert": "e5", "response": "   ", "status": "accepted"},
    {"id": "6", "question": "q6", "expert": "e6", "response": "r6", "status": None},
]


def test_load_sft_pairs(tmp_path):
    p = write(tmp_path / "b.jsonl", BOOSTED)
    assert load_sft_pairs(p, "response", True) == [
        ("q1", "r1"), ("q2", "e2"), ("q3", "w3"), ("q6", "r6")]
    assert load_sft_pairs(p, "response", False) == [("q1", "r1"), ("q6", "r6")]
    # the plain-SFT baseline (source=expert) sees exactly the same questions
    assert load_sft_pairs(p, "expert", True) == [
        ("q1", "e1"), ("q2", "e2"), ("q3", "e3"), ("q5", "e5"), ("q6", "e6")]
    assert load_sft_pairs(p, "expert", False) == [("q1", "e1"), ("q5", "e5"), ("q6", "e6")]
    for inc in (True, False):
        qs_resp = [q for q, _ in load_sft_pairs(p, "response", inc)]
        qs_exp = [q for q, _ in load_sft_pairs(p, "expert", inc)]
        # q5's response is blank, so only the expert source can use it
        assert [q for q in qs_exp if q != "q5"] == qs_resp
    # raw (un-boosted) file has no response field
    with pytest.raises(ValueError, match="no usable"):
        load_sft_pairs(FIXTURE, "response", True)
    assert len(load_sft_pairs(FIXTURE, "expert", True)) == 3


def test_completion_text_plain_tokenizer():
    prompt, comp = completion_text(WsTok(), "Q?", "It is 2")
    assert prompt.endswith("Assistant: ") and comp == "It is 2<eos>"


def test_tokenize_pair_masks_prompt_and_appends_eos():
    tok = WsTok()
    item = tokenize_pair(tok, "What is two?", "It is 2", max_len=100)
    prompt, _ = completion_text(tok, "What is two?", "It is 2")
    n_prompt = len(tok(prompt)["input_ids"])
    ids, labels = item["input_ids"], item["labels"]
    assert item["truncated"] is False
    assert len(ids) == len(labels) == n_prompt + 4
    assert labels[:n_prompt] == [IGNORE] * n_prompt
    assert labels[n_prompt:] == ids[n_prompt:]
    assert ids[-1] == tok.eos_token_id and labels[-1] == tok.eos_token_id
    # truncation keeps ids/labels aligned and is flagged
    t = tokenize_pair(tok, "What is two?", "It is 2", max_len=n_prompt + 2)
    assert t["truncated"] is True
    assert len(t["input_ids"]) == len(t["labels"]) == n_prompt + 2
    assert t["labels"][-2:] == t["input_ids"][-2:] and tok.eos_token_id not in t["labels"]
    # a single surviving completion token is still learnable
    assert tokenize_pair(tok, "What is two?", "It is 2", max_len=n_prompt + 1) is not None
    # prompt alone fills the context -> nothing to learn
    assert tokenize_pair(tok, "What is two?", "It is 2", max_len=n_prompt) is None
    # exact fit is not truncation
    assert tokenize_pair(tok, "What is two?", "It is 2", max_len=n_prompt + 4)["truncated"] is False


def test_collate():
    batch = [{"input_ids": [5, 6, 7], "labels": [IGNORE, 6, 7]},
             {"input_ids": [8], "labels": [8]}]
    out = collate(batch, pad_id=0)
    assert out["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert out["labels"].tolist() == [[IGNORE, 6, 7], [8, IGNORE, IGNORE]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert out["input_ids"].dtype == torch.long


# ----------------------------------------------------------------- train() on a tiny model


def _cfg(model_dir, *extra):
    from vsft.config import load_config

    return load_config(None, [
        f"model.name_or_path={model_dir}", "train.source=expert", "train.num_epochs=1",
        "train.per_device_batch_size=1", "train.grad_accum_steps=1", "train.log_every_steps=1",
        "train.max_seq_len=128", "train.learning_rate=1e-3", *extra,
    ])


def _completion_tokens(model_dir):
    from vsft.backends.hf import load_tokenizer

    tok = load_tokenizer(str(model_dir))
    total = 0
    for q, c in load_sft_pairs(FIXTURE, "expert", True):
        labels = tokenize_pair(tok, q, c, 128)["labels"]
        total += sum(x != IGNORE for x in labels[1:])
    return total


@pytest.mark.parametrize("lora", [0, 4])
def test_train_and_resume_tiny(tmp_path, tiny_model, lora):
    from transformers import AutoModelForCausalLM

    from vsft.train import train

    out = tmp_path / "run"
    final = train(_cfg(tiny_model, f"train.lora_r={lora}"), FIXTURE, out)
    assert final == out / "final"
    m = AutoModelForCausalLM.from_pretrained(final)
    assert not any(n.startswith("base_model") or "lora" in n for n in m.state_dict())
    hist = read_hist(out)
    assert len(hist) == 3 and all(math.isfinite(h["loss"]) for h in hist)
    assert {h["epoch"] for h in hist} == {0} and [h["step"] for h in hist] == [1, 2, 3]
    assert all(math.isfinite(h["grad_norm"]) for h in hist)
    man = json.loads((out / "train_manifest.json").read_text())
    g = man["geometry"]
    assert g["num_examples"] == 3 and g["total_steps"] == 3 and g["grad_accum_steps"] == 1
    assert len(g["data_sha256"]) == 64
    assert man["precision"]["configured_param_dtype"] == "auto"
    assert man["precision"]["effective_param_dtype"] == "torch.float32"
    assert man["precision"]["save_dtype"] == "bfloat16"
    assert man["precision"]["effective_mixed_precision"] in {"no", "bf16", "fp16"}
    state = json.loads((out / "state-epoch1" / "trainer_state.json").read_text())
    assert (state["step"], state["epoch"], state["batches_in_epoch"]) == (3, 1, 0)
    assert state["geometry"] == g
    assert (out / "checkpoint-epoch1").is_dir()
    if lora:
        assert (out / "checkpoint-epoch1" / "adapter_config.json").is_file()

    AutoModelForCausalLM.from_pretrained(out / "final")


@pytest.mark.parametrize("lora", [0, 4])
def test_resume_from_epoch_state_runs_only_remaining_epoch(tmp_path, tiny_model, lora):
    from vsft.train import train

    out = tmp_path / "run"
    cfg2 = _cfg(tiny_model, f"train.lora_r={lora}", "train.num_epochs=2")
    train(cfg2, FIXTURE, out)
    assert [h["step"] for h in read_hist(out)] == [1, 2, 3, 4, 5, 6]
    # resume the same 2-epoch run from the end of epoch 1: only epoch 2 runs again
    train(cfg2, FIXTURE, out, str(out / "state-epoch1"))
    hist = read_hist(out)
    # history is truncated to the resumed step, so steps are never duplicated
    assert [h["step"] for h in hist] == [1, 2, 3, 4, 5, 6]
    assert [h["epoch"] for h in hist] == [0, 0, 0, 1, 1, 1]
    assert json.loads((out / "train_result.json").read_text())["final_step"] == 6


def test_extend_one_epoch_run_refused(tmp_path, tiny_model):
    """Resume is strict by design: num_epochs (via total_steps) must match the saved run."""
    from vsft.train import train

    out = tmp_path / "run"
    train(_cfg(tiny_model), FIXTURE, out)
    with pytest.raises(ValueError, match="total_steps") as ei:
        train(_cfg(tiny_model, "train.num_epochs=2"), FIXTURE, out, str(out / "state-epoch1"))
    assert "num_epochs" in str(ei.value)
    assert [h["step"] for h in read_hist(out)] == [1, 2, 3]  # nothing ran


def test_resume_refuses_geometry_change(tmp_path, tiny_model):
    from vsft.train import train

    out = tmp_path / "run"
    train(_cfg(tiny_model, "train.grad_accum_steps=3"), FIXTURE, out)
    with pytest.raises(ValueError, match="grad_accum_steps"):
        train(_cfg(tiny_model, "train.grad_accum_steps=1"), FIXTURE, out,
              str(out / "state-epoch1"))
    other = tmp_path / "other.jsonl"
    other.write_text(FIXTURE.read_text().replace("12 + 30", "12+30"))
    with pytest.raises(ValueError, match="data_sha256"):
        train(_cfg(tiny_model, "train.grad_accum_steps=3"), other, out,
              str(out / "state-epoch1"))


def test_token_level_loss_normalization(tmp_path, tiny_model):
    """Window loss = sum(token NLL)/N_window: identical for 1x3 accumulation and one 3-batch."""
    from vsft.train import train

    a, b = tmp_path / "a", tmp_path / "b"
    train(_cfg(tiny_model, "train.grad_accum_steps=3", "train.gradient_checkpointing=false"),
          FIXTURE, a)
    train(_cfg(tiny_model, "train.per_device_batch_size=3", "train.gradient_checkpointing=false"),
          FIXTURE, b)
    ha, hb = read_hist(a), read_hist(b)
    assert len(ha) == len(hb) == 1
    n = _completion_tokens(tiny_model)
    assert ha[0]["tokens"] == hb[0]["tokens"] == n
    assert ha[0]["loss"] == pytest.approx(hb[0]["loss"], rel=1e-4)


def test_saved_dtype_and_weights_change(tmp_path, tiny_model):
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    from vsft.train import train

    before = AutoModelForCausalLM.from_pretrained(tiny_model).state_dict()
    final = train(_cfg(tiny_model, "train.gradient_checkpointing=false",
                       "train.learning_rate=1e-2"), FIXTURE, tmp_path / "x")
    [st] = list(final.glob("*.safetensors"))
    with safe_open(str(st), "pt") as f:
        dtypes = {f.get_tensor(k).dtype for k in f.keys()}
    assert dtypes == {torch.bfloat16}
    after = AutoModelForCausalLM.from_pretrained(final, dtype=torch.float32).state_dict()
    changed = [k for k in before if not torch.allclose(before[k], after[k], atol=2e-3)]
    assert changed
    f32 = train(_cfg(tiny_model, "train.save_dtype=float32"), FIXTURE, tmp_path / "y")
    [st] = list(f32.glob("*.safetensors"))
    with safe_open(str(st), "pt") as f:
        assert {f.get_tensor(k).dtype for k in f.keys()} == {torch.float32}


# ----------------------------------------------------------------- state rotation / resume


def _states(out):
    return sorted(d.name for d in out.glob("state-*"))


@pytest.mark.parametrize(
    "keep,expected",
    [
        (1, ["state-epoch1"]),
        (2, ["state-epoch1", "state-step3"]),
        (0, ["state-epoch1", "state-step1", "state-step2", "state-step3"]),
    ],
)
def test_keep_last_states(tmp_path, tiny_model, keep, expected):
    from vsft.train import train

    out = tmp_path / "run"
    train(_cfg(tiny_model, "train.save_every_steps=1", f"train.keep_last_states={keep}"),
          FIXTURE, out)
    assert _states(out) == expected
    for d in out.glob("state-*"):
        assert (d / "trainer_state.json").is_file()


def test_keep_last_states_default_is_two():
    from vsft.config import load_config

    assert load_config().train.keep_last_states == 2


@pytest.mark.parametrize("ov", ["train.keep_last_states=-1", "train.save_every_steps=-1"])
def test_keep_last_states_validation(ov):
    from vsft.config import load_config

    with pytest.raises(ValueError, match="keep_last_states"):
        load_config(None, [ov])


def test_rotate_states_unit(tmp_path):
    from vsft.train import _rotate_states

    def mk(name, step):
        d = tmp_path / name
        d.mkdir()
        (d / "trainer_state.json").write_text(json.dumps({"step": step}))
        return d

    a, b, c = mk("state-step5", 5), mk("state-step9", 9), mk("state-step7", 7)
    (tmp_path / "state-junk").mkdir()  # no trainer_state.json: never touched
    (tmp_path / "checkpoint-epoch1").mkdir()
    _rotate_states(tmp_path, 0, keep=b)
    assert a.exists() and b.exists() and c.exists()
    # the just-written dir survives even if it is not among the newest by step
    _rotate_states(tmp_path, 1, keep=a)
    assert a.exists() and b.exists() and not c.exists()
    _rotate_states(tmp_path, 1, keep=b)
    assert b.exists() and not a.exists()
    assert (tmp_path / "state-junk").exists() and (tmp_path / "checkpoint-epoch1").exists()


def test_resume_mid_epoch_matches_uninterrupted(tmp_path, tiny_model):
    from vsft.train import train

    cfg = _cfg(tiny_model, "train.save_every_steps=1", "train.keep_last_states=0",
               "train.gradient_checkpointing=false")
    out = tmp_path / "run"
    train(cfg, FIXTURE, out)
    ref = read_hist(out)
    train(cfg, FIXTURE, out, str(out / "state-step2"))
    hist = read_hist(out)
    assert [h["step"] for h in hist] == [1, 2, 3]
    # same data order (seedable sampler + skip) and restored optimizer -> same step-3 loss
    assert hist[2]["loss"] == pytest.approx(ref[2]["loss"], rel=1e-4)
    assert hist[:2] == ref[:2]
    assert json.loads((out / "train_result.json").read_text())["final_step"] == 3


def test_adamw_fused_flag_and_no_sync_single_process(tmp_path, tiny_model, monkeypatch):
    import accelerate

    from vsft.train import train

    seen = {}
    real = torch.optim.AdamW

    class SpyAdamW(real):
        def __init__(self, *a, **kw):
            seen.update(kw)
            super().__init__(*a, **kw)

    def no_sync(self, *a, **kw):
        raise AssertionError("no_sync must only be used under DDP (MULTI_GPU)")

    monkeypatch.setattr(torch.optim, "AdamW", SpyAdamW)
    monkeypatch.setattr(accelerate.Accelerator, "no_sync", no_sync)
    train(_cfg(tiny_model, "train.grad_accum_steps=3"), FIXTURE, tmp_path / "r")
    assert seen["fused"] is torch.cuda.is_available()
    assert len(read_hist(tmp_path / "r")) == 1


def test_no_sync_gate_source():
    """Static check: no_sync is gated on DistributedType.MULTI_GPU (cannot run DDP here)."""
    import inspect

    import vsft.train as tr

    src = inspect.getsource(tr.train)
    assert "use_no_sync = acc.distributed_type == DistributedType.MULTI_GPU" in src
    assert "acc.no_sync(model) if (use_no_sync and not last)" in src


# ---------------------------------------------------------------- graceful stop (Slurm)


def test_train_stop_file_then_resume(tmp_path, tiny_model, monkeypatch):
    from vsft.cli import main

    def args(out, *extra):
        sets = ["--set", f"model.name_or_path={tiny_model}"]
        for s in ("train.source=expert", "train.num_epochs=2", "train.per_device_batch_size=1",
                  "train.grad_accum_steps=1", "train.log_every_steps=1", "train.max_seq_len=128",
                  "train.learning_rate=1e-3", "train.gradient_checkpointing=false"):
            sets += ["--set", s]
        return ["train", "--data", str(FIXTURE), "--output-dir", str(out), *sets, *extra]

    ref = tmp_path / "ref"
    assert main(args(ref)) == 0
    ref_hist = read_hist(ref)
    assert [h["step"] for h in ref_hist] == [1, 2, 3, 4, 5, 6]

    out = tmp_path / "run"
    stop = tmp_path / "STOP"
    stop.write_text("")
    monkeypatch.setenv("VSFT_STOP_FILE", str(stop))
    monkeypatch.setenv("VSFT_RUN_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("VSFT_STAGE", "train")
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    assert main(args(out)) == 75
    st = json.loads((out / "STOPPED.json").read_text())
    assert st["step"] == 1 and Path(st["state_dir"]) == out / "state-preempt-step1"
    assert st["job_id"] == "4242"
    state = json.loads((out / "state-preempt-step1" / "trainer_state.json").read_text())
    assert (state["step"], state["epoch"], state["batches_in_epoch"]) == (1, 0, 1)
    assert not (out / "final").exists() and not (out / "train_result.json").exists()
    assert [h["step"] for h in read_hist(out)] == [1]
    started = json.loads((tmp_path / "jobs" / "status" / "train.STARTED").read_text())
    assert started["num_examples"] == 3

    # stop file still there -> the resumed run stops again after one more step
    assert main(args(out, "--resume-from", str(out / "state-preempt-step1"))) == 75
    assert json.loads((out / "STOPPED.json").read_text())["step"] == 2

    stop.unlink()
    monkeypatch.setenv("SLURM_JOB_ID", "4243")
    assert main(args(out, "--resume-from", str(out / "state-preempt-step2"))) == 0
    assert not (out / "STOPPED.json").exists()  # a completed run is no longer stopped
    assert json.loads((tmp_path / "jobs" / "status" / "train.STARTED").read_text())[
        "job_id"] == "4243"
    hist = read_hist(out)
    assert [h["step"] for h in hist] == [1, 2, 3, 4, 5, 6]
    assert json.loads((out / "train_result.json").read_text())["final_step"] == 6
    # same data order and optimizer state as the uninterrupted run
    for a, b in zip(hist, ref_hist, strict=True):
        assert a["loss"] == pytest.approx(b["loss"], rel=1e-4)
    from transformers import AutoModelForCausalLM

    AutoModelForCausalLM.from_pretrained(out / "final")


def test_stop_on_last_batch_of_epoch(tmp_path, tiny_model, monkeypatch):
    """Stopping after the epoch's last batch resumes into the next epoch without loss."""
    import vsft.hpc as hpc
    from vsft.train import Stopped, train

    calls = {"n": 0}

    def stop_on_third():
        calls["n"] += 1
        return calls["n"] == 3

    monkeypatch.setattr(hpc, "stop_requested", stop_on_third)
    cfg = _cfg(tiny_model, "train.num_epochs=2")
    with pytest.raises(Stopped) as ei:
        train(cfg, FIXTURE, tmp_path)
    assert ei.value.state_dir == tmp_path / "state-preempt-step3"
    state = json.loads((ei.value.state_dir / "trainer_state.json").read_text())
    assert (state["step"], state["epoch"], state["batches_in_epoch"]) == (3, 0, 3)
    monkeypatch.setattr(hpc, "stop_requested", lambda: False)
    train(cfg, FIXTURE, tmp_path, str(ei.value.state_dir))
    assert [h["step"] for h in read_hist(tmp_path)] == [1, 2, 3, 4, 5, 6]
    assert [h["epoch"] for h in read_hist(tmp_path)] == [0, 0, 0, 1, 1, 1]


def test_started_written_after_first_optimizer_step(tmp_path, tiny_model, monkeypatch):
    import vsft.hpc as hpc
    from vsft.train import train

    monkeypatch.setenv("VSFT_RUN_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("VSFT_STAGE", "train")
    marker = tmp_path / "jobs" / "status" / "train.STARTED"
    seen = []
    real = hpc.notify_started

    def spy(**info):
        # history is appended after the step that triggered STARTED
        hist = tmp_path / "run" / "train_history.jsonl"
        seen.append((info, hist.exists()))
        real(**info)

    monkeypatch.setattr(hpc, "notify_started", spy)
    train(_cfg(tiny_model), FIXTURE, tmp_path / "run")
    assert len(seen) == 1 and seen[0][0]["num_examples"] == 3
    assert seen[0][1] is False  # before step 1 was logged, but after its optimizer step
    assert marker.is_file()


def test_no_started_when_first_step_fails(tmp_path, tiny_model, monkeypatch):
    import vsft.hpc as hpc
    from vsft.train import train

    monkeypatch.setenv("VSFT_RUN_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("VSFT_STAGE", "train")

    def boom(self, *a, **kw):
        raise RuntimeError("simulated OOM in step")

    monkeypatch.setattr(torch.optim.AdamW, "step", boom)
    with pytest.raises(RuntimeError, match="simulated OOM"):
        train(_cfg(tiny_model), FIXTURE, tmp_path / "run")
    assert not (tmp_path / "jobs" / "status" / "train.STARTED").exists()
    del hpc


# ------------------------------------------------------------- end-of-turn / LoRA targets


class TemplTok:
    """Chat-template tokenizer whose generation prompt is NOT a prefix of the full
    conversation (like Qwen3.5's empty <think></think> block with enable_thinking=False)."""

    eos_token = "<|endoftext|>"

    def __init__(self, eot="<|im_end|>\n", name="fake/tok"):
        self.eot = eot
        self.name_or_path = name
        self.chat_template = f"fake template ending turns with {eot!r}"
        self.calls = 0

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False,
                            enable_thinking=None):
        self.calls += 1
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}{self.eot}" for m in msgs
                      if m["role"] != "assistant")
        if msgs[-1]["role"] == "assistant":
            return out + f"<|im_start|>assistant\n{msgs[-1]['content']}{self.eot}"
        return out + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def test_end_of_turn_from_template_and_fallback():
    from vsft.train import end_of_turn

    t = TemplTok()
    assert end_of_turn(t) == "<|im_end|>\n"
    prompt, comp = completion_text(t, "Q?", "It is 2")
    assert prompt.endswith("<think>\n\n</think>\n\n") and comp == "It is 2<|im_end|>\n"
    n = t.calls
    end_of_turn(t)
    assert t.calls == n  # cached per tokenizer
    ws = WsTok()  # kept alive: the cache is keyed by id() (see the id-reuse xfail below)
    assert end_of_turn(ws) == "<eos>"  # no chat template -> eos token
    blank = TemplTok(eot="")  # template adds nothing after the reply -> eos fallback
    assert end_of_turn(blank) == "<|endoftext|>"


def test_end_of_turn_cache_survives_id_reuse():
    import gc

    from vsft.train import end_of_turn

    for _ in range(200):
        a = TemplTok(eot="<A>\n")
        end_of_turn(a)
        old = id(a)
        del a
        gc.collect()
        b = TemplTok(eot="<B>\n")
        if id(b) == old:
            assert end_of_turn(b) == "<B>\n"
            return
    # CPython did not reuse an id this time: check the cache is not keyed by id() instead
    import vsft.train as tr

    assert all(isinstance(k, tuple) for k in tr._EOT_CACHE)
    assert end_of_turn(TemplTok(eot="<B>\n")) == "<B>\n"


def test_end_of_turn_real_tiny_tokenizer(tiny_model):
    from vsft.backends.hf import load_tokenizer
    from vsft.train import end_of_turn

    tok = load_tokenizer(str(tiny_model))
    assert end_of_turn(tok) == tok.eos_token  # tiny tokenizer has no chat template


@pytest.mark.parametrize(
    "ov,expected",
    [(None, ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
             "in_proj_qkv", "in_proj_z", "out_proj"]),
     ("train.lora_target_modules=all-linear", "all-linear"),
     ("train.lora_target_modules=[all-linear]", "all-linear"),
     ("train.lora_target_modules=[q_proj, v_proj]", ["q_proj", "v_proj"]),
     ("train.lora_target_modules=q_proj", ["q_proj"])],
)
def test_lora_target_modules_passed_to_peft(tmp_path, tiny_model, monkeypatch, ov, expected):
    import peft

    from vsft.train import train

    seen = []
    real = peft.LoraConfig

    def spy(**kw):
        seen.append(kw["target_modules"])
        return real(**kw)

    monkeypatch.setattr(peft, "LoraConfig", spy)
    extra = [ov] if ov else []
    train(_cfg(tiny_model, "train.lora_r=4", *extra), FIXTURE, tmp_path)
    assert seen[0] == expected or set(seen[0]) == set(expected)
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(tmp_path / "final")
    assert not any("lora" in k for k in m.state_dict())


def test_end_of_turn_cache_key():
    from vsft.train import end_of_turn

    a, b = TemplTok(eot="<A>\n"), TemplTok(eot="<A>\n")  # same name/template/eos
    assert end_of_turn(a) == "<A>\n"
    end_of_turn(b)
    assert b.calls == 0  # identical tokenizers share the cached value
    c = TemplTok(eot="<A>\n", name="other/tok")
    end_of_turn(c)
    assert c.calls == 1  # a different name_or_path is probed separately
    d = TemplTok(eot="<A>\n")
    d.eos_token = "<eos2>"
    end_of_turn(d)
    assert d.calls == 1


@pytest.mark.parametrize(
    "value,expected",
    [("all-linear", "all-linear"), (["all-linear"], "all-linear"), ("q_proj", ["q_proj"]),
     (["q_proj", "v_proj"], ["q_proj", "v_proj"]), (("q_proj",), ["q_proj"])],
)
def test_lora_targets(value, expected):
    from vsft.train import lora_targets

    assert lora_targets(value) == expected
