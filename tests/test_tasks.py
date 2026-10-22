"""Task support: medical/MMLU preparation, TaskConfig validation, judge-aware verifier gate,
and the config-get / config_models plumbing used by the Slurm scripts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import vsft.data as data_mod
from vsft.config import TaskConfig, load_config
from vsft.io import read_jsonl
from vsft.verifier import VerifyRequest, decide
from vsft.hpc import config_models

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs" / "h100"


# ---------------------------------------------------------------- fake HF datasets


TRAIN_FAKE = [
    {"Question": "What is aspirin used for?", "Complex_CoT": "Think about NSAIDs.",
     "Response": "Pain and fever relief."},
    {"Question": "  Describe   Insulin's role? ", "Complex_CoT": "Glucose homeostasis.",
     "Response": "It lowers blood glucose."},
    {"Question": "Empty trace?", "Complex_CoT": "", "Response": "x"},  # dropped
]
POOL_FAKE = [
    {"Open-ended Verifiable Question": "what is aspirin used for?",  # in train (case)
     "Ground-True Answer": "pain relief"},
    {"Open-ended Verifiable Question": "describe insulin's role?",  # in train (spacing)
     "Ground-True Answer": "lowers glucose"},
    {"Open-ended Verifiable Question": "Name the powerhouse of the cell.",
     "Ground-True Answer": "mitochondria"},
    {"Open-ended Verifiable Question": "Which vitamin is made in skin?",
     "Ground-True Answer": "vitamin D"},
    {"Open-ended Verifiable Question": "No answer here", "Ground-True Answer": "  "},  # dropped
    {"Open-ended Verifiable Question": "What carries oxygen in blood?",
     "Ground-True Answer": "hemoglobin"},
]
MMLU_FAKE = [
    {"question": f"Question {i}?", "subject": "anatomy" if i % 2 else "virology",
     "choices": ["w", "x", "y", "z"], "answer": i % 4}
    for i in range(20)
]


@pytest.fixture
def fake_hf(monkeypatch):
    import datasets

    calls = []

    def load(name, *args, split=None, **kw):
        calls.append((name, args, split))
        if name == "FreedomIntelligence/medical-o1-reasoning-SFT":
            assert args == ("en",) and split == "train"
            return TRAIN_FAKE
        if name == "FreedomIntelligence/medical-o1-verifiable-problem":
            assert not args and split == "train"
            return POOL_FAKE
        if name == "cais/mmlu":
            assert args == ("all",) and split == "test"
            return MMLU_FAKE
        raise AssertionError(f"unexpected dataset {name}")

    monkeypatch.setattr(datasets, "load_dataset", load)
    return calls


# ---------------------------------------------------------------- medical preparation


def test_prepare_medical_train_rows(tmp_path, fake_hf):
    info = data_mod.prepare_medical(tmp_path, n_eval=2, seed=0)
    assert info == {"train": 2, "test": 2}
    train = list(read_jsonl(tmp_path / "medical_train.jsonl"))
    assert [r["id"] for r in train] == ["med-train-0", "med-train-1"]
    for r in train:
        assert r["answer"] is None and r["answer_type"] == "judge"
    assert train[0]["expert"] == "Think about NSAIDs.\n\nPain and fever relief."
    assert train[0]["question"] == "What is aspirin used for?"


def test_prepare_medical_eval_excludes_training_questions(tmp_path, fake_hf):
    data_mod.prepare_medical(tmp_path, n_eval=3, seed=0)
    train_q = {data_mod._norm_problem(r["question"])
               for r in read_jsonl(tmp_path / "medical_train.jsonl")}
    test = list(read_jsonl(tmp_path / "medical_test.jsonl"))
    assert all(data_mod._norm_problem(r["question"]) not in train_q for r in test)
    # the two pool rows that duplicate training questions (case / spacing differ) are excluded,
    # so all three remaining candidates are sampled
    assert {r["question"] for r in test} == {
        "Name the powerhouse of the cell.",
        "Which vitamin is made in skin?",
        "What carries oxygen in blood?",
    }


def test_prepare_medical_eval_rows_and_ids(tmp_path, fake_hf):
    data_mod.prepare_medical(tmp_path, n_eval=3, seed=1)
    test = list(read_jsonl(tmp_path / "medical_test.jsonl"))
    ids = [r["id"] for r in test]
    assert len(set(ids)) == len(ids)
    by_id = {r["id"]: r for r in test}
    for r in test:
        assert r["answer_type"] == "judge"
        assert r["answer"] == r["expert"] and r["answer"].strip()
        assert r["answer"] in {"mitochondria", "vitamin D", "hemoglobin"}
    assert all(r["id"].startswith("med-eval-") for r in test)
    assert len(by_id) == 3


def test_prepare_medical_n_eval_too_large(tmp_path, fake_hf):
    # 3 eval candidates survive the train filter and the blank-answer filter
    with pytest.raises(ValueError, match="n_eval"):
        data_mod.prepare_medical(tmp_path, n_eval=4)
    assert not (tmp_path / "medical_test.jsonl").exists()


def test_prepare_medical_sampling_is_deterministic(tmp_path, fake_hf):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    data_mod.prepare_medical(a, n_eval=2, seed=7)
    data_mod.prepare_medical(b, n_eval=2, seed=7)
    assert (a / "medical_test.jsonl").read_text() == (b / "medical_test.jsonl").read_text()
    # every seed yields a valid sample from the same pool
    for seed in range(5):
        data_mod.prepare_medical(c, n_eval=2, seed=seed)
        assert len(list(read_jsonl(c / "medical_test.jsonl"))) == 2


# ---------------------------------------------------------------- MMLU preparation


def test_prepare_mmlu_rows(tmp_path, fake_hf):
    n = data_mod.prepare_mmlu(tmp_path, n=5, seed=0)
    assert n == 5
    rows = list(read_jsonl(tmp_path / "mmlu_test.jsonl"))
    assert len(rows) == 5 and len({r["id"] for r in rows}) == 5
    for r in rows:
        assert r["answer_type"] == "mc"
        assert r["answer"] in {"A", "B", "C", "D"}
        lines = r["question"].split("\n")
        assert "A. w" in lines and "B. x" in lines and "C. y" in lines and "D. z" in lines
        assert r["question"].endswith("Answer with the letter of the correct option.")
        assert r["subject"] in {"anatomy", "virology"}
    # the reference letter matches the choice index of the source row
    by_id = {int(r["id"].split("-")[1]): r for r in rows}
    for i, r in by_id.items():
        assert r["answer"] == "ABCD"[i % 4]


def test_prepare_mmlu_deterministic_and_capped(tmp_path, fake_hf):
    data_mod.prepare_mmlu(tmp_path / "a", n=7, seed=3)
    data_mod.prepare_mmlu(tmp_path / "b", n=7, seed=3)
    assert (tmp_path / "a" / "mmlu_test.jsonl").read_text() == (
        tmp_path / "b" / "mmlu_test.jsonl").read_text()
    assert data_mod.prepare_mmlu(tmp_path / "c", n=1000, seed=0) == len(MMLU_FAKE)


def test_preparers_registered():
    assert data_mod.PREPARERS["medical"] is data_mod.prepare_medical
    assert data_mod.PREPARERS["mmlu"] is data_mod.prepare_mmlu


# ---------------------------------------------------------------- TaskConfig


def test_task_defaults_reproduce_math():
    t = load_config().task
    assert isinstance(t, TaskConfig)
    assert t.name == "math" and t.train_file == "math_l345_train.jsonl"
    assert t.eval_files == ["math_l345_test.jsonl", "math500_test.jsonl", "gsm8k_test.jsonl"]
    assert t.prepare == ["math", "math500", "gsm8k"]


@pytest.mark.parametrize("name", ["math_qwen35.yaml", "medical_qwen35.yaml"])
def test_shipped_configs_load(name):
    cfg = load_config(CONFIGS / name)
    assert cfg.task.train_file.endswith(".jsonl")
    assert all(f.endswith(".jsonl") for f in cfg.task.eval_files)


def test_medical_config_values():
    cfg = load_config(CONFIGS / "medical_qwen35.yaml")
    assert cfg.task.name == "medical"
    assert cfg.task.prepare == ["medical", "mmlu", "math500", "gsm8k"]
    assert cfg.task.train_file == "medical_train.jsonl"
    assert cfg.task.eval_files == ["medical_test.jsonl", "mmlu_test.jsonl",
                                   "math500_test.jsonl", "gsm8k_test.jsonl"]
    assert cfg.eval.judge_model == "Qwen/Qwen3.5-9B"
    assert cfg.eval.judge_gpu_memory_utilization == pytest.approx(0.35)
    assert cfg.eval.judge_max_new_tokens == 16  # default


@pytest.mark.parametrize(
    "ov, match",
    [
        (["task.prepare=[nope]"], "task.prepare"),
        (["task.prepare=[]"], "task.prepare"),
        (["task.prepare=[medical, bogus]"], "bogus"),
        (["task.eval_files=[]"], "eval_files"),
        (["task.eval_files=[a.jsonl, a.jsonl]"], "duplicates"),
        (["task.eval_files=[sub/a.jsonl]"], "without directories"),
        (["task.eval_files=[a.json]"], "without directories"),
        (["task.train_file=dir/train.jsonl"], "train_file"),
        (["task.train_file=train.txt"], "train_file"),
        (["eval.judge_gpu_memory_utilization=0"], "judge_gpu"),
        (["eval.judge_gpu_memory_utilization=1"], "judge_gpu"),
        (["eval.judge_gpu_memory_utilization=1.2"], "judge_gpu"),
        (["eval.judge_max_new_tokens=0"], "judge_max_new_tokens"),
    ],
)
def test_task_validation_rejects(ov, match):
    with pytest.raises(ValueError, match=match):
        load_config(None, ov)


def test_task_validation_accepts_valid_overrides():
    cfg = load_config(None, ["task.prepare=[mmlu]", "task.eval_files=[mmlu_test.jsonl]",
                             "eval.judge_gpu_memory_utilization=0.5",
                             "eval.judge_model=''"])
    assert cfg.task.prepare == ["mmlu"] and cfg.eval.judge_model == ""


# ---------------------------------------------------------------- verifier judge gate


RESP = "Step one.\nStep two gives \\boxed{7}."


def test_decide_judge_skips_answer_gate():
    req = VerifyRequest("q", "e", RESP, reference_answer="7", answer_type="judge")
    v = decide(req, '{"correct": true}', 400, use_answer_check=True)
    assert v.answer_ok is None and v.reason == "accepted" and v.correct


def test_decide_judge_answer_gate_not_applied_even_on_mismatch():
    req = VerifyRequest("q", "e", RESP, reference_answer="9", answer_type="judge")
    v = decide(req, '{"correct": true}', 400, use_answer_check=True)
    assert v.answer_ok is None and v.correct


def test_decide_math_still_gated():
    req = VerifyRequest("q", "e", RESP, reference_answer="9", answer_type="math")
    v = decide(req, '{"correct": true}', 400, use_answer_check=True)
    assert v.answer_ok is False and not v.correct


def test_decide_judge_unparsed_not_accepted_by_answer_only():
    req = VerifyRequest("q", "e", RESP, reference_answer="7", answer_type="judge")
    v = decide(req, "no idea", 400, use_answer_check=True, accept_unparsed_if_answer_ok=True)
    assert not v.correct and v.answer_ok is None


# ---------------------------------------------------------------- hpc plumbing


def run_config_get(path, key, *extra):
    r = subprocess.run(
        [sys.executable, "-m", "vsft.hpc", "config-get", "--config", str(path), key, *extra],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_config_get_lists_one_per_line():
    out = run_config_get(CONFIGS / "medical_qwen35.yaml", "task.eval_files")
    assert out.splitlines() == ["medical_test.jsonl", "mmlu_test.jsonl",
                                "math500_test.jsonl", "gsm8k_test.jsonl"]
    out = run_config_get(CONFIGS / "medical_qwen35.yaml", "task.prepare")
    assert out.splitlines() == ["medical", "mmlu", "math500", "gsm8k"]


def test_config_get_scalar(tmp_path):
    out = run_config_get(CONFIGS / "medical_qwen35.yaml", "task.train_file")
    assert out.strip() == "medical_train.jsonl"


def test_config_models_includes_judge_when_distinct():
    cfg = load_config(CONFIGS / "medical_qwen35.yaml")
    assert config_models(cfg) == ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"]  # verifier == judge
    cfg = load_config(None, ["eval.judge_model=judge/Other"])
    assert config_models(cfg) == ["Qwen/Qwen3.5-4B", "judge/Other"]
    cfg = load_config(None, ["verifier.model=''", "eval.judge_model=''"])
    assert config_models(cfg) == ["Qwen/Qwen3.5-4B"]
    cfg = load_config(None, ["verifier.model=''", "eval.judge_model=Qwen/Qwen3.5-4B"])
    assert config_models(cfg) == ["Qwen/Qwen3.5-4B"]  # judge == student: not repeated


def test_config_get_models_key(tmp_path):
    out = run_config_get(CONFIGS / "medical_qwen35.yaml", "models")
    assert out.splitlines() == ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"]
