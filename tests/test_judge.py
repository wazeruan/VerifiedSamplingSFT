"""LLM judge for open-ended QA (answer_type "judge"): parsing, batching, evaluate and CLI wiring."""

from __future__ import annotations

import json

import pytest
from fakes import FakeBackend, parse_student

import vsft.backends.base as base_mod
import vsft.cli as cli_mod
from vsft.backends.base import Generation
from vsft.cli import _build_judge_backend, _has_judge_rows, main
from vsft.config import EvalConfig, load_config
from vsft.evaluate import evaluate
from vsft.io import read_jsonl
from vsft.judge import JUDGE_PROMPT, judge, parse_verdict


class Scripted(FakeBackend):
    """Backend whose reply to every prompt is computed by `reply(prompt)`."""

    def __init__(self, reply):
        super().__init__(lambda *a: [], lambda *a: "")
        self.reply = reply
        self.calls = []  # (n, temperature, top_p, max_tokens, prompts)

    def generate(self, prompts, n, temperature, top_p, max_tokens):
        self.calls.append((n, temperature, top_p, max_tokens, list(prompts)))
        return [[Generation(self.reply(p), True)] for p in prompts]


def is_judge_prompt(p: str) -> bool:
    return "You are an expert evaluator" in p


def judge_prompts(be):
    return [p for call in be.calls for p in call[4] if is_judge_prompt(p)]


# ---------------------------------------------------------------- parse_verdict


@pytest.mark.parametrize(
    "text, want",
    [
        ("CORRECT", True),
        ("correct", True),
        ("Correct.", True),
        ("**CORRECT**", True),
        ("INCORRECT", False),
        ("incorrect", False),
        ("INCORRECT.", False),
        ("  \nINCORRECT\n", False),
        ("CORRECT ... INCORRECT", False),  # last verdict word wins
        ("INCORRECT ... CORRECT", True),
        ("", None),
        ("maybe", None),
        ("INCORRECTLY", None),  # not a verdict word: no word boundary
        ("NOT CORRECT", False),
        ("NOT_CORRECT", False),
        ("not-correct", False),
        ("Not Correct.", False),
        ("The answer is correct. CORRECT", True),
    ],
)
def test_parse_verdict(text, want):
    assert parse_verdict(text) is want


def test_incorrect_is_never_read_as_correct():
    for text in ("INCORRECT", "Incorrect answer", "The response is INCORRECT."):
        assert parse_verdict(text) is False


# ---------------------------------------------------------------- judge()


def test_judge_empty_items_makes_no_call():
    be = Scripted(lambda p: "CORRECT")
    assert judge(be, []) == []
    assert be.calls == []


def test_judge_one_batched_greedy_call_in_order():
    be = Scripted(lambda p: "CORRECT" if "good" in p else "INCORRECT")
    items = [("q1", "ref1", "good"), ("q2", "ref2", "bad"), ("q3", "ref3", "good")]
    assert judge(be, items) == [True, False, True]
    assert len(be.calls) == 1
    n, temperature, top_p, max_tokens, prompts = be.calls[0]
    assert (n, temperature, top_p, max_tokens) == (1, 0.0, 1.0, 16)
    assert len(prompts) == 3


def test_judge_prompt_contains_question_reference_and_response():
    be = Scripted(lambda p: "CORRECT")
    judge(be, [("What is X?", "the key fact", "my answer text")])
    [p] = judge_prompts(be)
    assert "What is X?" in p and "the key fact" in p and "my answer text" in p
    assert JUDGE_PROMPT.split("\n")[0] in p


def test_judge_truncates_long_responses():
    be = Scripted(lambda p: "CORRECT")
    resp = "A" * 100 + "TAILMARK"
    judge(be, [("q", "r", resp)], max_response_chars=100)
    [p] = judge_prompts(be)
    assert "TAILMARK" not in p and "A" * 100 in p
    judge(be, [("q", "r", resp)])  # default limit (12000) keeps the whole response
    assert "TAILMARK" in judge_prompts(be)[-1]


def test_judge_unparsed_reply_is_none():
    be = Scripted(lambda p: "I cannot decide")
    assert judge(be, [("q", "r", "x")]) == [None]


def test_judge_max_tokens_passed_through():
    be = Scripted(lambda p: "CORRECT")
    judge(be, [("q", "r", "x")], max_tokens=7)
    assert be.calls[0][3] == 7


# ---------------------------------------------------------------- evaluate() with judge rows


JUDGE_ROWS = [
    {"id": "j1", "question": "jq1", "answer": "the key fact", "answer_type": "judge"},
    {"id": "j2", "question": "jq2", "answer": "another fact", "answer_type": "judge"},
    {"id": "j3", "question": "jq3", "answer": "third fact", "answer_type": "judge"},
]
MATH_ROW = {"id": "m1", "question": "mq", "answer": "4", "answer_type": "math"}


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def student_reply(p):
    """Student answers echo the question; the judge verdict depends on the answer text."""
    if is_judge_prompt(p):
        return "CORRECT" if "GOOD" in p else "INCORRECT"
    q = parse_student(p)
    return {"jq1": "GOOD answer", "jq2": "BAD answer", "jq3": "??", "mq": "\\boxed{4}"}[q]


def test_evaluate_judge_verdicts_drive_correctness(tmp_path):
    data = write(tmp_path / "open.jsonl", JUDGE_ROWS)
    student = Scripted(student_reply)
    judge_be = Scripted(lambda p: "INCORRECT" if "BAD" in p else "CORRECT")
    m = evaluate(student, data, tmp_path / "o", EvalConfig(batch_size=8, temperature=0.0),
                 judge_backend=judge_be)
    preds = list(read_jsonl(tmp_path / "o" / "open.predictions.jsonl"))
    assert [p["correct"] for p in preds] == [True, False, True]
    assert [p["judge_verdict"] for p in preds] == [True, False, True]
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert m["num_judge_unparsed"] == 0
    # judge prediction shows the student's text, not an extracted answer
    assert preds[0]["prediction"] == "GOOD answer"


def test_evaluate_unparsed_verdict_counts_as_incorrect(tmp_path):
    data = write(tmp_path / "open.jsonl", JUDGE_ROWS[:2])
    judge_be = Scripted(lambda p: "hmm" if "GOOD" in p else "CORRECT")
    m = evaluate(Scripted(student_reply), data, tmp_path / "o", EvalConfig(batch_size=8),
                 judge_backend=judge_be)
    preds = list(read_jsonl(tmp_path / "o" / "open.predictions.jsonl"))
    assert preds[0]["judge_verdict"] is None and preds[0]["correct"] is False
    assert m["num_judge_unparsed"] == 1
    assert m["accuracy"] == pytest.approx(0.5)


def test_evaluate_judge_batches_per_chunk(tmp_path):
    data = write(tmp_path / "open.jsonl", JUDGE_ROWS)
    judge_be = Scripted(lambda p: "CORRECT")
    evaluate(Scripted(student_reply), data, tmp_path / "o", EvalConfig(batch_size=2),
             judge_backend=judge_be)
    # chunks of 2 and 1 rows -> one judge generate call per chunk
    assert [len(c[4]) for c in judge_be.calls] == [2, 1]
    assert all(c[0] == 1 and c[1] == 0.0 for c in judge_be.calls)


def test_evaluate_judge_max_tokens_from_config(tmp_path):
    data = write(tmp_path / "open.jsonl", JUDGE_ROWS[:1])
    judge_be = Scripted(lambda p: "CORRECT")
    evaluate(Scripted(student_reply), data, tmp_path / "o",
             EvalConfig(batch_size=8, judge_max_new_tokens=5), judge_backend=judge_be)
    assert judge_be.calls[0][3] == 5


def test_evaluate_judge_rows_need_a_judge(tmp_path):
    data = write(tmp_path / "open.jsonl", JUDGE_ROWS)
    with pytest.raises(ValueError, match="judge"):
        evaluate(Scripted(student_reply), data, tmp_path / "o", EvalConfig(batch_size=8))
    assert not (tmp_path / "o" / "open.metrics.json").exists()


def test_evaluate_mixed_file_uses_judge_only_for_judge_rows(tmp_path):
    data = write(tmp_path / "mixed.jsonl", [JUDGE_ROWS[0], MATH_ROW])
    judge_be = Scripted(lambda p: "CORRECT")
    m = evaluate(Scripted(student_reply), data, tmp_path / "o", EvalConfig(batch_size=8),
                 judge_backend=judge_be)
    preds = {p["id"]: p for p in read_jsonl(tmp_path / "o" / "mixed.predictions.jsonl")}
    assert preds["j1"]["correct"] is True and preds["j1"]["judge_verdict"] is True
    assert preds["m1"]["correct"] is True and "judge_verdict" not in preds["m1"]
    assert len(judge_prompts(judge_be)) == 1  # the math row is never sent to the judge
    assert m["num_examples"] == 2


def test_evaluate_without_judge_rows_never_calls_judge(tmp_path):
    data = write(tmp_path / "math.jsonl", [MATH_ROW])

    class Boom(Scripted):
        def generate(self, *a, **k):
            raise AssertionError("judge called for a file without judge rows")

    evaluate(Scripted(student_reply), data, tmp_path / "o", EvalConfig(batch_size=8),
             judge_backend=Boom(lambda p: ""))


# ---------------------------------------------------------------- CLI wiring


@pytest.fixture
def built(monkeypatch):
    """Route every build_backend call through a recording factory; each backend answers
    student prompts with a boxed answer and judge prompts with CORRECT."""
    made = []

    def build(model_cfg, backend_cfg):
        be = Scripted(lambda p: "CORRECT" if is_judge_prompt(p) else "\\boxed{4}")
        be.name = model_cfg.name_or_path
        be.gpu = backend_cfg.gpu_memory_utilization
        made.append(be)
        return be

    monkeypatch.setattr(base_mod, "build_backend", build)
    monkeypatch.setattr(cli_mod, "build_backend", lambda m, b: base_mod.build_backend(m, b))
    return made


def test_has_judge_rows(tmp_path):
    assert _has_judge_rows(write(tmp_path / "a.jsonl", JUDGE_ROWS)) is True
    assert _has_judge_rows(write(tmp_path / "b.jsonl", [MATH_ROW, JUDGE_ROWS[0]])) is True
    assert _has_judge_rows(write(tmp_path / "c.jsonl", [])) is False


def test_build_judge_backend_reuses_student_when_unset_or_same(built):
    student = object()
    for ov in ([], ["eval.judge_model=''"], ["eval.judge_model=Qwen/Qwen3.5-4B"]):
        assert _build_judge_backend(load_config(None, ov), student) is student
    assert built == []


def test_build_judge_backend_builds_second_engine(built):
    cfg = load_config(None, ["eval.judge_model=Qwen/Qwen3.5-9B",
                             "eval.judge_gpu_memory_utilization=0.3",
                             "backend.gpu_memory_utilization=0.5"])
    jb = _build_judge_backend(cfg, object())
    assert jb is built[0] and jb.name == "Qwen/Qwen3.5-9B" and jb.gpu == pytest.approx(0.3)
    # the student's config is untouched
    assert cfg.model.name_or_path == "Qwen/Qwen3.5-4B"
    assert cfg.backend.gpu_memory_utilization == pytest.approx(0.5)


def test_cli_eval_judge_summary(tmp_path, built):
    data = write(tmp_path / "open.jsonl", JUDGE_ROWS[:2])
    od = tmp_path / "ev-judge"
    assert main(["eval", "--model", "my/model", "--data", str(data), "--output-dir", str(od),
                 "--set", "eval.judge_model=judge/9B"]) == 0
    summ = json.loads((od / "eval_summary.json").read_text())
    assert summ["decoding"]["judge_model"] == "judge/9B"
    assert summ["accuracy"]["open"] == pytest.approx(1.0)
    assert [b.name for b in built] == ["my/model", "judge/9B"]


def test_cli_eval_judge_defaults_to_student(tmp_path, built):
    data = write(tmp_path / "open.jsonl", JUDGE_ROWS[:1])
    od = tmp_path / "ev"
    main(["eval", "--model", "my/model", "--data", str(data), "--output-dir", str(od)])
    summ = json.loads((od / "eval_summary.json").read_text())
    assert summ["decoding"]["judge_model"] == "my/model"
    assert len(built) == 1  # the judge shares the student's engine


def test_cli_eval_no_judge_summary_is_none(tmp_path, built):
    data = write(tmp_path / "math.jsonl", [MATH_ROW])
    od = tmp_path / "ev"
    main(["eval", "--model", "my/model", "--data", str(data), "--output-dir", str(od),
          "--set", "eval.judge_model=judge/9B"])
    summ = json.loads((od / "eval_summary.json").read_text())
    assert summ["decoding"]["judge_model"] is None
    assert [b.name for b in built] == ["my/model"]  # no judge engine loaded


def test_cli_eval_mixed_file_with_math_row_first(tmp_path, built):
    data = write(tmp_path / "mixed.jsonl", [MATH_ROW, JUDGE_ROWS[0]])
    od = tmp_path / "ev"
    assert main(["eval", "--model", "my/model", "--data", str(data), "--output-dir", str(od),
                 "--set", "eval.judge_model=judge/9B"]) == 0
    preds = {p["id"]: p for p in read_jsonl(od / "mixed.predictions.jsonl")}
    assert preds["j1"]["judge_verdict"] is True
