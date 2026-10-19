import json

import pytest
from fakes import FakeBackend, parse_student

from vsft.backends.base import Generation
from vsft.evaluate import evaluate
from vsft.io import read_jsonl


class EvalBackend(FakeBackend):
    def __init__(self, answers):
        super().__init__(lambda *a: [], lambda *a: "")
        self.answers = answers  # question -> (text, finished)
        self.calls = []

    def generate(self, prompts, n, temperature, top_p, max_tokens):
        self.calls.append((len(prompts), n, temperature, max_tokens))
        return [[Generation(*self.answers[parse_student(p)])] for p in prompts]


def cfg(**kw):
    from vsft.config import EvalConfig

    return EvalConfig(**{"max_new_tokens": 9, "temperature": 0.0, "batch_size": 2, **kw})


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


ROWS = [
    {"id": "a", "question": "qa", "answer": "1", "level": "Level 3"},
    {"id": "b", "question": "qb", "answer": "\\frac{1}{2}", "level": "Level 3"},
    {"id": "c", "question": "qc", "answer": "C", "answer_type": "mc", "level": "Level 5"},
    {"id": "d", "question": "qd", "answer": "4", "subject": "algebra"},
]
ANS = {
    "qa": ("so \\boxed{1}", True),
    "qb": ("so \\boxed{0.5}", True),
    "qc": ("The answer is (B).", True),
    "qd": ("I ran out of", False),
}


def test_evaluate_metrics_and_files(tmp_path):
    data = write(tmp_path / "set.jsonl", ROWS)
    be = EvalBackend(ANS)
    m = evaluate(be, data, tmp_path / "out", cfg())
    assert m["num_examples"] == 4 and m["accuracy"] == pytest.approx(0.5)
    assert m["accuracy_by_level"] == {"Level 3": 1.0, "Level 5": 0.0}
    assert m["num_unfinished"] == 1 and m["num_no_answer"] == 1
    assert be.calls == [(2, 1, 0.0, 9), (2, 1, 0.0, 9)]
    preds = list(read_jsonl(tmp_path / "out" / "set.predictions.jsonl"))
    assert [p["correct"] for p in preds] == [True, True, False, False]
    assert preds[2]["prediction"] == "B" and preds[3]["prediction"] is None
    assert preds[3]["subject"] == "algebra" and "level" not in preds[3]
    assert json.loads((tmp_path / "out" / "set.metrics.json").read_text()) == m


def test_evaluate_limit_and_errors(tmp_path):
    data = write(tmp_path / "set.jsonl", ROWS)
    assert evaluate(EvalBackend(ANS), data, tmp_path / "o", cfg(), limit=1)["num_examples"] == 1
    with pytest.raises(ValueError, match="no rows"):
        evaluate(EvalBackend(ANS), write(tmp_path / "e.jsonl", []), tmp_path / "o", cfg())
    bad = write(tmp_path / "b.jsonl", [{"id": "x", "question": "qa"}])
    with pytest.raises(ValueError, match="no reference answer"):
        evaluate(EvalBackend(ANS), bad, tmp_path / "o", cfg())


def test_evaluate_zero_answer(tmp_path):
    data = write(tmp_path / "z.jsonl", [{"id": "z", "question": "qz", "answer": 0},
                                        {"id": "y", "question": "qy", "answer": 0}])
    be = EvalBackend({"qz": ("\\boxed{0}", True), "qy": ("\\boxed{1}", True)})
    m = evaluate(be, data, tmp_path / "o", cfg())
    assert m["accuracy"] == 0.5
    bad = write(tmp_path / "e.jsonl", [{"id": "e", "question": "qz", "answer": ""}])
    with pytest.raises(ValueError, match="no reference answer"):
        evaluate(be, bad, tmp_path / "o", cfg())
