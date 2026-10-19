import pytest

import vsft.data as data_mod
from vsft.io import read_jsonl


class FakeDS(list):
    pass


def _math_rows(subject, split):
    return FakeDS(
        [
            {"problem": f"{subject}-{split}-p0", "solution": "so \\boxed{1}", "level": "Level 3"},
            {"problem": f"{subject}-{split}-p1", "solution": "so \\boxed{2}", "level": "Level 1"},
            {"problem": f"{subject}-{split}-p2", "solution": "no box", "level": "Level 5"},
            {"problem": f"{subject}-{split}-p3", "solution": "\\boxed{\\frac{1}{2}}",
             "level": "Level 5"},
        ]
    )


@pytest.fixture
def fake_hub(monkeypatch):
    import datasets

    calls = []

    def load_dataset(name, config=None, split=None):
        calls.append((name, config, split))
        if name == "EleutherAI/hendrycks_math":
            return _math_rows(config, split)
        if name == "HuggingFaceH4/MATH-500":
            # second problem duplicates a MATH pool problem (modulo case/whitespace)
            return FakeDS([{"problem": "P", "solution": "S \\boxed{3}", "answer": "3"},
                           {"problem": "  ALGEBRA-test-p0 ", "solution": "s", "answer": "1"}])
        if name == "openai/gsm8k":
            return FakeDS([
                {"question": "Q1", "answer": "a <<1+1=2>>\nso 2\n#### 1,234"},
                {"question": "Q2", "answer": "x #### y #### 7"},
            ])
        raise AssertionError(name)

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    return calls


def test_prepare_math(tmp_path, fake_hub):
    n = data_mod.prepare_math(tmp_path, test_size=3, seed=0)
    train = list(read_jsonl(tmp_path / "math_l345_train.jsonl"))
    test = list(read_jsonl(tmp_path / "math_l345_test.jsonl"))
    # 7 subjects x 2 splits x 2 kept rows (Level 1 and the unboxed one are filtered),
    # minus the one problem that is also in MATH-500
    assert n == {"train": 24, "test": 3} and len(train) == 24 and len(test) == 3
    rows = train + test
    assert len({r["id"] for r in rows}) == 27
    assert "algebra-test-p0" not in {r["question"] for r in rows}
    assert {r["level"] for r in rows} == {"Level 3", "Level 5"}
    assert {r["answer"] for r in rows} == {"1", "\\frac{1}{2}"}
    assert all(r["answer_type"] == "math" and r["expert"] for r in rows)
    assert {r["subject"] for r in rows} == set(data_mod.MATH_SUBJECTS)
    assert len(fake_hub) == 15 and ("HuggingFaceH4/MATH-500", None, "test") in fake_hub
    # deterministic split
    data_mod.prepare_math(tmp_path / "again", test_size=3, seed=0)
    assert list(read_jsonl(tmp_path / "again" / "math_l345_test.jsonl")) == test
    with pytest.raises(ValueError, match="test_size"):
        data_mod.prepare_math(tmp_path / "x", test_size=27)


def test_prepare_math_levels(tmp_path, fake_hub):
    data_mod.prepare_math(tmp_path, levels=(1,), test_size=1)
    rows = list(read_jsonl(tmp_path / "math_l345_train.jsonl"))
    assert {r["answer"] for r in rows} == {"2"}


def test_prepare_math500_and_gsm8k(tmp_path, fake_hub):
    assert data_mod.prepare_math500(tmp_path) == 2
    r = next(iter(read_jsonl(tmp_path / "math500_test.jsonl")))
    assert r == {"id": "math500-0", "question": "P", "expert": "S \\boxed{3}", "answer": "3",
                 "answer_type": "math"}
    assert data_mod.prepare_gsm8k(tmp_path) == 2
    a, b = read_jsonl(tmp_path / "gsm8k_test.jsonl")
    assert a["answer"] == "1234" and a["expert"] == "a <<1+1=2>>\nso 2"
    assert b["answer"] == "7" and b["expert"] == "x #### y"
    assert set(data_mod.PREPARERS) == {"math", "math500", "gsm8k"}
