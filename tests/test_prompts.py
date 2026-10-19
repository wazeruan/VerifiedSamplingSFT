import random

import pytest

from vsft.prompts import (
    PROPOSAL_PROMPTS,
    proposal_messages,
    segment_steps,
    student_messages,
    verifier_messages,
)


def _check_invariant(resp, steps):
    for s in steps:
        assert resp[s.start : s.end] == s.text
        assert s.text == s.text.strip() and s.text
    assert [s.start for s in steps] == sorted(s.start for s in steps)
    for a, b in zip(steps, steps[1:], strict=False):
        assert a.end <= b.start


def test_segment_lines_with_blank_and_indent():
    resp = "  First step.\n\n\tSecond step  \n   \nThird\r\n"
    steps = segment_steps(resp)
    assert [s.text for s in steps] == ["First step.", "Second step", "Third"]
    _check_invariant(resp, steps)
    assert resp[: steps[1].start] == "  First step.\n\n\t"


def test_segment_long_line_split_at_sentences():
    sent = "This is a fairly long sentence number {}. "
    line = "".join(sent.format(i) for i in range(20)).rstrip()
    resp = "Intro\n" + line + "\nDone."
    steps = segment_steps(resp, max_step_chars=100)
    _check_invariant(resp, steps)
    assert len(steps) > 3
    assert steps[0].text == "Intro" and steps[-1].text == "Done."
    assert steps[1].text == "This is a fairly long sentence number 0."


def test_segment_short_line_not_split():
    resp = "A. B. C. D."
    assert [s.text for s in segment_steps(resp, max_step_chars=400)] == [resp]


def test_segment_empty():
    assert segment_steps("") == []
    assert segment_steps("\n \n\t\n") == []


def test_segment_invariant_fuzz():
    rng = random.Random(0)
    alphabet = ["a", "B", " ", ".", "\n", "!", "?", "\\", "$", "(", "\t", "x", "Q", "é"]
    for _ in range(300):
        resp = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 300)))
        steps = segment_steps(resp, max_step_chars=rng.choice([20, 50, 400]))
        _check_invariant(resp, steps)


def test_messages():
    assert student_messages("Q?")[-1] == {"role": "user", "content": "Q?"}
    m = proposal_messages("What {is} x?", "exp {0}", "generic")
    assert "What {is} x?" in m[0]["content"] and "exp {0}" in m[0]["content"]
    assert "\\boxed{}" in m[0]["content"]
    assert "Task: Q" in proposal_messages("Q", "E", "chemistry")[0]["content"]
    with pytest.raises(KeyError):
        proposal_messages("Q", "E", "nope")
    assert set(PROPOSAL_PROMPTS) >= {"generic", "chemistry"}


def test_verifier_messages():
    body = verifier_messages("Q", "E", ["s1", "s2"], "42")[0]["content"]
    assert "[1] s1\n[2] s2" in body
    assert "Reference final answer: 42" in body
    assert '{"correct": true}' in body
    body = verifier_messages("Q", "E", ["s1"], None)[0]["content"]
    assert "Reference final answer" not in body
