import pytest
from fakes import FakeBackend, verifier_cfg

from vsft.verifier import Verifier, VerifyRequest, decide, parse_verifier_output


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"correct": true}', (True, None)),
        ('{"correct": false, "first_error_step": 3}', (False, 3)),
        ('```json\n{"correct": false, "first_error_step": 2}\n```', (False, 2)),
        ('Let me think. \\boxed{5} ... {"correct": true} trailing', (True, None)),
        ('{"correct": true}\nOn reflection: {"correct": false, "first_error_step": 1}', (False, 1)),
        ('{"correct": false, "first_error_step": "4"}', (False, 4)),
        ('{"correct": false, "first_error_step": null}', (False, None)),
        ('{"correct": false, "first_error_step": "two"}', (False, None)),
        ('{"result": {"correct": true}}', (True, None)),
        ("correct: false, first_error_step: 5", (False, 5)),
        ('"correct": False, "first_error_step": [2]', (False, 2)),  # not valid JSON
        ("first_error_step: 3", (False, 3)),
        ("CORRECT: TRUE", (True, None)),
        ("I cannot decide.", (None, None)),
        ("", (None, None)),
    ],
)
def test_parse_verifier_output(raw, expected):
    assert parse_verifier_output(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("{'correct': False, 'first_error_step': 2}", (False, 2)),
        ("Verdict: {'correct': True}", (True, None)),
        ("{'correct': 'false', 'first_error_step': '3'}", (False, 3)),
        ("{'correct': None, 'first_error_step': 1}", (False, 1)),
        ("{'correct': __import__('os')}", (None, None)),  # literal_eval only, no code
    ],
)
def test_parse_python_dict_style(raw, expected):
    assert parse_verifier_output(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"correct": "false", "first_error_step": 2}', (False, 2)),
        ('{"correct": "True"}', (True, None)),
        ('{"correct": " FALSE "}', (False, None)),
        ('{"correct": "maybe"}', (None, None)),
        ('{"correct": null, "first_error_step": 3}', (False, 3)),
        ('{"correct": "unsure", "first_error_step": 1}', (False, 1)),
        ('{"correct": false, "first_error_step": true}', (False, None)),
    ],
)
def test_parse_string_bools_and_step_implies_false(raw, expected):
    assert parse_verifier_output(raw) == expected


RESP = "Step one.\nStep two.\nStep three.\nStep four.\nSo \\boxed{42}."


def req(resp=RESP, ref="42", u=0):
    return VerifyRequest("Q", "E", resp, ref, "math", u)


def starts(resp):
    from vsft.prompts import segment_steps

    return [s.start for s in segment_steps(resp)]


def cut(resp, k):
    """Expected restart offset for wrong step k: end of step k-1 (0 when k == 0)."""
    from vsft.prompts import segment_steps

    return segment_steps(resp)[k - 1].end if k > 0 else 0


def test_restart_offset_helper():
    from vsft.prompts import segment_steps
    from vsft.verifier import restart_offset

    steps = segment_steps(RESP)
    assert restart_offset(steps, 0) == 0
    assert restart_offset(steps, 2) == len("Step one.\nStep two.")
    for k in range(1, len(steps)):
        kept = RESP[: restart_offset(steps, k)]
        assert kept == kept.rstrip()  # never ends in the separator whitespace


def test_decide_accept():
    v = decide(req(), '{"correct": true}', 400, True)
    assert v.correct and v.reason == "accepted" and v.restart_char == len(RESP)
    assert v.answer_ok is True and v.num_steps == 5


def test_decide_error_step_offset():
    v = decide(req(), '{"correct": false, "first_error_step": 3}', 400, True)
    assert not v.correct and v.reason == "llm_error_step" and v.error_step == 2
    assert v.restart_char == RESP.index("Step three") - 1
    assert RESP[: v.restart_char] == "Step one.\nStep two."


def test_decide_answer_override():
    v = decide(req(ref="41"), '{"correct": true}', 400, True)
    assert not v.correct and v.reason == "answer_mismatch" and v.answer_ok is False
    # backoff from the end: 5 - 2 = step index 3
    assert v.error_step == 3 and v.restart_char == cut(RESP, 3)
    # disabled answer check -> accepted
    assert decide(req(ref="41"), '{"correct": true}', 400, False).correct
    # no reference -> accepted
    assert decide(req(ref=None), '{"correct": true}', 400, True).correct


def test_decide_answer_override_keeps_llm_step():
    v = decide(req(ref="41"), '{"correct": false, "first_error_step": 2}', 400, True)
    assert v.reason == "llm_error_step" and v.error_step == 1


@pytest.mark.parametrize("u,k", [(0, 3), (1, 1), (2, 0), (5, 0)])
def test_decide_backoff_exponential(u, k):
    v = decide(req(u=u), "garbage", 400, True)
    assert v.reason == "unparsed" and v.error_step == k and v.restart_char == cut(RESP, k)


@pytest.mark.parametrize("raw", ['{"correct": false, "first_error_step": 0}',
                                 '{"correct": false, "first_error_step": 99}',
                                 '{"correct": false}'])
def test_decide_out_of_range_step_backs_off(raw):
    v = decide(req(), raw, 400, True)
    assert v.reason == "unparsed" and v.error_step == 3


def test_decide_error_step_one_restarts_from_scratch():
    v = decide(req(resp="  lead\nnext"), '{"correct": false, "first_error_step": 1}', 400, True)
    assert v.restart_char == 0 and v.error_step == 0


def test_accept_unparsed_if_answer_ok():
    v = decide(req(), "no json here", 400, True, accept_unparsed_if_answer_ok=True)
    assert v.correct and v.reason == "accepted_answer_only" and v.restart_char == len(RESP)
    # off by default
    assert not decide(req(), "no json here", 400, True).correct
    # needs a matching answer
    v = decide(req(ref="41"), "no json", 400, True, accept_unparsed_if_answer_ok=True)
    assert not v.correct and v.reason == "answer_mismatch"
    # no reference -> answer_ok None -> not accepted
    v = decide(req(ref=None), "no json", 400, True, accept_unparsed_if_answer_ok=True)
    assert not v.correct and v.reason == "unparsed"
    # an explicit verifier rejection is never overridden
    v = decide(req(), '{"correct": false, "first_error_step": 2}', 400, True, True)
    assert not v.correct and v.reason == "llm_error_step"
    # answer check disabled -> nothing to accept on
    assert not decide(req(), "no json", 400, False, True).correct


def test_answer_gate_applies_to_zero_reference():
    v = decide(req(resp="So \\boxed{1}", ref="0"), '{"correct": true}', 400, True)
    assert not v.correct and v.answer_ok is False
    v = decide(req(resp="So \\boxed{0}", ref="0"), '{"correct": true}', 400, True)
    assert v.correct


def test_verifier_passes_accept_unparsed_flag():
    be = FakeBackend(lambda *a: [], lambda q, s: "???")
    ver = Verifier(be, verifier_cfg(accept_unparsed_if_answer_ok=True))
    [v] = ver.verify([VerifyRequest("Q", "E", "x \\boxed{7}", "7")])
    assert v.correct and v.reason == "accepted_answer_only"


def test_decide_empty():
    v = decide(req(resp="  \n"), '{"correct": true}', 400, True)
    assert not v.correct and v.reason == "empty" and v.restart_char == 0


def test_verifier_batch_prompts():
    seen = []

    def judge(q, steps):
        seen.append((q, steps))
        return '{"correct": true}' if q == "Q1" else '{"correct": false, "first_error_step": 1}'

    be = FakeBackend(lambda *a: [], judge)
    ver = Verifier(be, verifier_cfg(temperature=0.0, max_new_tokens=17))
    vs = ver.verify([
        VerifyRequest("Q1", "E", "a\nb \\boxed{1}", "1"),
        VerifyRequest("Q2", "E", "x\ny", None),
    ])
    assert [v.correct for v in vs] == [True, False]
    assert seen == [("Q1", ["a", "b \\boxed{1}"]), ("Q2", ["x", "y"])]
    assert be.gen_kwargs[-1] == dict(n=1, temperature=0.0, top_p=1.0, max_tokens=17)
    assert len(be.verifier_calls) == 1
    assert "Reference final answer: 1" in be.verifier_calls[0][0]
    assert ver.verify([]) == []


def test_decide_verifier_too_long_never_accepts():
    for raw in ('{"correct": true}', '{"correct": false, "first_error_step": 2}', "???"):
        v = decide(req(), raw, 400, True, accept_unparsed_if_answer_ok=True, prompt_too_long=True)
        assert not v.correct and v.reason == "verifier_too_long"
        assert v.error_step == 3 and v.restart_char == cut(RESP, 3)  # unlocalized backoff
    v = decide(req(u=1), '{"correct": true}', 400, True, prompt_too_long=True)
    assert v.error_step == 1


def test_verifier_flags_too_long_prompts():
    be = FakeBackend(lambda *a: [], lambda q, s: '{"correct": true}')
    ver = Verifier(be, verifier_cfg())
    short = VerifyRequest("Q", "E", "x \\boxed{7}", "7")
    longr = VerifyRequest("Q2", "E", "y " * 300 + "\\boxed{7}", "7")
    n_short = be.count_tokens(be.render(__import__("vsft.prompts", fromlist=["x"])
                                       .verifier_messages("Q", "E", ["x \\boxed{7}"], "7")))
    be.max_len = n_short + 1
    a, b = ver.verify([short, longr])
    assert a.correct and a.reason == "accepted"
    assert not b.correct and b.reason == "verifier_too_long"
