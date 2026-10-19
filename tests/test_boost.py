import pytest
from fakes import ACCEPT, FakeBackend, boost_cfg, parse_proposal, reject_at, verifier_cfg

from vsft.boost import Booster, Example


def ex(i="e1", q="What is 12 + 30?", expert="12 + 30 = 42.\nSo \\boxed{42}.", ans="42"):
    return Example(id=i, question=q, expert=expert, answer=ans)


def scores_by_response(table, default=-5.0):
    return lambda q, r: table.get(r, default)


def make(propose, judge, score=None, bcfg=None, vcfg=None):
    be = FakeBackend(propose, judge, score)
    return be, Booster(be, bcfg or boost_cfg(), vcfg or verifier_cfg())


def test_example_from_dict():
    e = Example.from_dict({"id": 3, "question": "q", "expert": "e"})
    assert e.id == "3" and e.answer is None and e.answer_type == "math"
    with pytest.raises(ValueError, match="expert"):
        Example.from_dict({"id": 1, "question": "q"})


def test_winner_is_argmax_and_accepted():
    cands = ["A \\boxed{42}", "B \\boxed{42}", "C \\boxed{42}"]
    table = {cands[0]: -1.0, cands[1]: -0.3, cands[2]: -0.9}
    be, b = make(
        lambda q, p, r: [(c, True) for c in cands],
        lambda q, s: ACCEPT,
        scores_by_response(table),
        boost_cfg(num_candidates=3, tie_epsilon=0.0),
    )
    [res] = b.boost([ex()])
    assert res.status == "accepted" and res.response == cands[1]
    assert res.num_rounds == 1 and res.rounds[0]["winner"] == 1
    assert res.final_mean_logprob == pytest.approx(-0.3)
    assert res.rounds[0]["scores"] == [-1.0, -0.3, -0.9]
    assert len(be.proposal_calls) == 1


def test_epsilon_tie_break_prefers_expert_overlap():
    expert = "12 + 30 = 42.\nSo \\boxed{42}."
    far = "Unrelated text \\boxed{42}"
    near = "12 + 30 = 42. So \\boxed{42}"
    table = {far: -0.50, near: -0.51}
    for eps, want in [(0.02, near), (0.001, far)]:
        _, b = make(
            lambda q, p, r: [(far, True), (near, True)],
            lambda q, s: ACCEPT,
            scores_by_response(table),
            boost_cfg(tie_epsilon=eps),
        )
        [res] = b.boost([ex(expert=expert)])
        assert res.response == want, eps
        if eps == 0.02:
            assert res.rounds[0]["tied"] == [0, 1]


def test_truncated_candidate_excluded():
    good, trunc = "fin \\boxed{42}", "trunc"
    _, b = make(
        lambda q, p, r: [(trunc, False), (good, True)],
        lambda q, s: ACCEPT,
        scores_by_response({trunc: -0.1, good: -3.0}),
    )
    [res] = b.boost([ex()])
    assert res.response == good and res.rounds[0]["finished"] == [False, True]


def test_prefix_restart_at_error_step_and_passed_to_next_round():
    r0 = "Step one ok.\nStep two wrong.\nSo \\boxed{41}."
    cont = "\nStep two right.\nSo \\boxed{42}."
    prefixes = []

    def propose(q, prefix, r):
        prefixes.append(prefix)
        return [(r0, True), (r0 + " x", True)] if r == 0 else [(cont, True), ("zz", True)]

    def judge(q, steps):
        return reject_at(2) if steps[1] == "Step two wrong." else ACCEPT

    be, b = make(propose, judge, scores_by_response({r0: -0.1, cont: -0.2}))
    [res] = b.boost([ex()])
    expected_prefix = "Step one ok."  # cut at end of step 1, separator excluded
    assert prefixes == ["", expected_prefix]
    # the prefix is the literal tail of the proposal prompt (assistant continuation)
    assert be.proposal_calls[1][0].endswith("Assistant: " + expected_prefix)
    assert parse_proposal(be.proposal_calls[1][0])[1] == expected_prefix
    r0log = res.rounds[0]
    assert r0log["reason"] == "llm_error_step" and r0log["error_step"] == 1
    assert r0log["restart_char"] == len(expected_prefix) == r0.index("Step two") - 1
    assert r0log["restart_token"] == len(expected_prefix.split())
    assert res.rounds[1]["prefix_chars"] == len(expected_prefix)
    assert res.rounds[1]["prefix_tokens"] == 3
    assert res.status == "accepted" and res.response == expected_prefix + cont
    # candidates in round 1 were scored as prefix + continuation
    assert expected_prefix + cont in be.score_calls[1][1]


def test_accepted_examples_stop_while_others_continue():
    def propose(q, prefix, r):
        return [("ok \\boxed{42}", True)]

    calls = {"hard": 0}

    def judge(q, steps):
        if q == "hard":
            calls["hard"] += 1
            return ACCEPT if calls["hard"] >= 3 else reject_at(1)
        return ACCEPT

    be, b = make(propose, judge, bcfg=boost_cfg(num_candidates=1, max_rounds=5))
    res = b.boost([ex("a", q="easy"), ex("b", q="hard"), ex("c", q="easy2")])
    assert [r.num_rounds for r in res] == [1, 3, 1]
    assert all(r.status == "accepted" for r in res)
    assert [len(c) for c in be.proposal_calls] == [3, 1, 1]
    assert [len(c) for c in be.verifier_calls] == [3, 1, 1]
    assert [r.id for r in res] == ["a", "b", "c"]


@pytest.mark.parametrize(
    "mode,status",
    [("expert", "fallback_expert"), ("drop", "dropped"), ("last", "fallback_last")],
)
def test_on_failure_modes(mode, status):
    expert = "12 + 30 = 42.\nSo \\boxed{42}."
    outs = iter(["w0 \\boxed{1}", "w1 \\boxed{1}", "w2 \\boxed{1}"])
    score = lambda q, r: -0.25 if r == expert else -0.75  # noqa: E731
    be, b = make(
        lambda q, p, r: [(next(outs), True)],
        lambda q, s: reject_at(1),
        score,
        boost_cfg(num_candidates=1, max_rounds=3, on_failure=mode),
    )
    [res] = b.boost([ex(expert=expert)])
    assert res.status == status and res.num_rounds == 3
    assert len(be.proposal_calls) == 3
    assert res.expert_mean_logprob == pytest.approx(-0.25)
    if mode == "expert":
        assert res.response == expert and res.final_mean_logprob == pytest.approx(-0.25)
    elif mode == "drop":
        assert res.response is None and res.final_mean_logprob is None
    else:
        assert res.response == "w2 \\boxed{1}" and res.final_mean_logprob == pytest.approx(-0.75)


def test_include_expert_candidate_round0_only():
    expert = "12 + 30 = 42.\nSo \\boxed{42}."
    table = {expert: -0.01}

    def judge(q, steps):
        return reject_at(2)

    be, b = make(
        lambda q, p, r: [("s1 \\boxed{0}", True), ("s2 \\boxed{0}", True)],
        judge,
        scores_by_response(table, default=-2.0),
        boost_cfg(include_expert_candidate=True, max_rounds=2),
    )
    [res] = b.boost([ex(expert=expert)])
    r0, r1 = res.rounds
    assert len(r0["scores"]) == 3 and r0["winner"] == 2 and r0["winner_source"] == "expert"
    assert len(r1["scores"]) == 2 and r1["winner_source"] == "sample"
    # rejected expert at step 2 -> its first line becomes the next prefix
    assert r1["prefix_chars"] == len("12 + 30 = 42.")
    assert be.proposal_calls[1][0].endswith("Assistant: 12 + 30 = 42.")


def test_unlocalized_backoff_and_reset():
    five = "s1\ns2\ns3\ns4\ns5 \\boxed{42}"
    verdicts = iter(["???", "???", reject_at(1), "???"])
    prefixes = []

    def propose(q, prefix, r):
        prefixes.append(prefix)
        return [(five if r == 0 else "\nt1\nt2\nt3\nt4\nt5", True)]

    _, b = make(
        propose,
        lambda q, s: next(verdicts),
        bcfg=boost_cfg(num_candidates=1, max_rounds=4, on_failure="drop"),
        vcfg=verifier_cfg(use_answer_check=False),
    )
    [res] = b.boost([ex()])
    reasons = [r["reason"] for r in res.rounds]
    assert reasons == ["unparsed", "unparsed", "llm_error_step", "unparsed"]
    # round 0: 5 steps, u=0 -> keep 5-2=3 steps
    assert prefixes[1] == "s1\ns2\ns3"
    # round 1: response = "s1\ns2\ns3" + 5 t-steps = 8 steps, u=1 -> keep 8-4=4 steps
    assert prefixes[2] == "s1\ns2\ns3\nt1"
    # round 2: localized at step 1 -> empty prefix, counter reset
    assert prefixes[3] == ""
    # round 3: 5 steps, u back to 0 -> error_step 3
    assert res.rounds[3]["error_step"] == 3
    assert res.status == "dropped"


def test_answer_check_override_blocks_acceptance():
    wrong, right = "So \\boxed{41}", "So \\boxed{42}"
    outs = iter([wrong, right])
    _, b = make(
        lambda q, p, r: [(next(outs), True)],
        lambda q, s: ACCEPT,
        bcfg=boost_cfg(num_candidates=1),
    )
    [res] = b.boost([ex()])
    assert [r["reason"] for r in res.rounds] == ["answer_mismatch", "accepted"]
    assert res.response == right

    _, b = make(
        lambda q, p, r: [(wrong, True)],
        lambda q, s: ACCEPT,
        bcfg=boost_cfg(num_candidates=1),
        vcfg=verifier_cfg(use_answer_check=False),
    )
    [res] = b.boost([ex()])
    assert res.status == "accepted" and res.response == wrong


def test_generation_kwargs_and_scoring_prompt():
    be, b = make(
        lambda q, p, r: [("x \\boxed{42}", True)] * 3,
        lambda q, s: ACCEPT,
        bcfg=boost_cfg(num_candidates=3, temperature=0.7, top_p=0.9, max_new_tokens=11),
    )
    b.boost([ex()])
    assert be.gen_kwargs[0] == dict(n=3, temperature=0.7, top_p=0.9, max_tokens=11)
    prompts, _ = be.score_calls[0]
    # scored under the plain student prompt: no expert trace in it
    assert all("expert solution" not in p.lower() for p in prompts)
    assert all("What is 12 + 30?" in p for p in prompts)
    # after the loop: one call scores the expert traces, then the emitted responses in full
    assert be.score_calls[-2][1] == ["12 + 30 = 42.\nSo \\boxed{42}."]
    assert be.score_calls[-1][1] == ["x \\boxed{42}"]


def test_chemistry_prompt_used():
    be, b = make(
        lambda q, p, r: [("x \\boxed{42}", True)],
        lambda q, s: ACCEPT,
        bcfg=boost_cfg(num_candidates=1, proposal_prompt="chemistry"),
    )
    b.boost([ex()])
    assert "expert chemist" in be.proposal_calls[0][0]


def test_result_dict_roundtrip():
    _, b = make(lambda q, p, r: [("x \\boxed{42}", True)], lambda q, s: ACCEPT,
                bcfg=boost_cfg(num_candidates=1))
    [res] = b.boost([ex()])
    d = res.to_dict()
    assert d["id"] == "e1" and isinstance(d["rounds"], list) and d["rounds"][0]["round"] == 0
    import json

    json.dumps(d)


# ---------------------------------------------------------------- milestone 2 additions


def test_example_from_dict_answer_cast_and_validation():
    e = Example.from_dict({"id": 1, "question": "q", "expert": "e", "answer": 0})
    assert e.answer == "0"
    e = Example.from_dict({"id": 1, "question": "q", "expert": "e", "answer": 2.5})
    assert e.answer == "2.5"
    for blank in ("", "   ", "\n"):
        e = Example.from_dict({"id": 1, "question": "q", "expert": "e", "answer": blank})
        assert e.answer is None
    for bad in ({"question": "", "expert": "e"}, {"question": "  \n", "expert": "e"},
                {"question": "q", "expert": ""}, {"question": 5, "expert": "e"}):
        with pytest.raises(ValueError, match="non-empty string"):
            Example.from_dict({"id": "x", **bad})


def test_zero_answer_still_gated():
    e = Example.from_dict({"id": "z", "question": "What is 1 - 1?", "expert": "1 - 1 = 0.",
                           "answer": 0})
    outs = iter(["So \\boxed{1}", "So \\boxed{0}"])
    _, b = make(lambda q, p, r: [(next(outs), True)], lambda q, s: ACCEPT,
                bcfg=boost_cfg(num_candidates=1))
    [res] = b.boost([e])
    assert [r["reason"] for r in res.rounds] == ["answer_mismatch", "accepted"]
    assert res.response == "So \\boxed{0}" and res.answer == "0"


@pytest.mark.parametrize("empty", ["", "   ", "\n\n"])
def test_empty_continuation_never_wins(empty):
    def propose(q, prefix, r):
        if r == 0:
            return [("A1.\nA2 wrong.", True), ("zz", True)]
        # an empty continuation is just the verified prefix, which scores highest
        return [(empty, True), ("\nA2 fixed \\boxed{42}", True)]

    def judge(q, steps):
        return reject_at(2) if "A2 wrong." in steps else ACCEPT

    def score(q, r):
        return -0.01 if r.strip() == "A1." else -2.0

    _, b = make(propose, judge, score, boost_cfg(num_candidates=2, max_rounds=2))
    [res] = b.boost([ex()])
    assert res.rounds[1]["winner"] == 1
    assert res.response == "A1.\nA2 fixed \\boxed{42}" and res.status == "accepted"


def test_score_scope_continuation_excludes_prefix_tokens():
    # prefix "p1 p2" scores -9 per token; continuation tokens score per scripted value
    r0 = "p1 p2\nbad step"

    def propose(q, prefix, r):
        if r == 0:
            return [(r0, True), (r0, True)]
        return [("\nc1 c2 \\boxed{42}", True), ("\nd1 d2 d3 \\boxed{42}", True)]

    def score(q, resp):
        toks = resp.split()
        return [-9.0 if t.startswith("p") else (-0.2 if t.startswith("c") else -0.5)
                for t in toks]

    def judge(q, steps):
        return reject_at(2) if "bad step" in steps else ACCEPT

    for scope in ("continuation", "full"):
        _, b = make(propose, judge, score, boost_cfg(score_scope=scope, tie_epsilon=0.0))
        [res] = b.boost([ex()])
        r1 = res.rounds[1]
        # candidate 0 tokens: p1 p2 c1 c2 \boxed{42}; candidate 1: p1 p2 d1 d2 d3 \boxed{42}
        full0 = (-9 * 2 - 0.2 - 0.2 - 0.5) / 5  # c wins only without the prefix
        full1 = (-9 * 2 - 0.5 * 4) / 6
        if scope == "continuation":
            assert r1["scores"] == [pytest.approx(-0.3, abs=1e-5),
                                    pytest.approx(-0.5, abs=1e-5)]
            assert r1["winner"] == 0
        else:
            assert r1["scores"] == [pytest.approx(full0, abs=1e-5), pytest.approx(full1, abs=1e-5)]
            assert r1["winner"] == 1


def test_final_logprob_is_full_response_under_continuation_scope():
    r0 = "p1 p2\nbad step"

    def propose(q, prefix, r):
        return [(r0, True)] if r == 0 else [("\nc1 \\boxed{42}", True)]

    def score(q, resp):
        return [-9.0 if t.startswith("p") else -1.0 for t in resp.split()]

    _, b = make(propose, lambda q, s: reject_at(2) if "bad step" in s else ACCEPT, score,
                boost_cfg(num_candidates=1, score_scope="continuation"))
    [res] = b.boost([ex()])
    assert res.status == "accepted" and res.response == "p1 p2\nc1 \\boxed{42}"
    # comparable with expert_mean_logprob (full response under the student prompt)
    assert res.final_mean_logprob == pytest.approx((-9 * 2 - 1 - 1) / 4)
    # selection still used the continuation-only score
    assert res.rounds[1]["scores"] == [pytest.approx(-1.0)]


@pytest.mark.parametrize("mode,status", [("expert", "fallback_expert"), ("drop", "dropped")])
def test_too_long_prompt_skipped_rest_continue(mode, status):
    long_q = "word " * 200
    be, b = make(
        lambda q, p, r: [("ok \\boxed{42}", True)],
        lambda q, s: ACCEPT,
        bcfg=boost_cfg(num_candidates=1, on_failure=mode),
    )
    be.max_len = 150  # proposal prompt is ~70 whitespace tokens + the question
    res = b.boost([ex("short"), ex("long", q=long_q.strip()), ex("short2", q="What is 1+1?")])
    by = {r.id: r for r in res}
    assert by["long"].skipped == "too_long" and by["long"].status == status
    assert by["long"].num_rounds == 0
    assert by["short"].status == by["short2"].status == "accepted"
    assert by["short"].skipped is None
    assert len(be.proposal_calls) == 1 and len(be.proposal_calls[0]) == 2
    assert all(long_q.strip() not in p for p in be.proposal_calls[0])


def test_too_long_after_prefix_growth_uses_last():
    # prompt fits in round 0, the kept prefix pushes it over the limit in round 1
    big = "\n".join(f"step{i} " + "w " * 10 for i in range(8)) + "\nend \\boxed{1}"
    be, b = make(
        lambda q, p, r: [(big, True)],
        lambda q, s: reject_at(9),
        bcfg=boost_cfg(num_candidates=1, on_failure="last", max_rounds=3),
    )
    plen = be.count_tokens(b.backend.render(
        __import__("vsft.prompts", fromlist=["x"]).proposal_messages(ex().question, ex().expert)))
    be.max_len = plen + 32 + 40  # room for round 0, not for an ~80-token prefix
    [res] = b.boost([ex()])
    assert res.num_rounds == 1 and res.skipped == "too_long"
    assert res.status == "fallback_last" and res.response == big


def test_scores_none_instead_of_inf_and_json_safe():
    import json

    def score(q, r):
        return [] if r.startswith("EMPTY") else -1.0

    _, b = make(
        lambda q, p, r: [("EMPTY tokens", True), ("x \\boxed{42}", True)],
        lambda q, s: ACCEPT,
        score,
    )
    [res] = b.boost([ex()])
    assert res.rounds[0]["scores"] == [None, -1.0]
    json.dumps(res.to_dict(), allow_nan=False)

    # expert with no scorable tokens -> expert_mean_logprob None (not -inf)
    _, b = make(lambda q, p, r: [("x \\boxed{42}", True)], lambda q, s: reject_at(1),
                lambda q, r: [] if r == ex().expert else -1.0,
                boost_cfg(num_candidates=1, max_rounds=1))
    [res] = b.boost([ex()])
    assert res.expert_mean_logprob is None and res.final_mean_logprob is None
    json.dumps(res.to_dict(), allow_nan=False)
