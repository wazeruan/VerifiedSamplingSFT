import math

import pytest

from vsft.selection import Candidate, select_winner, token_f1


def C(resp, lp, n=5, fin=True):
    return Candidate(resp, lp, n, fin)


EXPERT = "add twelve and thirty to get forty two"


def test_argmax():
    cands = [C("a", -1.0), C("b", -0.5), C("c", -0.9)]
    sel = select_winner(cands, EXPERT, tie_epsilon=0.0)
    assert sel.index == 1 and sel.tied == [1] and sel.pool == [0, 1, 2]


def test_epsilon_tie_broken_by_overlap():
    cands = [C("unrelated words here", -0.50), C("add twelve and thirty to get forty two", -0.51)]
    assert select_winner(cands, EXPERT, tie_epsilon=0.02).index == 1
    assert select_winner(cands, EXPERT, tie_epsilon=0.02).tied == [0, 1]
    assert select_winner(cands, EXPERT, tie_epsilon=0.005).index == 0


def test_tie_band_boundary_inclusive():
    cands = [C("x", -0.5), C(EXPERT, -0.75)]
    assert select_winner(cands, EXPERT, tie_epsilon=0.25).index == 1


def test_equal_overlap_falls_back_to_logprob_then_index():
    cands = [C("same", -0.51), C("same", -0.50), C("same", -0.50)]
    assert select_winner(cands, EXPERT, tie_epsilon=0.1).index == 1


def test_truncated_excluded_when_finished_exists():
    cands = [C("a", -0.1, fin=False), C("b", -2.0, fin=True)]
    sel = select_winner(cands, EXPERT, 0.0, drop_truncated=True)
    assert sel.index == 1 and sel.pool == [1]
    assert select_winner(cands, EXPERT, 0.0, drop_truncated=False).index == 0


def test_all_truncated_keeps_all():
    cands = [C("a", -0.1, fin=False), C("b", -2.0, fin=False)]
    assert select_winner(cands, EXPERT, 0.0).index == 0


def test_empty_candidates_excluded():
    cands = [C("", -math.inf, n=0), C("b", -3.0)]
    assert select_winner(cands, EXPERT, 0.0).index == 1
    only_empty = [C("", -math.inf, n=0), C("", -math.inf, n=0)]
    assert select_winner(only_empty, EXPERT, 0.0).index in (0, 1)


def test_no_candidates():
    with pytest.raises(ValueError):
        select_winner([], EXPERT, 0.0)


def test_token_f1():
    assert token_f1("a b c", "a b c") == pytest.approx(1.0)
    assert token_f1("a b", "c d") == 0.0
    assert token_f1("", "x") == 0.0
    assert token_f1("12 + 30 = 42", "12+30=42") == pytest.approx(1.0)


def test_empty_continuation_excluded_from_pool():
    cands = [
        Candidate("prefix only", -0.01, 2, True, empty_continuation=True),
        Candidate("prefix + more", -1.0, 3, True),
        Candidate("prefix + trunc", -0.5, 3, False),
    ]
    sel = select_winner(cands, EXPERT, 0.0)
    assert sel.index == 1 and sel.pool == [1]
    assert select_winner(cands, EXPERT, 0.0, drop_truncated=False).index == 2
    # if every candidate is an empty continuation, fall back to them
    only = [Candidate("p", -0.2, 1, True, empty_continuation=True),
            Candidate("p", -0.1, 1, True, empty_continuation=True)]
    assert select_winner(only, EXPERT, 0.0).index == 1
