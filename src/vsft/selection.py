"""Winner selection among regenerated candidates.

Replaces the paper's MH random walk: each candidate is scored by its mean per-token
log-probability under the student (plain task prompt). The highest score wins; every
candidate within `tie_epsilon` of the best is considered tied, and the tie is broken by
token-level F1 overlap with the expert trace (keeps the most information-faithful one).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


@dataclass
class Candidate:
    response: str  # full response = kept prefix + new continuation
    mean_logprob: float
    num_tokens: int
    finished: bool
    source: str = "sample"  # sample | expert
    empty_continuation: bool = False  # the model added nothing after the kept prefix


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?|\S", text.lower())


def token_f1(a: str, b: str) -> float:
    ca, cb = Counter(_words(a)), Counter(_words(b))
    common = sum((ca & cb).values())
    if common == 0:
        return 0.0
    prec, rec = common / sum(ca.values()), common / sum(cb.values())
    return 2 * prec * rec / (prec + rec)


@dataclass
class Selection:
    index: int
    tied: list[int]  # indices inside the epsilon band (includes the winner)
    pool: list[int]  # indices eligible for selection


def select_winner(
    candidates: list[Candidate],
    expert: str,
    tie_epsilon: float,
    drop_truncated: bool = True,
) -> Selection:
    if not candidates:
        raise ValueError("no candidates to select from")
    pool = [i for i, c in enumerate(candidates) if c.num_tokens > 0]
    nonempty = [i for i in pool if not candidates[i].empty_continuation]
    pool = nonempty or pool
    if drop_truncated:
        finished = [i for i in pool if candidates[i].finished]
        pool = finished or pool
    pool = pool or list(range(len(candidates)))
    best = max(candidates[i].mean_logprob for i in pool)
    tied = [i for i in pool if candidates[i].mean_logprob >= best - tie_epsilon]
    if len(tied) == 1:
        return Selection(tied[0], tied, pool)
    overlap = {i: token_f1(candidates[i].response, expert) for i in tied}
    # Ties on overlap fall back to the higher likelihood, then the lower index.
    winner = max(tied, key=lambda i: (overlap[i], candidates[i].mean_logprob, -i))
    return Selection(winner, tied, pool)
