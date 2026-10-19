"""Verifier-guided boosting of expert traces (replaces the paper's MH random walk).

For each example (question q, expert trace tau, optional reference answer a):

    prefix <- ""                                    # verified-so-far part of the response
    for round r = 0 .. max_rounds-1:
        K candidates  x_k = prefix + continuation_k,   continuation_k ~ kappa(. | q, tau, prefix)
        score_k       = mean_t log p_student(x_k,t | q, x_k,<t)   # plain task prompt
        winner        = argmax score, ties within eps broken by overlap with tau
        verdict       = Verifier(q, tau, a, winner)    # larger verifier model, prompted with tau
        if verdict.correct: emit winner; stop
        prefix        = winner[: verdict.restart_char]  # regenerate from the wrong step
    on exhaustion: emit tau (on_failure=expert), drop, or emit last winner

Examples are advanced together in batches so every round issues one batched generate,
one batched scoring pass and one batched verifier call.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field

from vsft.backends.base import Backend, mean_logprob
from vsft.prompts import proposal_messages, student_messages
from vsft.selection import Candidate, select_winner
from vsft.verifier import Verifier, VerifyRequest

log = logging.getLogger(__name__)


@dataclass
class Example:
    id: str
    question: str
    expert: str
    answer: str | None = None
    answer_type: str = "math"

    @classmethod
    def from_dict(cls, d: dict) -> Example:
        missing = {"id", "question", "expert"} - set(d)
        if missing:
            raise ValueError(f"example missing fields {sorted(missing)}: {str(d)[:200]}")
        for key in ("question", "expert"):
            if not isinstance(d[key], str) or not d[key].strip():
                raise ValueError(f"example {d['id']!r}: '{key}' must be a non-empty string")
        answer = d.get("answer")
        return cls(
            id=str(d["id"]),
            question=d["question"],
            expert=d["expert"],
            # An empty reference answer means "no reference" (skip the answer gate).
            answer=None if answer is None or str(answer).strip() == "" else str(answer),
            answer_type=d.get("answer_type", "math"),
        )


@dataclass
class RoundLog:
    round: int
    prefix_chars: int
    prefix_tokens: int
    scores: list[float | None]
    finished: list[bool]
    winner: int
    tied: list[int]
    winner_source: str
    correct: bool
    reason: str
    error_step: int | None
    num_steps: int
    restart_char: int
    restart_token: int  # token index in the winner where regeneration restarts


@dataclass
class _State:
    ex: Example
    prefix: str = ""
    unlocalized: int = 0
    rounds: list[RoundLog] = field(default_factory=list)
    last_winner: Candidate | None = None
    done: bool = False
    accepted: bool = False
    skipped: str | None = None  # reason the example left the loop early (e.g. too_long)


@dataclass
class BoostResult:
    id: str
    question: str
    response: str | None  # None when dropped
    status: str  # accepted | fallback_expert | fallback_last | dropped
    skipped: str | None
    answer: str | None
    answer_type: str
    expert: str
    num_rounds: int
    final_mean_logprob: float | None
    expert_mean_logprob: float | None
    rounds: list[dict]
    seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def _finite(x: float | None) -> float | None:
    """JSON-safe score: None instead of +-inf/nan."""
    return round(x, 5) if x is not None and math.isfinite(x) else None


class BoostStopped(Exception):
    """A stop was requested mid-batch; the batch's partial results are discarded."""


class Booster:
    def __init__(self, backend: Backend, boost_cfg, verifier_cfg, verifier_backend=None):
        # The student generates and scores; the (possibly larger) verifier grades.
        self.backend = backend
        self.cfg = boost_cfg
        self.verifier = Verifier(verifier_backend or backend, verifier_cfg)

    # -- scoring -----------------------------------------------------------------------
    def score(
        self, questions: list[str], responses: list[str], skip: list[int] | None = None
    ) -> list[tuple[float, int]]:
        """Mean log-prob of each response under the plain student prompt.

        `skip[i]` leading tokens are excluded from the mean (used by score_scope=
        "continuation" to drop the shared prefix); the token count returned is the full one.
        """
        prompts = [self.backend.render(student_messages(q)) for q in questions]
        lps = self.backend.token_logprobs(prompts, responses)
        skip = skip or [0] * len(lps)
        out = []
        for lp, k in zip(lps, skip, strict=True):
            tail = lp[k:] if k < len(lp) else []
            out.append((mean_logprob(tail), len(lp)))
        return out

    def _too_long(self, prompt: str) -> bool:
        max_len = getattr(self.backend, "max_len", None)
        # Require room for at least a short continuation.
        return max_len is not None and self.backend.count_tokens(prompt) >= max_len - 32

    # -- one round for a batch of active states ------------------------------------------
    def _round(self, states: list[_State], r: int) -> None:
        c = self.cfg
        prompts = [
            self.backend.render(
                proposal_messages(s.ex.question, s.ex.expert, c.proposal_prompt), s.prefix
            )
            for s in states
        ]
        keep = [i for i, p in enumerate(prompts) if not self._too_long(p)]
        for i in set(range(len(states))) - set(keep):
            states[i].done, states[i].skipped = True, "too_long"
            log.warning("example %s: proposal prompt exceeds max_model_len", states[i].ex.id)
        states = [states[i] for i in keep]
        prompts = [prompts[i] for i in keep]
        if not states:
            return
        gens = self.backend.generate(
            prompts,
            n=c.num_candidates,
            temperature=c.temperature,
            top_p=c.top_p,
            max_tokens=c.max_new_tokens,
        )
        # Flatten candidates for a single scoring pass.
        flat_q, flat_resp, flat_skip, owners, meta = [], [], [], [], []
        for si, (s, g) in enumerate(zip(states, gens, strict=True)):
            n_prefix = self.backend.count_tokens(s.prefix) if c.score_scope == "continuation" else 0
            for gen in g:
                flat_q.append(s.ex.question)
                flat_resp.append(s.prefix + gen.text)
                flat_skip.append(n_prefix)
                owners.append(si)
                meta.append((gen.finished, "sample", not gen.text.strip()))
            if c.include_expert_candidate and r == 0:
                flat_q.append(s.ex.question)
                flat_resp.append(s.ex.expert)
                flat_skip.append(0)
                owners.append(si)
                meta.append((True, "expert", False))
        scores = self.score(flat_q, flat_resp, flat_skip)
        per_state: list[list[Candidate]] = [[] for _ in states]
        for q_resp, (mlp, ntok), si, (fin, src, empty) in zip(
            flat_resp, scores, owners, meta, strict=True
        ):
            per_state[si].append(Candidate(q_resp, mlp, ntok, fin, src, empty))

        winners, selections = [], []
        for s, cands in zip(states, per_state, strict=True):
            sel = select_winner(cands, s.ex.expert, c.tie_epsilon, c.drop_truncated)
            winners.append(cands[sel.index])
            selections.append((sel, cands))

        verdicts = self.verifier.verify(
            [
                VerifyRequest(
                    s.ex.question,
                    s.ex.expert,
                    w.response,
                    s.ex.answer,
                    s.ex.answer_type,
                    s.unlocalized,
                )
                for s, w in zip(states, winners, strict=True)
            ]
        )
        for s, w, (sel, cands), v in zip(states, winners, selections, verdicts, strict=True):
            restart_token = self.backend.count_tokens(w.response[: v.restart_char])
            s.rounds.append(
                RoundLog(
                    round=r,
                    prefix_chars=len(s.prefix),
                    prefix_tokens=self.backend.count_tokens(s.prefix),
                    scores=[_finite(x.mean_logprob) for x in cands],
                    finished=[x.finished for x in cands],
                    winner=sel.index,
                    tied=sel.tied,
                    winner_source=w.source,
                    correct=v.correct,
                    reason=v.reason,
                    error_step=v.error_step,
                    num_steps=v.num_steps,
                    restart_char=v.restart_char,
                    restart_token=restart_token,
                )
            )
            s.last_winner = w
            if v.correct:
                s.done = s.accepted = True
                continue
            s.unlocalized = 0 if v.reason == "llm_error_step" else s.unlocalized + 1
            s.prefix = w.response[: v.restart_char]

    # -- public API -----------------------------------------------------------------------
    def boost(self, examples: list[Example], should_stop=None) -> list[BoostResult]:
        """Boost a batch. Raises BoostStopped (nothing emitted) if `should_stop()` turns
        true between rounds, so the caller can exit and a rerun redoes the whole batch."""
        t0 = time.time()
        states = [_State(ex) for ex in examples]
        for r in range(self.cfg.max_rounds):
            active = [s for s in states if not s.done]
            if not active:
                break
            if should_stop is not None and should_stop():
                raise BoostStopped
            self._round(active, r)
            log.info("round %d: %d/%d accepted", r, sum(s.accepted for s in states), len(states))
        elapsed = (time.time() - t0) / max(1, len(states))
        for s in states:
            if s.skipped:
                log.info("example %s skipped: %s", s.ex.id, s.skipped)
        final_resp = []
        for s in states:
            if s.accepted:
                final_resp.append((s.last_winner.response, "accepted"))
            elif self.cfg.on_failure == "expert":
                final_resp.append((s.ex.expert, "fallback_expert"))
            elif self.cfg.on_failure == "last" and s.last_winner is not None:
                final_resp.append((s.last_winner.response, "fallback_last"))
            else:
                final_resp.append((None, "dropped"))
        # Report final and expert likelihoods (useful to track the KL-style shift).
        qs = [s.ex.question for s in states]
        expert_scores = self.score(qs, [s.ex.expert for s in states])
        # Re-score emitted responses over all their tokens so the number is comparable to
        # the expert score whatever score_scope was used for selection.
        final_scores = self.score(qs, [resp or "" for resp, _ in final_resp])
        results = []
        for s, (resp, status), (elp, _), (full_lp, _) in zip(
            states, final_resp, expert_scores, final_scores, strict=True
        ):
            if resp is None:
                flp = None
            elif status == "fallback_expert":
                flp = elp
            else:
                flp = full_lp
            results.append(
                BoostResult(
                    id=s.ex.id,
                    question=s.ex.question,
                    response=resp,
                    status=status,
                    skipped=s.skipped,
                    answer=s.ex.answer,
                    answer_type=s.ex.answer_type,
                    expert=s.ex.expert,
                    num_rounds=len(s.rounds),
                    final_mean_logprob=_finite(flp),
                    expert_mean_logprob=_finite(elp),
                    rounds=[asdict(x) for x in s.rounds],
                    seconds=round(elapsed, 3),
                )
            )
        return results
