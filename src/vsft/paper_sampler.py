"""The authors' method: block-wise MCMC sampling from the expert-conditioned target
(Karan, Chen & Du, arXiv:2610.02140, Algorithm 1), run as a batched sampler.

Per example, the response is built one block of B tokens at a time. Each block:

1. The proposal kappa (the student conditioned on the question, the expert trace and the
   kept prefix) samples a continuation of up to B tokens.
2. N_MCMC steps refine it. A cut point m is drawn uniformly inside the block, the tail
   from m is resampled from the proposal, and the candidate is kept whenever its student
   log-likelihood is at least as high. This is the paper's greedy acceptance from
   Appendix C.2: it avoids the second forward pass that exact MH needs.
3. The block is appended to the prefix. Generation stops at EOS or at T tokens.

Calls are batched across examples: each step makes one generate and one scoring call.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import asdict, dataclass

from vsft.backends.base import Backend
from vsft.boost import Example
from vsft.prompts import proposal_messages, student_messages

log = logging.getLogger(__name__)
NEG_INF = float("-inf")


@dataclass
class SampledResult:
    id: str
    question: str
    response: str
    status: str  # paper_sampled (EOS reached) | paper_truncated (hit T)
    answer: str | None
    answer_type: str
    expert: str
    num_blocks: int
    mcmc_accepted: int
    mcmc_steps: int
    final_mean_logprob: float | None
    expert_mean_logprob: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Item:
    ex: Example
    ids: list[int]  # kept response tokens (complete blocks)
    done: bool = False
    finished: bool = False  # EOS emitted
    cand: list[int] | None = None  # current block candidate: ids + block tokens
    cand_fin: bool = False
    block_budget: int = 0  # tokens this block may add (B, or the room left before T)
    cand_lp: float = NEG_INF
    accepted: int = 0
    steps: int = 0


class PaperSampler:
    def __init__(self, backend: Backend, cfg):
        self.backend = backend
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

    # -- token helpers -----------------------------------------------------------------
    def _text(self, ids: list[int]) -> str:
        return self.backend.tokenizer.decode(ids, skip_special_tokens=True)

    def _ids(self, text: str) -> list[int]:
        return self.backend.tokenizer(text, add_special_tokens=False)["input_ids"]

    # -- proposal ----------------------------------------------------------------------
    def _propose(self, items: list[tuple[Example, list[int]]], budgets: list[int]):
        """Sample continuations of `budgets[i]` tokens after prefix i. Returns (ids, eos)."""
        if not items:
            return []
        prompts = [
            self.backend.render(
                proposal_messages(ex.question, ex.expert, self.cfg.proposal_prompt),
                self._text(prefix),
            )
            for ex, prefix in items
        ]
        gens = self.backend.generate(
            prompts,
            n=1,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=max(budgets),
        )
        out = []
        for g, budget in zip(gens, budgets, strict=True):
            full = self._ids(g[0].text)
            if len(full) > budget:
                out.append((full[:budget], False))  # cut at the block boundary
            else:
                out.append((full, bool(g[0].finished)))
        return out

    # -- scoring: log p(response | question) under the student ---------------------------
    def _loglik(self, items: list[tuple[Example, list[int]]]) -> list[float]:
        """Total student log-likelihood of each response (-inf when empty)."""
        live = [i for i, (_, ids) in enumerate(items) if ids]
        out = [NEG_INF] * len(items)
        if not live:
            return out
        prompts = [self.backend.render(student_messages(items[i][0].question)) for i in live]
        lps = self.backend.token_logprobs(prompts, [self._text(items[i][1]) for i in live])
        for i, lp in zip(live, lps, strict=True):
            out[i] = float(sum(lp))
        return out

    # -- the algorithm -----------------------------------------------------------------
    def run(self, examples: list[Example]) -> list[SampledResult]:
        cfg = self.cfg
        B, T = cfg.block_size, cfg.max_new_tokens
        t0 = time.time()
        items = [_Item(ex, []) for ex in examples]
        n_blocks = math.ceil(T / B)
        for k in range(n_blocks):
            active = [it for it in items if not it.done]
            if not active:
                break
            # 1. proposal for the whole block
            base = [(it.ex, it.ids) for it in active]
            # The last block is only as long as the room left before T; proposing more would
            # be discarded and would make the acceptance test score tokens that are not kept.
            budgets = [min(B, T - len(it.ids)) for it in active]
            block = self._propose(base, budgets)
            for it, budget, (ids, fin) in zip(active, budgets, block, strict=True):
                it.block_budget = budget
                it.cand = it.ids + ids
                it.cand_fin = fin
            cand_lp = self._loglik([(it.ex, it.cand) for it in active])
            for it, lp in zip(active, cand_lp, strict=True):
                it.cand_lp = lp
            # 2. N_MCMC refinements of the block tail
            for _ in range(cfg.n_mcmc):
                movable = [it for it in active if len(it.cand) > len(it.ids)]
                if not movable:
                    break
                cuts = {}
                for it in movable:
                    cuts[id(it)] = self.rng.randrange(len(it.ids), len(it.cand))
                kept = [(it.ex, it.cand[: cuts[id(it)]]) for it in movable]
                budgets = [it.block_budget - (cuts[id(it)] - len(it.ids)) for it in movable]
                tails = self._propose(kept, budgets)
                new_cands = [
                    (k_ids + tail, fin) for (_, k_ids), (tail, fin) in zip(kept, tails, strict=True)
                ]
                new_lp = self._loglik(
                    [(it.ex, c) for it, (c, _) in zip(movable, new_cands, strict=True)]
                )
                for it, (c, fin), lp in zip(movable, new_cands, new_lp, strict=True):
                    it.steps += 1
                    # greedy acceptance (paper Appendix C.2): keep the higher-likelihood one
                    if lp >= it.cand_lp:
                        it.cand, it.cand_fin, it.cand_lp = c, fin, lp
                        it.accepted += 1
            # 3. commit the block
            for it in active:
                it.ids = it.cand
                if len(it.ids) > T:  # cut by the length limit: not finished, whatever the EOS
                    it.ids = it.ids[:T]
                    it.finished = False
                else:
                    it.finished = it.cand_fin
                it.done = it.finished or len(it.ids) >= T
            log.info("paper block %d/%d: %d active", k + 1, n_blocks, len(active))
        results = []
        exp_lp = self._loglik([(it.ex, self._ids(it.ex.expert)) for it in items])
        final_lp = self._loglik([(it.ex, it.ids) for it in items])
        for it, flp, elp in zip(items, final_lp, exp_lp, strict=True):
            n_tok = max(1, len(it.ids))
            results.append(
                SampledResult(
                    id=it.ex.id,
                    question=it.ex.question,
                    response=self._text(it.ids),
                    status="paper_sampled" if it.finished else "paper_truncated",
                    answer=it.ex.answer,
                    answer_type=it.ex.answer_type,
                    expert=it.ex.expert,
                    num_blocks=math.ceil(len(it.ids) / self.cfg.block_size),
                    mcmc_accepted=it.accepted,
                    mcmc_steps=it.steps,
                    final_mean_logprob=None if flp == NEG_INF else flp / n_tok,
                    expert_mean_logprob=None
                    if elp == NEG_INF
                    else elp / max(1, len(self._ids(it.ex.expert))),
                )
            )
        log.info("paper sampler: %d examples in %.0fs", len(items), time.time() - t0)
        return results
