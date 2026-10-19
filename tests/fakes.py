"""Deterministic fake backend for exercising the boosting loop without a model."""

from __future__ import annotations

import re
from collections.abc import Callable
from types import SimpleNamespace

from vsft.backends.base import Backend, Generation

ASSISTANT = "\n\nAssistant: "


class FakeTokenizer:
    """Whitespace tokenizer; no chat template, so Backend.render uses the plain format."""

    chat_template = None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def parse_proposal(prompt: str) -> tuple[str, str]:
    """(question, prefix) from a rendered proposal prompt."""
    head, prefix = prompt.rsplit(ASSISTANT, 1)
    q = re.search(r"Question: (.*?)\n\nThis is an expert solution", head, re.S)
    if q is None:
        q = re.search(r"Task: (.*?)\n\nThis is an example", head, re.S)
    return q.group(1), prefix


def parse_verifier(prompt: str) -> tuple[str, list[str]]:
    q = re.search(r"Question: (.*?)\n\nExpert solution:", prompt, re.S).group(1)
    block = re.search(
        r"Candidate solution, split into numbered steps:\n(.*?)\n\nCheck every step", prompt, re.S
    ).group(1)
    steps = [re.sub(r"^\[\d+\] ", "", ln) for ln in block.split("\n")]
    return q, steps


def parse_student(prompt: str) -> str:
    head = prompt.rsplit(ASSISTANT, 1)[0]
    return head.split("\n\nUser: ", 1)[1]


class FakeBackend(Backend):
    """Scripted backend.

    propose(question, prefix, round_for_question) -> list[(text, finished)]
    judge(question, steps) -> raw verifier text
    score(question, response) -> float (every token gets this value) or a per-token list
    max_len: optional context length (Booster skips proposal prompts that do not fit)
    """

    def __init__(
        self,
        propose: Callable[[str, str, int], list[tuple[str, bool]]],
        judge: Callable[[str, list[str]], str],
        score: Callable[[str, str], float | list[float]] | None = None,
        max_len: int | None = None,
    ):
        self.tokenizer = FakeTokenizer()
        self.max_len = max_len
        self._propose = propose
        self._judge = judge
        self._score = score or (lambda q, r: -1.0)
        self.proposal_calls: list[list[str]] = []
        self.verifier_calls: list[list[str]] = []
        self.score_calls: list[tuple[list[str], list[str]]] = []
        self.gen_kwargs: list[dict] = []
        self._rounds: dict[str, int] = {}

    def generate(self, prompts, n, temperature, top_p, max_tokens):
        self.gen_kwargs.append(
            dict(n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        )
        is_ver = ["You are a strict grader" in p for p in prompts]
        assert all(is_ver) or not any(is_ver), "mixed verifier/proposal batch"
        out = []
        if all(is_ver) and prompts:
            self.verifier_calls.append(list(prompts))
            for p in prompts:
                assert n == 1
                q, steps = parse_verifier(p)
                out.append([Generation(self._judge(q, steps), True)])
            return out
        self.proposal_calls.append(list(prompts))
        for p in prompts:
            q, prefix = parse_proposal(p)
            r = self._rounds.get(q, 0)
            self._rounds[q] = r + 1
            gens = self._propose(q, prefix, r)
            assert len(gens) == n, f"script returned {len(gens)} gens, n={n}"
            out.append([Generation(t, f) for t, f in gens])
        return out

    def token_logprobs(self, prompts, responses):
        self.score_calls.append((list(prompts), list(responses)))
        res = []
        for p, r in zip(prompts, responses, strict=True):
            q = parse_student(p)
            v = self._score(q, r)
            if isinstance(v, list):
                # [] simulates a response truncated away entirely (e.g. by max_model_len)
                assert len(v) in (0, len(r.split())), "scripted logprobs must match tokens"
                res.append(list(v))
            else:
                res.append([v] * len(r.split()))
        return res


def boost_cfg(**kw):
    base = dict(
        num_candidates=2,
        max_rounds=3,
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=64,
        tie_epsilon=0.02,
        score_scope="full",
        drop_truncated=True,
        include_expert_candidate=False,
        on_failure="expert",
        proposal_prompt="generic",
        examples_per_batch=16,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def verifier_cfg(**kw):
    base = dict(
        max_new_tokens=32,
        temperature=0.0,
        use_answer_check=True,
        max_step_chars=400,
        accept_unparsed_if_answer_ok=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


ACCEPT = '{"correct": true}'


def reject_at(k: int) -> str:
    return f'{{"correct": false, "first_error_step": {k}}}'
