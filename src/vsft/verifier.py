"""Verifier: the student model, prompted with the expert trace, grades the winning
candidate and localizes the first wrong step. The step's character offset (and token
index) is where regeneration restarts.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field

from vsft.answers import answers_match, extract_answer
from vsft.backends.base import Backend
from vsft.prompts import segment_steps, verifier_messages


@dataclass
class VerifyRequest:
    question: str
    expert: str
    response: str
    reference_answer: str | None = None
    answer_type: str = "math"
    unlocalized_failures: int = 0  # consecutive past failures without a usable step


@dataclass
class Verdict:
    correct: bool
    restart_char: int  # keep response[:restart_char] and regenerate after it
    error_step: int | None  # 0-based index into `steps` (None if correct/unknown)
    num_steps: int
    reason: str  # accepted | llm_error_step | answer_mismatch | unparsed | empty
    llm_correct: bool | None = None
    answer_ok: bool | None = None
    raw: str = ""
    step_starts: list[int] = field(default_factory=list)


_BOOL = re.compile(r'"?correct"?\s*:\s*(true|false)', re.I)
_STEP = re.compile(r'"?first_error_step"?\s*:\s*"?\[?(\d+)', re.I)


def _as_bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in {"true", "false"}:
        return v.strip().lower() == "true"
    return None


def parse_verifier_output(text: str) -> tuple[bool | None, int | None]:
    """Return (correct, 1-based first_error_step) from the verifier's reply."""
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text))):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            try:  # Python-dict style reply: {'correct': False, 'first_error_step': 2}
                obj = ast.literal_eval(m.group(0))
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                continue
        if isinstance(obj, dict) and "correct" in obj:
            corr = _as_bool(obj.get("correct"))
            step = obj.get("first_error_step")
            try:
                step = int(step) if step is not None and not isinstance(step, bool) else None
            except (TypeError, ValueError):
                step = None
            if corr is None and step is not None:
                corr = False
            return corr, step
    b, s = _BOOL.search(text), _STEP.search(text)
    corr = None if b is None else b.group(1).lower() == "true"
    step = int(s.group(1)) if s else None
    if corr is None and step is not None:
        corr = False
    return corr, step


def restart_offset(steps, k: int) -> int:
    """Character offset to cut at when step k (0-based) is wrong.

    Cutting at the end of step k-1 (not the start of step k) keeps the whitespace or
    newline separator out of the prefix, so the model generates it itself and the token
    boundary stays in-distribution.
    """
    return steps[k - 1].end if k > 0 else 0


def decide(
    req: VerifyRequest,
    raw: str,
    max_step_chars: int,
    use_answer_check: bool,
    accept_unparsed_if_answer_ok: bool = False,
    prompt_too_long: bool = False,
) -> Verdict:
    steps = segment_steps(req.response, max_step_chars)
    starts = [s.start for s in steps]
    if not steps:
        return Verdict(False, 0, None, 0, "empty", raw=raw)
    llm_correct, step1 = parse_verifier_output(raw)
    if prompt_too_long:  # the verifier never saw the candidate: never accept, back off
        llm_correct, step1 = None, None
    answer_ok = None
    if use_answer_check and req.reference_answer is not None:
        answer_ok = answers_match(
            extract_answer(req.response, req.answer_type), req.reference_answer, req.answer_type
        )
    common = dict(
        num_steps=len(steps),
        llm_correct=llm_correct,
        answer_ok=answer_ok,
        raw=raw,
        step_starts=starts,
    )
    if llm_correct is True and answer_ok is not False:
        return Verdict(True, len(req.response), None, reason="accepted", **common)
    if (
        not prompt_too_long
        and llm_correct is None
        and answer_ok is True
        and accept_unparsed_if_answer_ok
    ):
        return Verdict(True, len(req.response), None, reason="accepted_answer_only", **common)
    if llm_correct is False and step1 is not None and 1 <= step1 <= len(steps):
        k = step1 - 1
        return Verdict(False, restart_offset(steps, k), k, reason="llm_error_step", **common)
    # No usable localization: back off exponentially from the end of the response.
    k = max(0, len(steps) - 2 ** (req.unlocalized_failures + 1))
    if prompt_too_long:
        reason = "verifier_too_long"
    else:
        reason = "answer_mismatch" if answer_ok is False else "unparsed"
    return Verdict(False, restart_offset(steps, k), k, reason=reason, **common)


class Verifier:
    def __init__(self, backend: Backend, cfg):
        self.backend = backend
        self.cfg = cfg

    def verify(self, requests: list[VerifyRequest]) -> list[Verdict]:
        if not requests:
            return []
        prompts = []
        for r in requests:
            steps = [s.text for s in segment_steps(r.response, self.cfg.max_step_chars)]
            msgs = verifier_messages(r.question, r.expert, steps, r.reference_answer)
            prompts.append(self.backend.render(msgs))
        gens = self.backend.generate(
            prompts,
            n=1,
            temperature=self.cfg.temperature,
            top_p=1.0,
            max_tokens=self.cfg.max_new_tokens,
        )
        max_len = getattr(self.backend, "max_len", None)
        too_long = [
            max_len is not None and self.backend.count_tokens(p) >= max_len for p in prompts
        ]
        return [
            decide(
                r,
                g[0].text,
                self.cfg.max_step_chars,
                self.cfg.use_answer_check,
                self.cfg.accept_unparsed_if_answer_ok,
                prompt_too_long=tl,
            )
            for r, g, tl in zip(requests, gens, too_long, strict=True)
        ]
