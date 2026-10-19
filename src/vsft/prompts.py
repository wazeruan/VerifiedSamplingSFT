"""Prompt templates and step segmentation.

Three prompts are involved:

* student prompt  - the plain task prompt (question only). Candidates are *scored* under
  this prompt, i.e. we measure log p(response | question) under the student/base model,
  exactly as the paper's target likelihood p(x).
* proposal prompt - question + expert trace + partial response; the student continues
  "in its own words" (the paper's information-preserving proposal kappa_C).
* verifier prompt - question + expert trace (+ reference answer) + the candidate split into
  numbered steps; the model answers which step is the first incorrect one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

STUDENT_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."

PROPOSAL_PROMPTS: dict[str, str] = {
    "generic": (
        "You are given a question, an expert solution, and a partial response. Examine the "
        "expert solution and identify all the information in its reasoning.\n\n"
        "Question: {question}\n\n"
        "This is an expert solution to the query:\n{expert}\n\n"
        "Starting with the partial response, continue in your own words, including the "
        "thinking process. Your response must remain logically consistent with the expert "
        "solution and reach a complete, correct final answer within \\boxed{{}}. Output only "
        "the response itself."
    ),
    "chemistry": (
        "You are an expert chemist given a chemistry problem, its solution, and an initial, "
        "partial response. Study which steps the solution already provides and continue the "
        "partial response consistently.\n\n"
        "Task: {question}\n\n"
        "This is an example for a solution to the problem:\n{expert}\n\n"
        "Continue the partial response in your own words, including the thinking process. "
        "Ensure the final answer exactly matches that of the provided solution and is "
        "written within \\boxed{{}}. Output only the response itself."
    ),
}

VERIFIER_PROMPT = (
    "You are a strict grader. Compare a candidate solution with a trusted expert solution.\n\n"
    "Question: {question}\n\n"
    "Expert solution:\n{expert}\n\n"
    "{reference_line}"
    "Candidate solution, split into numbered steps:\n{steps}\n\n"
    "Check every step of the candidate against the expert solution. A step is wrong if it "
    "contains a calculation error, a false claim, contradicts the expert solution, or "
    "reaches a different final answer. Rephrasing or extra correct detail is fine.\n"
    "Reply with a single JSON object and nothing else:\n"
    '{{"correct": true}} if every step is correct, or\n'
    '{{"correct": false, "first_error_step": <number of the first wrong step>}}'
)


def student_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": STUDENT_SYSTEM},
        {"role": "user", "content": question},
    ]


def proposal_messages(question: str, expert: str, template: str = "generic") -> list[dict]:
    if template not in PROPOSAL_PROMPTS:
        raise KeyError(f"unknown proposal prompt {template!r}; have {sorted(PROPOSAL_PROMPTS)}")
    body = PROPOSAL_PROMPTS[template].format(question=question, expert=expert)
    return [{"role": "user", "content": body}]


def verifier_messages(
    question: str, expert: str, steps: list[str], reference_answer: str | None
) -> list[dict[str, str]]:
    numbered = "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(steps))
    ref = f"Reference final answer: {reference_answer}\n\n" if reference_answer else ""
    body = VERIFIER_PROMPT.format(
        question=question, expert=expert, reference_line=ref, steps=numbered
    )
    return [{"role": "user", "content": body}]


@dataclass(frozen=True)
class Step:
    text: str
    start: int  # character offset of the step in the response
    end: int


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\$(])")


def segment_steps(response: str, max_step_chars: int = 400) -> list[Step]:
    """Split a response into steps with exact character offsets.

    Steps are non-empty lines; lines longer than `max_step_chars` are further split at
    sentence boundaries. Offsets always index into `response`, so `response[:step.start]`
    is the prefix to keep when regeneration restarts at that step.
    """
    steps: list[Step] = []
    for m in re.finditer(r"[^\n]+", response):
        line, base = m.group(0), m.start()
        if not line.strip():
            continue
        pieces = [(0, len(line))]
        if len(line) > max_step_chars:
            pieces, cur = [], 0
            for sm in _SENT_SPLIT.finditer(line):
                if sm.start() - cur >= 20:
                    pieces.append((cur, sm.start()))
                    cur = sm.end()
            pieces.append((cur, len(line)))
        for a, b in pieces:
            text = line[a:b]
            lead = len(text) - len(text.lstrip())
            text = text.strip()
            if text:
                steps.append(
                    Step(text=text, start=base + a + lead, end=base + a + lead + len(text))
                )
    return steps
