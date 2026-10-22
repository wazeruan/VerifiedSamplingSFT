"""LLM judge for open-ended QA (answer_type "judge").

The paper grades medical answers with GPT-5-mini: the judge sees the question, a reference
answer and the model's response, and says whether the response contains the reference's
key information. This module does the same with any local backend.
"""

from __future__ import annotations

import re

from vsft.backends.base import Backend

JUDGE_PROMPT = (
    "You are an expert evaluator. Compare the model response with the reference answer.\n\n"
    "Question:\n{question}\n\n"
    "Reference answer:\n{reference}\n\n"
    "Model response:\n{response}\n\n"
    "The response is CORRECT if it contains the key information of the reference answer, "
    "even if it is phrased differently or adds correct detail. It is INCORRECT if it "
    "contradicts the reference, misses the main point, or gives wrong information. Judge "
    "accuracy and completeness, not style.\n"
    "Output ONLY one word: CORRECT or INCORRECT."
)

_VERDICT = re.compile(r"\b(INCORRECT|CORRECT)\b")


def parse_verdict(text: str) -> bool | None:
    """True for CORRECT, False for INCORRECT, None if the reply names neither.

    The last verdict word wins, and "INCORRECT" is matched before "CORRECT" so it is never
    mistaken for it.
    """
    hits = _VERDICT.findall(text.upper())
    if not hits:
        return None
    return hits[-1] == "CORRECT"


def judge(
    backend: Backend,
    items: list[tuple[str, str, str]],
    max_tokens: int = 16,
    max_response_chars: int = 12000,
) -> list[bool | None]:
    """Grade (question, reference, response) triples with one batched greedy call."""
    if not items:
        return []
    prompts = [
        backend.render(
            [
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        question=q, reference=ref, response=resp[:max_response_chars]
                    ),
                }
            ]
        )
        for q, ref, resp in items
    ]
    gens = backend.generate(prompts, n=1, temperature=0.0, top_p=1.0, max_tokens=max_tokens)
    return [parse_verdict(g[0].text) for g in gens]
