"""Final-answer extraction and comparison (math / multiple choice / short text)."""

from __future__ import annotations

import re


def extract_boxed(text: str) -> str | None:
    """Return the content of the last \\boxed{...} (or \\fbox{...}), handling nested braces."""
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    m = re.match(r"\\(?:boxed|fbox)\s*\{", text[idx:])
    if m is None:
        # "\boxed 5" form: the answer is the next whitespace-delimited token
        m = re.match(r"\\boxed\s+([^\s$]+)", text[idx:])
        return m.group(1).rstrip(".,;") if m else None
    i = idx + m.end() - 1
    depth, start = 0, i + 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start:j]
    return None


# Multiple choice: an explicit "answer" cue followed by an uppercase letter, or a line that
# is only "(B)" / "B." / "B)". Letters are case-sensitive so "I think" or "a ..." never match.
_MC_CUE = re.compile(r"(?i:answer)\s*(?:(?i:is))?\s*[:：]?\s*\(?([A-J])\b(?![a-z])")
# Lowercase letters only count right after a cue and at the end of a clause ("answer: b").
_MC_CUE_LOWER = re.compile(
    r"(?i:answer)\s*(?:(?i:is))?\s*[:：]?\s*\(?([a-j])\)?\s*(?:[.)]|$)", re.M
)
_MC_LINE = re.compile(r"^\s*\(?([A-J])[).:]?\s*$", re.MULTILINE)


def extract_answer(text: str, answer_type: str = "math") -> str | None:
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    if answer_type == "mc":
        hits = _MC_CUE.findall(text) or _MC_CUE_LOWER.findall(text) or _MC_LINE.findall(text)
        return hits[-1].upper() if hits else None
    m = re.findall(r"(?:final answer|answer)\s*(?:is)?\s*[:：]\s*(.+)", text, re.I)
    return m[-1].strip() if m else None


def _strip_latex(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    s = re.sub(r"\\(?:mathrm|textbf|mathbf)\{(.*?)\}", r"\1", s)
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("\\,", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac").replace("^\\circ", "")
    s = s.replace("^{\\circ}", "").replace("\\%", "").replace("%", "").replace("\\$", "")
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    # \frac12 -> \frac{1}{2}
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    return s.lower()


def _as_number(s: str) -> float | None:
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", s)
    if m:
        den = float(m.group(3))
        return None if den == 0 else (-1 if m.group(1) else 1) * float(m.group(2)) / den
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)", s)
    if m:
        den = float(m.group(2))
        return None if den == 0 else float(m.group(1)) / den
    try:
        return float(s)
    except ValueError:
        return None


def normalize_answer(s, answer_type: str = "math") -> str | None:
    if s is None:
        return None
    s = str(s)
    if answer_type == "mc":
        s = re.sub(r"\\(?:text|textbf|mathrm|mathbf)\{(.*?)\}", r"\1", s)
        m = re.search(r"\b([A-J])\b", s.upper())
        return m.group(1) if m else None
    if answer_type == "text":
        return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    return _strip_latex(s)


def answers_match(pred: str | None, ref: str | None, answer_type: str = "math") -> bool:
    p, r = normalize_answer(pred, answer_type), normalize_answer(ref, answer_type)
    if p is None or r is None:
        return False
    if p == r:
        return True
    if answer_type == "math":
        pn, rn = _as_number(p), _as_number(r)
        if pn is not None and rn is not None:
            return abs(pn - rn) <= 1e-6 * max(1.0, abs(rn))
    return False
