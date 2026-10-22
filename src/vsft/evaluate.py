"""Single-shot accuracy evaluation (greedy by default), as in the paper."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from vsft.answers import answers_match, extract_answer
from vsft.backends.base import Backend
from vsft.io import read_jsonl, write_json_atomic, write_jsonl_atomic
from vsft.judge import judge
from vsft.prompts import student_messages

log = logging.getLogger(__name__)


def evaluate(
    backend: Backend,
    data_path: str | Path,
    out_dir: str | Path,
    eval_cfg,
    limit: int = 0,
    judge_backend: Backend | None = None,
) -> dict:
    rows = list(read_jsonl(data_path))
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"no rows in {data_path}")

    # Rows without a reference answer cannot be graded. Skip them, and record the count, so
    # every method evaluated on this file uses the same rows (the summary's n shows it).
    def unanswerable(r) -> bool:
        return r.get("answer") is None or str(r["answer"]).strip() == ""

    skipped = [r.get("id") for r in rows if unanswerable(r)]
    if skipped:
        log.warning(
            "skipping %d rows without a reference answer, e.g. %s", len(skipped), skipped[:3]
        )
        rows = [r for r in rows if not unanswerable(r)]
    if not rows:
        raise ValueError(f"no gradable rows in {data_path}")
    preds = []
    bs = eval_cfg.batch_size
    for i in range(0, len(rows), bs):
        chunk = rows[i : i + bs]
        prompts = [backend.render(student_messages(r["question"])) for r in chunk]
        gens = backend.generate(
            prompts,
            n=1,
            temperature=eval_cfg.temperature,
            top_p=1.0,
            max_tokens=eval_cfg.max_new_tokens,
        )
        # Open-ended QA rows are graded by the LLM judge, the rest by answer matching.
        judged = [i for i, r in enumerate(chunk) if r.get("answer_type") == "judge"]
        verdicts: dict[int, bool | None] = {}
        if judged:
            if judge_backend is None:
                raise ValueError(
                    f"{Path(data_path).name} has open-ended rows (answer_type 'judge') but no "
                    "judge backend was given (set eval.judge_model)"
                )
            res = judge(
                judge_backend,
                [(chunk[i]["question"], str(chunk[i]["answer"]), gens[i][0].text) for i in judged],
                max_tokens=getattr(eval_cfg, "judge_max_new_tokens", 16),
            )
            verdicts = dict(zip(judged, res, strict=True))
        for i, (r, g) in enumerate(zip(chunk, gens, strict=True)):
            atype = r.get("answer_type", "math")
            if atype == "judge":
                pred = g[0].text.strip()[:300] or None  # shown for inspection only
                correct = verdicts.get(i) is True
            else:
                pred = extract_answer(g[0].text, atype)
                correct = answers_match(pred, str(r["answer"]), atype)
            preds.append(
                {
                    "id": r["id"],
                    "prediction": pred,
                    "answer": str(r["answer"]),
                    "correct": correct,
                    **({"judge_verdict": verdicts.get(i)} if atype == "judge" else {}),
                    "finished": g[0].finished,
                    "response": g[0].text,
                    **{k: r[k] for k in ("level", "subject") if k in r},
                }
            )
        log.info("eval %s: %d/%d", Path(data_path).name, len(preds), len(rows))
    acc = sum(p["correct"] for p in preds) / len(preds)
    by_level: dict[str, list[bool]] = defaultdict(list)
    for p in preds:
        if "level" in p:
            by_level[p["level"]].append(p["correct"])
    metrics = {
        "data": str(data_path),
        "num_examples": len(preds),
        "accuracy": acc,
        "num_unfinished": sum(not p["finished"] for p in preds),
        "num_no_answer": sum(p["prediction"] is None for p in preds),
        "num_judge_unparsed": sum(
            1 for p in preds if "judge_verdict" in p and p["judge_verdict"] is None
        ),
        "accuracy_by_level": {k: sum(v) / len(v) for k, v in sorted(by_level.items())},
    }
    out = Path(out_dir)
    stem = Path(data_path).stem
    write_jsonl_atomic(out / f"{stem}.predictions.jsonl", preds)
    write_json_atomic(out / f"{stem}.metrics.json", metrics)
    return metrics
