"""Dataset preparation into the project's JSONL schema.

Schema (one object per line):
    id, question, expert (expert trace), answer (reference final answer), answer_type
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

from vsft.answers import extract_boxed
from vsft.io import write_jsonl_atomic

log = logging.getLogger(__name__)

MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def _norm_problem(text: str) -> str:
    return " ".join(text.split()).lower()


def prepare_math(
    out_dir: str | Path, levels=(3, 4, 5), test_size: int = 1024, seed: int = 0
) -> dict[str, int]:
    """MATH levels 3-5 (paper: 9254 problems, 8230 train / 1024 test).

    Train and test splits of hendrycks_math are pooled, filtered by level, shuffled with a
    fixed seed and re-split so the test set has `test_size` problems.
    """
    from datasets import load_dataset

    keep = {f"Level {lv}" for lv in levels}
    rows = []
    for subject in MATH_SUBJECTS:
        for split in ("train", "test"):
            ds = load_dataset("EleutherAI/hendrycks_math", subject, split=split)
            for i, ex in enumerate(ds):
                if ex["level"] not in keep:
                    continue
                ans = extract_boxed(ex["solution"])
                if ans is None or not str(ans).strip():  # no answer (e.g. an empty \\boxed{})
                    continue
                rows.append(
                    {
                        "id": f"math-{subject}-{split}-{i}",
                        "question": ex["problem"],
                        "expert": ex["solution"],
                        "answer": ans,
                        "answer_type": "math",
                        "level": ex["level"],
                        "subject": subject,
                    }
                )
    # MATH-500 is a subset of the MATH test split; keep it out of training so it stays a
    # clean held-out benchmark.
    held_out = {
        _norm_problem(ex["problem"]) for ex in load_dataset("HuggingFaceH4/MATH-500", split="test")
    }
    n_before = len(rows)
    rows = [r for r in rows if _norm_problem(r["question"]) not in held_out]
    log.info("excluded %d MATH-500 problems from the pool", n_before - len(rows))
    rng = random.Random(seed)
    rng.shuffle(rows)
    if test_size >= len(rows):
        raise ValueError(f"test_size {test_size} >= number of problems {len(rows)}")
    test, train = rows[:test_size], rows[test_size:]
    out = Path(out_dir)
    write_jsonl_atomic(out / "math_l345_train.jsonl", train)
    write_jsonl_atomic(out / "math_l345_test.jsonl", test)
    log.info("MATH L%s: %d train / %d test -> %s", levels, len(train), len(test), out)
    return {"train": len(train), "test": len(test)}


def prepare_math500(out_dir: str | Path) -> int:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = [
        {
            "id": f"math500-{i}",
            "question": ex["problem"],
            "expert": ex["solution"],
            "answer": ex["answer"],
            "answer_type": "math",
        }
        for i, ex in enumerate(ds)
    ]
    write_jsonl_atomic(Path(out_dir) / "math500_test.jsonl", rows)
    return len(rows)


def prepare_gsm8k(out_dir: str | Path) -> int:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for i, ex in enumerate(ds):
        sol, _, ans = ex["answer"].rpartition("####")
        rows.append(
            {
                "id": f"gsm8k-{i}",
                "question": ex["question"],
                "expert": sol.strip(),
                "answer": ans.strip().replace(",", ""),
                "answer_type": "math",
            }
        )
    write_jsonl_atomic(Path(out_dir) / "gsm8k_test.jsonl", rows)
    return len(rows)


def prepare_medical(out_dir: str | Path, n_eval: int = 1000, seed: int = 0) -> dict[str, int]:
    """HuatuoGPT-o1 medical QA, as in the paper.

    Train: FreedomIntelligence/medical-o1-reasoning-SFT (en). The expert trace is the
    reasoning (Complex_CoT) followed by the final response; there is no short reference
    answer, so train rows carry none (the verifier works from the expert trace alone).
    Eval: `n_eval` questions sampled at random from medical-o1-verifiable-problem (the RL
    set), minus any question that is also in the training set; graded by an LLM judge
    against the short ground-truth answer (answer_type "judge").
    """
    from datasets import load_dataset

    train_ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")
    train, seen = [], set()
    for i, ex in enumerate(train_ds):
        q, cot, resp = ex["Question"], ex["Complex_CoT"], ex["Response"]
        if not (q and cot and resp):
            continue
        seen.add(_norm_problem(q))
        train.append(
            {
                "id": f"med-train-{i}",
                "question": q,
                "expert": f"{cot.strip()}\n\n{resp.strip()}",
                "answer": None,
                "answer_type": "judge",
            }
        )
    pool_ds = load_dataset("FreedomIntelligence/medical-o1-verifiable-problem", split="train")
    pool = []
    for i, ex in enumerate(pool_ds):
        q, a = ex["Open-ended Verifiable Question"], ex["Ground-True Answer"]
        if q and a and str(a).strip() and _norm_problem(q) not in seen:
            pool.append(
                {
                    "id": f"med-eval-{i}",
                    "question": q,
                    "expert": str(a),
                    "answer": str(a),
                    "answer_type": "judge",
                }
            )
    if n_eval > len(pool):
        raise ValueError(f"n_eval {n_eval} > {len(pool)} eval candidates")
    test = random.Random(seed).sample(pool, n_eval)
    out = Path(out_dir)
    write_jsonl_atomic(out / "medical_train.jsonl", train)
    write_jsonl_atomic(out / "medical_test.jsonl", test)
    log.info("medical: %d train / %d test -> %s", len(train), len(test), out)
    return {"train": len(train), "test": len(test)}


def prepare_mmlu(out_dir: str | Path, n: int = 1000, seed: int = 0) -> int:
    """MMLU (cais/mmlu, all subjects) as a retention benchmark: `n` random test questions,
    multiple choice with the letter as the reference answer."""
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    letters = "ABCD"
    rows = [
        {
            "id": f"mmlu-{i}",
            "question": ex["question"].strip()
            + "\n\n"
            + "\n".join(f"{letters[j]}. {c}" for j, c in enumerate(ex["choices"]))
            + "\n\nAnswer with the letter of the correct option.",
            "expert": "",
            "answer": letters[ex["answer"]],
            "answer_type": "mc",
            "subject": ex["subject"],
        }
        for i, ex in enumerate(ds)
    ]
    rows = random.Random(seed).sample(rows, min(n, len(rows)))
    write_jsonl_atomic(Path(out_dir) / "mmlu_test.jsonl", rows)
    return len(rows)


PREPARERS = {
    "math": prepare_math,
    "math500": prepare_math500,
    "medical": prepare_medical,
    "mmlu": prepare_mmlu,
    "gsm8k": prepare_gsm8k,
}
