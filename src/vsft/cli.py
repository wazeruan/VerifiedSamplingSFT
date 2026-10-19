"""Command-line entry point: `vsft <command> ...`."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

from vsft.config import load_config


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, help="YAML config file")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override a config value (repeatable), e.g. --set boost.num_candidates=8",
    )


def cmd_boost(args) -> int:
    from vsft.backends.base import build_backend
    from vsft.boost import Booster, Example
    from vsft.io import append_jsonl, completed_ids, read_jsonl, repair_tail, write_json_atomic

    cfg = load_config(args.config, args.overrides)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    examples = [Example.from_dict(d) for d in read_jsonl(args.input)]
    if args.limit:
        examples = examples[: args.limit]
    ids = [e.id for e in examples]
    if len(set(ids)) != len(ids):
        raise SystemExit("input has duplicate ids")
    repair_tail(out)
    done = completed_ids(out)
    todo = [e for e in examples if e.id not in done]
    logging.info(
        "boost: %d examples, %d already done, %d to go", len(examples), len(done), len(todo)
    )
    if todo:
        booster = Booster(build_backend(cfg.model, cfg.backend), cfg.boost, cfg.verifier)
        bs = cfg.boost.examples_per_batch
        for i in range(0, len(todo), bs):
            results = booster.boost(todo[i : i + bs])
            append_jsonl(out, (r.to_dict() for r in results))
            logging.info("boost: wrote %d/%d", min(i + bs, len(todo)), len(todo))
    # Summary over the whole output file.
    out.touch(exist_ok=True)
    wanted = set(ids)  # summarize only rows belonging to this input (and --limit)
    rows = [r for r in read_jsonl(out) if str(r.get("id")) in wanted]
    status = Counter(r["status"] for r in rows)
    rounds = [r["num_rounds"] for r in rows]
    gains = [
        r["final_mean_logprob"] - r["expert_mean_logprob"]
        for r in rows
        if r["status"] == "accepted"
        and r["final_mean_logprob"] is not None
        and r["expert_mean_logprob"] is not None
    ]
    summary = {
        "input": str(args.input),
        "output": str(out),
        "num_rows": len(rows),
        "status": dict(status),
        "accept_rate": status.get("accepted", 0) / max(1, len(rows)),
        "mean_rounds": sum(rounds) / max(1, len(rounds)),
        "mean_logprob_gain_vs_expert": sum(gains) / len(gains) if gains else None,
        "config": cfg.to_dict(),
    }
    write_json_atomic(out.with_suffix(".summary.json"), summary)
    logging.info("boost summary: %s", {k: v for k, v in summary.items() if k != "config"})
    return 0


def cmd_prepare(args) -> int:
    from vsft.data import PREPARERS

    for name in args.datasets:
        if name not in PREPARERS:
            raise SystemExit(f"unknown dataset {name!r}; choose from {sorted(PREPARERS)}")
    for name in args.datasets:
        logging.info("prepare %s: %s", name, PREPARERS[name](args.out_dir))
    return 0


def cmd_train(args) -> int:
    from vsft.train import train

    cfg = load_config(args.config, args.overrides)
    if args.resume_from and not (Path(args.resume_from) / "trainer_state.json").exists():
        raise SystemExit(f"--resume-from {args.resume_from} has no trainer_state.json")
    final = train(cfg, args.data, args.output_dir, args.resume_from)
    logging.info("train: final model at %s", final)
    return 0


def cmd_eval(args) -> int:
    from vsft.backends.base import build_backend
    from vsft.evaluate import evaluate

    cfg = load_config(args.config, args.overrides)
    if args.model:
        cfg.model.name_or_path = str(args.model)
    backend = build_backend(cfg.model, cfg.backend)
    results = {}
    for data in args.data:
        m = evaluate(backend, data, args.output_dir, cfg.eval, args.limit)
        results[Path(data).stem] = m["accuracy"]
        logging.info("eval %s: accuracy %.4f (n=%d)", data, m["accuracy"], m["num_examples"])
    from vsft.io import write_json_atomic

    write_json_atomic(
        Path(args.output_dir) / "eval_summary.json",
        {"model": cfg.model.name_or_path, "accuracy": results},
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vsft", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("boost", help="rewrite expert traces with the verifier-guided loop")
    _common(b)
    b.add_argument(
        "--input", required=True, type=Path, help="JSONL with id/question/expert[/answer]"
    )
    b.add_argument("--output", required=True, type=Path, help="output JSONL (resumes if present)")
    b.add_argument("--limit", type=int, default=0, help="only the first N examples (0 = all)")
    b.set_defaults(func=cmd_boost)

    pr = sub.add_parser("prepare", help="download and convert datasets to JSONL")
    pr.add_argument("datasets", nargs="+", help="math | math500 | gsm8k")
    pr.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    pr.set_defaults(func=cmd_prepare)

    t = sub.add_parser("train", help="SFT on boosted (or raw expert) traces")
    _common(t)
    t.add_argument("--data", required=True, type=Path, help="boosted JSONL (or raw JSONL)")
    t.add_argument("--output-dir", required=True, type=Path)
    t.add_argument("--resume-from", type=Path, help="state-* directory written by a prior run")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", help="single-shot accuracy on one or more JSONL test sets")
    _common(e)
    e.add_argument("--model", help="model path/name (overrides model.name_or_path)")
    e.add_argument("--data", required=True, nargs="+", type=Path)
    e.add_argument("--output-dir", required=True, type=Path)
    e.add_argument("--limit", type=int, default=0)
    e.set_defaults(func=cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
