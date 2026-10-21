"""Comparison table: initial model vs plain SFT (expert traces) vs Sampling SFT.

Stdlib only and Python >= 3.6 compatible, so it also runs on a login node without the
project venv (e.g. EL8 system python3):

    python3 src/vsft/compare.py RUN_DIR/eval            # or: vsft compare RUN_DIR/eval
    python3 src/vsft/compare.py a/eval_summary.json b/  # any mix of files and directories

Each `eval_summary.json` carries a `label` (base | sft | sampling, written by `vsft eval
--label`); older summaries get their label from the directory name `<label>-<jobid>`.
When a label appears more than once, the newest summary wins.

Rows are only comparable when every method was evaluated on the same files (same sha256
and example count) with the same decoding settings; mismatches are reported, and the
command exits 2 unless --allow-mismatch is given.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

ORDER = ["base", "sft", "sampling", "paper"]
NAMES = {
    "base": "Initial model",
    "sft": "Plain SFT (expert traces)",
    "sampling": "Sampling SFT (verifier-guided)",
    "paper": "Authors' method (block MCMC SFT)",
}


def _label_of(path, summary):
    if summary.get("label"):
        return str(summary["label"])
    name = path.parent.name
    m = re.match(r"(.+?)(?:-final)?-\d+$", name)  # "<label>-<jobid>" or legacy "<x>-final-<jobid>"
    return m.group(1) if m else name


def collect(paths):
    """label -> summary dict (newest per label), with "label" and "path" added."""
    files = []
    for p in map(Path, paths):
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.rglob("eval_summary.json")))
    found = {}
    for f in files:
        try:
            s = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(s, dict) or not isinstance(s.get("accuracy"), dict):
            continue
        label = _label_of(f, s)
        mtime = os.path.getmtime(str(f))
        if label not in found or mtime >= found[label][0]:
            s = dict(s)
            s.update(label=label, path=str(f))
            found[label] = (mtime, s)
    return {k: v for k, (_, v) in found.items()}


def _ordered(results):
    return [x for x in ORDER if x in results] + sorted(set(results) - set(ORDER))


def _datasets(results):
    out = []
    for lab in _ordered(results):
        for d in results[lab]["accuracy"]:
            if d not in out:
                out.append(d)
    return out


def check_comparable(results):
    """Human-readable problems that make rows not comparable (empty list = comparable)."""
    problems = []
    labels = _ordered(results)
    for d in _datasets(results):
        missing = [lab for lab in labels if d not in results[lab]["accuracy"]]
        if missing:
            problems.append("{}: not evaluated for {}".format(d, ", ".join(missing)))
        seen = {}
        for lab in labels:
            info = (results[lab].get("datasets") or {}).get(d)
            if info is None:
                continue
            seen.setdefault((info.get("sha256"), info.get("n")), []).append(lab)
        if len(seen) > 1:
            parts = [
                "{} (n={}, sha {})".format("/".join(v), k[1], (k[0] or "?")[:8])
                for k, v in seen.items()
            ]
            problems.append("{}: different test data: {}".format(d, "; ".join(parts)))
    dec = {}
    for lab in labels:
        key = json.dumps(results[lab].get("decoding"), sort_keys=True)
        dec.setdefault(key, []).append(lab)
    if len(dec) > 1:
        problems.append(
            "different decoding settings: "
            + "; ".join("{} {}".format("/".join(v), k) for k, v in dec.items())
        )
    no_meta = [lab for lab in labels if not results[lab].get("datasets")]
    if no_meta:
        problems.append(
            "no dataset hashes recorded for {} (older eval); comparability not verified".format(
                ", ".join(no_meta)
            )
        )
    return problems


def _fmt(x):
    return "-" if x is None else f"{100 * x:.1f}"


def table(results):
    labels = _ordered(results)
    datasets = _datasets(results)
    head = ["Method"] + datasets + ["Avg"]
    rows = [head, ["---"] * len(head)]

    def avg(acc):
        vals = [acc[d] for d in datasets if acc.get(d) is not None]
        return sum(vals) / len(vals) if vals and len(vals) == len(datasets) else None

    for lab in labels:
        acc = results[lab]["accuracy"]
        rows.append([NAMES.get(lab, lab)] + [_fmt(acc.get(d)) for d in datasets] + [_fmt(avg(acc))])
    if "sft" in results and "sampling" in results:
        a, b = results["sampling"]["accuracy"], results["sft"]["accuracy"]
        delta = [
            "-" if a.get(d) is None or b.get(d) is None else f"{100 * (a[d] - b[d]):+.1f}"
            for d in datasets
        ]
        aa, ab = avg(a), avg(b)
        delta.append("-" if aa is None or ab is None else f"{100 * (aa - ab):+.1f}")
        rows.append(["Sampling - Plain SFT (pts)"] + delta)
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def main(argv=None):
    p = argparse.ArgumentParser(prog="vsft compare", description=__doc__.split("\n\n")[0])
    p.add_argument("paths", nargs="+", help="eval_summary.json files or directories to search")
    p.add_argument("--json", dest="json_out", help="also write the collected numbers here")
    p.add_argument(
        "--allow-mismatch", action="store_true", help="print the table even if rows differ"
    )
    a = p.parse_args(argv)
    results = collect(a.paths)
    if not results:
        print("no eval_summary.json found under: " + " ".join(a.paths), file=sys.stderr)
        return 1
    problems = check_comparable(results)
    hard = [x for x in problems if not x.startswith("no dataset hashes")]
    print("Single-shot accuracy (%):\n")
    print(table(results))
    missing = [x for x in ORDER if x not in results]
    if missing:
        print("\nnot evaluated yet: " + ", ".join(missing))
    for lab in _ordered(results):
        archs = ",".join(results[lab].get("architectures") or []) or "?"
        print(
            "  {}: {} [{}]  ({})".format(
                lab, results[lab].get("model"), archs, results[lab]["path"]
            )
        )
    if problems:
        print("\nWARNING - rows may not be comparable:")
        for x in problems:
            print("  - " + x)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(results, indent=2) + "\n")
    return 2 if hard and not a.allow_mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
