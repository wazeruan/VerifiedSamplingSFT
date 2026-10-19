"""`vsft compare`: base vs plain SFT vs Sampling SFT accuracy table (stdlib only)."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from vsft.compare import collect, main, table

ROOT = Path(__file__).resolve().parents[1]
COMPARE_PY = ROOT / "src" / "vsft" / "compare.py"


def summary(d: Path, acc: dict, label=None, model="m", mtime=None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    s = {"model": model, "accuracy": acc}
    if label is not None:
        s["label"] = label
    f = d / "eval_summary.json"
    f.write_text(json.dumps(s))
    if mtime is not None:
        os.utime(f, (mtime, mtime))
    return f


def rows(text):
    """Markdown table -> list of cell lists (header first, separator dropped)."""
    out = []
    for line in text.splitlines():
        if line.startswith("|") and "---" not in line:
            out.append([c.strip() for c in line.strip("|").split("|")])
    return out


FULL = {"math500": 0.5, "gsm8k": 0.7}


def test_labels_from_summary_and_order(tmp_path):
    summary(tmp_path / "a", {"math500": 0.6, "gsm8k": 0.8}, "sampling", "ckpt/s")
    summary(tmp_path / "b", FULL, "base", "Qwen/Qwen3.5-4B")
    summary(tmp_path / "c", {"math500": 0.55, "gsm8k": 0.75}, "sft")
    summary(tmp_path / "d", {"math500": 0.1, "gsm8k": 0.2}, "zz_other")
    summary(tmp_path / "e", {"math500": 0.3, "gsm8k": 0.3}, "aa_other")
    res = collect([tmp_path])
    assert set(res) == {"base", "sft", "sampling", "zz_other", "aa_other"}
    assert res["sampling"]["model"] == "ckpt/s" and res["sampling"]["label"] == "sampling"
    r = rows(table(res))
    assert r[0] == ["Method", "math500", "gsm8k", "Avg"]
    assert [x[0] for x in r[1:]] == [
        "Initial model", "Plain SFT (expert traces)", "Sampling SFT", "aa_other", "zz_other",
        "Sampling - Plain SFT (pts)"]
    assert r[1] == ["Initial model", "50.0", "70.0", "60.0"]
    assert r[3] == ["Sampling SFT", "60.0", "80.0", "70.0"]
    assert r[-1] == ["Sampling - Plain SFT (pts)", "+5.0", "+5.0", "+5.0"]


def test_fallback_label_from_dir_name(tmp_path):
    summary(tmp_path / "sft-12345", FULL)  # no label key
    summary(tmp_path / "sampling-final-777", FULL)  # legacy "<label>-final-<id>"
    summary(tmp_path / "sampling-7", FULL, label=None)
    summary(tmp_path / "base-abc", FULL)  # no trailing digits: whole dir name
    summary(tmp_path / "x" / "eval", FULL)
    (tmp_path / "nolab").mkdir()
    f = summary(tmp_path / "nolab" / "base-99", FULL)
    data = json.loads(f.read_text())
    data["label"] = None  # `vsft eval` without --label writes null
    f.write_text(json.dumps(data))
    res = collect([tmp_path])
    assert {"sft", "sampling", "base-abc", "eval", "base"} == set(res)
    # newest "sampling" wins between sampling-7 and sampling-final-777; both map to it
    assert Path(res["sampling"]["path"]).parent.name in {"sampling-7", "sampling-final-777"}
    assert Path(res["sft"]["path"]).parent.name == "sft-12345"


@pytest.mark.parametrize(
    "dirname,label",
    [("plain-sft-2024", "plain-sft"), ("sft-final-12", "sft"), ("final-3", "final"),
     ("sampling-final", "sampling-final"), ("sft-final-x-1", "sft-final-x")],
)
def test_label_from_dir_name_variants(tmp_path, dirname, label):
    summary(tmp_path / dirname, FULL)
    assert set(collect([tmp_path])) == {label}


def test_newest_summary_per_label_wins(tmp_path):
    old = summary(tmp_path / "sft-1", {"math500": 0.1}, mtime=1_000_000)
    new = summary(tmp_path / "sft-2", {"math500": 0.9}, mtime=2_000_000)
    res = collect([tmp_path])
    assert res["sft"]["path"] == str(new) and res["sft"]["accuracy"] == {"math500": 0.9}
    # order of the given paths does not matter
    assert collect([new, old])["sft"]["path"] == str(new)
    # an explicit label beats the dir name, newest still wins
    summary(tmp_path / "zzz", {"math500": 0.5}, label="sft", mtime=3_000_000)
    assert collect([tmp_path])["sft"]["accuracy"] == {"math500": 0.5}


def test_missing_datasets_and_avg(tmp_path):
    summary(tmp_path / "a", {"math500": 0.5, "gsm8k": 0.7}, "sft")
    summary(tmp_path / "b", {"math500": 0.6}, "sampling")  # gsm8k not evaluated
    summary(tmp_path / "c", {"gsm8k": 0.4, "aime": 0.1}, "base")
    r = rows(table(collect([tmp_path])))
    assert r[0] == ["Method", "gsm8k", "aime", "math500", "Avg"]  # base first: its order
    by = {x[0]: x[1:] for x in r[1:]}
    assert by["Initial model"] == ["40.0", "10.0", "-", "-"]
    assert by["Plain SFT (expert traces)"] == ["70.0", "-", "50.0", "-"]
    assert by["Sampling SFT"] == ["-", "-", "60.0", "-"]
    assert by["Sampling - Plain SFT (pts)"] == ["-", "-", "+10.0", "-"]


def test_no_delta_row_without_both(tmp_path):
    summary(tmp_path / "a", FULL, "base")
    summary(tmp_path / "b", FULL, "sampling")
    t = table(collect([tmp_path]))
    assert "Sampling - Plain SFT" not in t
    summary(tmp_path / "c", {"math500": 0.6, "gsm8k": 0.6}, "sft")
    r = rows(table(collect([tmp_path])))
    assert r[-1] == ["Sampling - Plain SFT (pts)", "-10.0", "+10.0", "+0.0"]


def test_zero_accuracy_is_a_value(tmp_path):
    summary(tmp_path / "a", {"math500": 0.0, "gsm8k": 0.0}, "base")
    assert rows(table(collect([tmp_path])))[1] == ["Initial model", "0.0", "0.0", "0.0"]


def test_collect_skips_bad_files(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "eval_summary.json").write_text("{torn")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "eval_summary.json").write_text('["not", "a", "dict"]')
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "eval_summary.json").write_text('{"accuracy": 3}')
    (tmp_path / "other.json").write_text(json.dumps({"label": "base", "accuracy": FULL}))
    assert collect([tmp_path, tmp_path / "missing"]) == {}
    # an explicit file path is read whatever its name
    assert set(collect([tmp_path / "other.json"])) == {"base"}


def test_main_output_and_json(tmp_path, capsys):
    summary(tmp_path / "r" / "sft-1", FULL)
    summary(tmp_path / "r" / "sampling-2", {"math500": 0.6, "gsm8k": 0.9})
    out = tmp_path / "cmp.json"
    assert main([str(tmp_path / "r"), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "Single-shot accuracy (%)" in text
    assert "| Sampling - Plain SFT (pts) | +10.0 | +20.0 | +15.0 |" in text
    assert "not evaluated yet: base" in text
    data = json.loads(out.read_text())
    assert set(data) == {"sft", "sampling"}
    assert data["sampling"]["accuracy"] == {"math500": 0.6, "gsm8k": 0.9}
    assert data["sft"]["label"] == "sft" and data["sft"]["path"].endswith("eval_summary.json")


def test_main_nothing_found(tmp_path, capsys):
    assert main([str(tmp_path)]) == 1
    assert "no eval_summary.json found" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_cli_wrapper(tmp_path, capsys):
    from vsft.cli import main as vsft_main

    summary(tmp_path / "base-1", FULL)
    out = tmp_path / "o.json"
    assert vsft_main(["compare", str(tmp_path), "--json", str(out)]) == 0
    assert "| Initial model | 50.0 | 70.0 | 60.0 |" in capsys.readouterr().out
    assert set(json.loads(out.read_text())) == {"base"}
    assert vsft_main(["compare", str(tmp_path / "empty")]) == 1


def _run_standalone(python, args, tmp_path, *flags):
    # cwd outside the repo and no PYTHONPATH: only the stdlib is importable
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
    return subprocess.run([python, *flags, str(COMPARE_PY), *args], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=60)


def test_standalone_script_without_site_packages(tmp_path):
    summary(tmp_path / "runs" / "sft-1", FULL)
    summary(tmp_path / "runs" / "sampling-2", FULL)
    # -I: isolated (no user site, no env vars), -S: no site-packages at all
    p = _run_standalone(sys.executable, [str(tmp_path / "runs")], tmp_path, "-I", "-S")
    assert p.returncode == 0, p.stderr
    assert "| Sampling - Plain SFT (pts) | +0.0 | +0.0 | +0.0 |" in p.stdout
    p = _run_standalone(sys.executable, [str(tmp_path / "nothing")], tmp_path, "-I", "-S")
    assert p.returncode == 1


def test_standalone_with_system_python(tmp_path):
    py = "/usr/bin/python3" if Path("/usr/bin/python3").exists() else shutil.which("python3")
    if not py or str(ROOT / ".venv") in str(Path(py).resolve()):
        pytest.skip("no system python3 outside the venv")
    summary(tmp_path / "runs" / "base-1", FULL, label="base")
    out = tmp_path / "o.json"
    p = _run_standalone(py, [str(tmp_path / "runs"), "--json", str(out)], tmp_path, "-I")
    assert p.returncode == 0, p.stderr
    assert "| Initial model | 50.0 | 70.0 | 60.0 |" in p.stdout
    assert json.loads(out.read_text())["base"]["accuracy"] == FULL


def test_compare_imports_only_stdlib():
    import ast

    tree = ast.parse(COMPARE_PY.read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    assert mods <= stdlib, mods - stdlib


# ------------------------------------------------------------------ comparability checks


def meta(n=3, sha="a" * 64, limit=0):
    return {"n": n, "sha256": sha, "limit": limit}


def full_summary(d, label, acc, datasets=None, decoding=None, mtime=None):
    d.mkdir(parents=True, exist_ok=True)
    s = {"model": "m", "label": label, "accuracy": acc,
         "datasets": datasets if datasets is not None else {k: meta() for k in acc},
         "decoding": decoding or {"temperature": 0.0, "max_new_tokens": 1024}}
    f = d / "eval_summary.json"
    f.write_text(json.dumps(s))
    if mtime:
        os.utime(f, (mtime, mtime))
    return f


def test_check_comparable_clean(tmp_path, capsys):
    from vsft.compare import check_comparable

    for lab in ("base", "sft", "sampling"):
        full_summary(tmp_path / lab, lab, dict(FULL))
    res = collect([tmp_path])
    assert check_comparable(res) == []
    assert main([str(tmp_path)]) == 0
    assert "WARNING" not in capsys.readouterr().out


def test_missing_dataset_is_hard(tmp_path, capsys):
    from vsft.compare import check_comparable

    full_summary(tmp_path / "a", "sft", dict(FULL))
    full_summary(tmp_path / "b", "sampling", {"math500": 0.6})
    probs = check_comparable(collect([tmp_path]))
    assert probs == ["gsm8k: not evaluated for sampling"]
    assert main([str(tmp_path)]) == 2
    out = capsys.readouterr().out
    assert "WARNING - rows may not be comparable" in out and "| Sampling SFT |" in out
    assert main([str(tmp_path), "--allow-mismatch"]) == 0


def test_different_test_data_is_hard(tmp_path):
    from vsft.compare import check_comparable

    full_summary(tmp_path / "a", "sft", {"math500": 0.5},
                 datasets={"math500": meta(n=500, sha="1" * 64)})
    full_summary(tmp_path / "b", "sampling", {"math500": 0.6},
                 datasets={"math500": meta(n=100, sha="1" * 64)})
    full_summary(tmp_path / "c", "base", {"math500": 0.4},
                 datasets={"math500": meta(n=500, sha="2" * 64)})
    [p] = check_comparable(collect([tmp_path]))
    assert p.startswith("math500: different test data:")
    assert "sft (n=500, sha 11111111)" in p and "sampling (n=100" in p and "base (n=500, sha 2222"\
        in p
    assert main([str(tmp_path)]) == 2


def test_different_decoding_is_hard(tmp_path):
    from vsft.compare import check_comparable

    full_summary(tmp_path / "a", "sft", dict(FULL))
    full_summary(tmp_path / "b", "sampling", dict(FULL),
                 decoding={"temperature": 0.7, "max_new_tokens": 1024})
    [p] = check_comparable(collect([tmp_path]))
    assert p.startswith("different decoding settings:") and "sampling" in p
    assert main([str(tmp_path)]) == 2


def test_missing_hashes_are_soft(tmp_path, capsys):
    from vsft.compare import check_comparable

    full_summary(tmp_path / "a", "sft", dict(FULL))
    summary(tmp_path / "sampling-1", dict(FULL))  # older eval: no datasets/decoding keys
    probs = check_comparable(collect([tmp_path]))
    # the old summary also lacks "decoding", which differs from the new one's
    assert any(p.startswith("no dataset hashes recorded for sampling") for p in probs)
    soft_only = [p for p in probs if not p.startswith("no dataset hashes")]
    assert main([str(tmp_path)]) == (2 if soft_only else 0)


def test_only_soft_problems_exit_zero(tmp_path, capsys):
    summary(tmp_path / "sft-1", dict(FULL))
    summary(tmp_path / "sampling-2", dict(FULL))
    assert main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "no dataset hashes recorded for sft, sampling" in out


def test_cli_passes_allow_mismatch(tmp_path):
    from vsft.cli import main as vsft_main

    full_summary(tmp_path / "a", "sft", dict(FULL))
    full_summary(tmp_path / "b", "sampling", {"math500": 0.6})
    assert vsft_main(["compare", str(tmp_path)]) == 2
    assert vsft_main(["compare", str(tmp_path), "--allow-mismatch"]) == 0


def test_eval_summaries_are_comparable_end_to_end(tmp_path, monkeypatch):
    """Real `vsft eval` summaries (FakeBackend) feed compare without mismatches."""
    from fakes import FakeBackend

    import vsft.backends.base as base_mod
    from vsft.backends.base import Generation
    from vsft.cli import main as vsft_main

    fixture = Path(__file__).parent / "fixtures" / "tiny_math.jsonl"
    monkeypatch.setattr(base_mod, "build_backend",
                        lambda m, b: FakeBackend(lambda *a: [], lambda *a: ""))
    monkeypatch.setattr(FakeBackend, "generate", lambda self, p, n, temperature, top_p,
                        max_tokens: [[Generation("\\boxed{42}", True)] for _ in p])
    for lab in ("base", "sft", "sampling"):
        assert vsft_main(["eval", "--data", str(fixture), "--output-dir",
                          str(tmp_path / f"{lab}-1"), "--label", lab]) == 0
    s = json.loads((tmp_path / "sft-1" / "eval_summary.json").read_text())
    import hashlib

    assert s["datasets"] == {"tiny_math": {
        "n": 3, "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(), "limit": 0}}
    assert s["decoding"] == {"temperature": 0.0, "max_new_tokens": 1024}
    assert main([str(tmp_path)]) == 0
    # a --limit run is different test data (n differs)
    assert vsft_main(["eval", "--data", str(fixture), "--output-dir",
                      str(tmp_path / "sampling-2"), "--label", "sampling", "--limit", "2"]) == 0
    s2 = json.loads((tmp_path / "sampling-2" / "eval_summary.json").read_text())
    assert s2["datasets"]["tiny_math"]["n"] == 2 and s2["datasets"]["tiny_math"]["limit"] == 2
    assert main([str(tmp_path)]) == 2


def test_compare_source_is_py36_compatible():
    import ast

    src = COMPARE_PY.read_text()
    tree = ast.parse(src, feature_version=(3, 6))
    assert not any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
                   for n in ast.walk(tree))
    # no walrus / positional-only / match etc. parsed fine above; no PEP 604 at runtime
    assert " | None" not in src
