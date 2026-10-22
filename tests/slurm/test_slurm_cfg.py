"""slurm/lib/cfg.py: the stdlib-only YAML subset reader used by slurm/submit.sh.

Runs under /usr/bin/python3 (login-node interpreter, no project venv) and must agree with
PyYAML on every key the scripts read from configs/h100/*.yaml.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
CFG_PY = REPO / "slurm" / "lib" / "cfg.py"
CONFIGS = REPO / "configs" / "h100"
PY = "/usr/bin/python3"

pytestmark = pytest.mark.skipif(not Path(PY).exists(), reason="needs /usr/bin/python3")


def run(path, *args):
    r = subprocess.run([PY, str(CFG_PY), str(path), *args], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr
    return r.stdout


def yml(tmp_path, text, name="c.yaml"):
    p = tmp_path / name
    p.write_text(text)
    return p


# ---------------------------------------------------------------- scalars


def test_bare_quoted_and_empty_scalars(tmp_path):
    p = yml(tmp_path, "task:\n"
                      "  bare: medical_train.jsonl\n"
                      "  dq: \"double quoted\"\n"
                      "  sq: 'single quoted'\n"
                      "  empty: \"\"\n"
                      "  num: 1.0e-4\n"
                      "  flag: true\n")
    assert run(p, "task.bare") == "medical_train.jsonl\n"
    assert run(p, "task.dq") == "double quoted\n"
    assert run(p, "task.sq") == "single quoted\n"
    assert run(p, "task.empty") == "\n"  # the empty string, printed as a blank line
    assert run(p, "task.num") == "1.0e-4\n"
    assert run(p, "task.flag") == "true\n"


def test_trailing_comments_and_comment_lines(tmp_path):
    p = yml(tmp_path, "# header comment\n\ntask:  # section comment\n"
                      "  name: math   # inline\n"
                      "  # a line between keys\n"
                      "  train_file: x.jsonl # why\n")
    assert run(p, "task.name") == "math\n"
    assert run(p, "task.train_file") == "x.jsonl\n"


def test_section_and_key_lookup_is_exact(tmp_path):
    p = yml(tmp_path, "task:\n  name: a\nmodel:\n  name: b\n")
    assert run(p, "task.name") == "a\n"
    assert run(p, "model.name") == "b\n"


# ---------------------------------------------------------------- lists


def test_flow_list(tmp_path):
    p = yml(tmp_path, "task:\n  prepare: [medical, mmlu, math500, gsm8k]\n  empty: []\n")
    assert run(p, "task.prepare").splitlines() == ["medical", "mmlu", "math500", "gsm8k"]
    assert run(p, "task.empty") == "\n"


def test_flow_list_quoted_items(tmp_path):
    p = yml(tmp_path, "task:\n  eval_files: [\"a.jsonl\", 'b.jsonl']\n")
    assert run(p, "task.eval_files").splitlines() == ["a.jsonl", "b.jsonl"]


def test_block_list_with_comments(tmp_path):
    p = yml(tmp_path, "task:\n  eval_files:  # the sets\n"
                      "    - medical_test.jsonl\n"
                      "    - mmlu_test.jsonl   # retention\n"
                      "    - \"gsm8k_test.jsonl\"\n"
                      "model:\n  name_or_path: x/y\n")
    assert run(p, "task.eval_files").splitlines() == [
        "medical_test.jsonl", "mmlu_test.jsonl", "gsm8k_test.jsonl"]
    assert run(p, "model.name_or_path") == "x/y\n"  # block list closed by the next key


def test_explicit_empty_list_is_not_defaulted(tmp_path):
    p = yml(tmp_path, "task:\n  eval_files: []\n")
    assert run(p, "task.eval_files", "fallback.jsonl") == "\n"


# ---------------------------------------------------------------- defaults


def test_missing_key_prints_defaults_one_per_line(tmp_path):
    p = yml(tmp_path, "task:\n  name: math\n")
    assert run(p, "task.eval_files", "a.jsonl", "b.jsonl") == "a.jsonl\nb.jsonl\n"
    assert run(p, "task.name", "fallback") == "math\n"  # present key wins over the default


def test_missing_section_prints_defaults(tmp_path):
    p = yml(tmp_path, "model:\n  name_or_path: x\n")
    assert run(p, "verifier.model", "none-set") == "none-set\n"


def test_missing_key_without_defaults_prints_blank(tmp_path):
    p = yml(tmp_path, "task:\n  name: math\n")
    assert run(p, "task.nope").strip() == ""


def test_usage_error_exits_nonzero(tmp_path):
    r = subprocess.run([PY, str(CFG_PY), str(tmp_path / "x.yaml")], capture_output=True,
                       text=True)
    assert r.returncode != 0


# ---------------------------------------------------------------- known limitations


def test_quoted_hash_is_not_a_comment(tmp_path):
    p = yml(tmp_path, "task:\n  name: \"run#1\"\n")
    assert run(p, "task.name") == "run#1\n"


def test_hash_inside_quotes_with_space_and_real_comment(tmp_path):
    p = yml(tmp_path, "task:\n  a: \"a #b\"\n  b: 'x #y'  # real comment\n"
                      "  c: \"q\"  # real comment\n  d: run#1 # real comment\n")
    assert run(p, "task.a") == "a #b\n"
    assert run(p, "task.b") == "x #y\n"
    assert run(p, "task.c") == "q\n"
    assert run(p, "task.d") == "run#1\n"


def test_bare_hash_without_space_is_not_a_comment(tmp_path):
    p = yml(tmp_path, "task:\n  name: a#b\n")
    assert run(p, "task.name") == "a#b\n"


def test_flow_list_quoted_item_with_comma(tmp_path):
    p = yml(tmp_path, "task:\n  eval_files: [\"a, b.jsonl\", c.jsonl]\n")
    assert run(p, "task.eval_files").splitlines() == ["a, b.jsonl", "c.jsonl"]


def test_flow_list_single_quoted_items_with_comma(tmp_path):
    p = yml(tmp_path, "task:\n  eval_files: ['a, b.jsonl', 'c, d.jsonl', e.jsonl]\n")
    assert run(p, "task.eval_files").splitlines() == ["a, b.jsonl", "c, d.jsonl", "e.jsonl"]


# ---------------------------------------------------------------- agreement with PyYAML


def _expect_scalar(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


@pytest.mark.parametrize("config", ["math_qwen35.yaml", "medical_qwen35.yaml"])
def test_agrees_with_pyyaml_on_shipped_configs(config):
    path = CONFIGS / config
    data = yaml.safe_load(path.read_text())
    checked = 0
    for section in ("task", "model", "verifier", "eval"):
        for key, want in (data.get(section) or {}).items():
            dotted = f"{section}.{key}"
            got = run(path, dotted)
            if isinstance(want, list):
                assert got.splitlines() == [str(x) for x in want], dotted
            elif isinstance(want, float):
                # PyYAML parses 1.0e-4 as a float; the reader keeps the text
                assert float(got.strip()) == pytest.approx(want), dotted
            else:
                assert got.strip() == _expect_scalar(want), dotted
            checked += 1
    assert checked >= 9  # math has 9 keys in task/model/verifier/eval; medical has more


def test_shipped_config_keys_exist_and_defaults_match(tmp_path):
    """The task.* keys the Slurm scripts read exist in the medical config."""
    path = CONFIGS / "medical_qwen35.yaml"
    assert run(path, "task.name").strip() == "medical"
    assert run(path, "task.train_file").strip() == "medical_train.jsonl"
    assert run(path, "eval.judge_model").strip() == "Qwen/Qwen3.5-9B"
    # math config predates the task section: the script falls back to the math defaults
    math_cfg = CONFIGS / "math_qwen35.yaml"
    assert run(math_cfg, "task.train_file", "math_l345_train.jsonl").strip() == (
        "math_l345_train.jsonl")
