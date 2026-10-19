"""Item 1/2: argument validation and --dry-run rendering of slurm/submit.sh."""

import re
import shlex

import pytest
from conftest import TS_RE


def test_help(h):
    r = h.run("submit.sh", "--help")
    assert r.returncode == 0
    assert "Usage: slurm/submit.sh STAGE --account" in r.stdout
    for flag in ("--venv", "--cache-root", "--online", "--preempt-grace", "--allow-duplicate",
                 "--eval-data", "--no-monitor"):
        assert flag in r.stdout
    assert h.run("submit.sh", "boost", "-h").returncode == 0


def test_no_args_is_usage_error(h):
    r = h.run("submit.sh")
    assert r.returncode == 2 and "Usage" in r.stderr


@pytest.mark.parametrize(
    "args,msg",
    [
        (["nonsense", "--account", "a"], "unknown stage"),
        (["boost", "--account", "a", "--bogus"], "unknown argument"),
        (["boost"], "--account is required"),
        (["boost", "--account", "a b"], "unexpected characters"),
        (["boost", "--account", "a", "--gpus", "a100:9"], "--gpus"),
        (["boost", "--account", "a", "--gpus", "2"], "--gpus"),
        (["boost", "--account", "a", "--mem", "128GB"], "--mem"),
        (["boost", "--account", "a", "--mem", "0G"], "--mem"),
        (["boost", "--account", "a", "--cpu-mem", "x"], "--cpu-mem"),
        (["boost", "--account", "a", "--depend-on", "abc"], "--depend-on"),
        (["boost", "--account", "a", "--depend-on", "12,13"], "--depend-on"),
        (["boost", "--account", "a", "--depend-on", "12:"], "--depend-on"),
        (["boost", "--account", "a", "--monitor-timeout", "5s"], "--monitor-timeout"),
        (["boost", "--account", "a", "--preempt-grace", "-1"], "--preempt-grace"),
        (["boost", "--account", "a", "--run-dir", "R1", "--resume", "R2"], "mutually exclusive"),
        (["boost", "--account", "a", "--resume", "R2", "--config", "c.yaml"], "fresh runs"),
        (["boost", "--account", "a", "--stages", "boost"], "only valid with pipeline"),
        (["pipeline", "--account", "a", "--stages", "boost,gpucheck"], "invalid pipeline stage"),
        (["pipeline", "--account", "a", "--stages", ","], "stage"),
        (["pipeline", "--account", "a", "--", "--limit", "3"], "single stage"),
        (["boost", "--account", "a", "--train-resume", "x"], "train or train_sft"),
        (["prepare", "--account", "a", "--datasets", "math aime"], "unknown dataset"),
        (["boost", "--account"], "requires a value"),
        (["boost", "--account", "--dry-run"], "requires a value"),
    ],
)
def test_rejected_without_submission(h, args, msg):
    r = h.run("submit.sh", *args)
    assert r.returncode == 2, (r.stdout, r.stderr)
    assert msg in r.stderr
    assert h.sbatch_calls() == []
    assert not h.runs.exists()


def test_missing_sbatch_is_rejected(h):
    env = {"PATH": f"{h.bin}:/usr/bin:/bin"}
    r = h.submit("gpucheck", env=env)
    assert r.returncode == 2 and "sbatch not found" in r.stderr
    assert not h.runs.exists()


def test_setup_without_uv_is_rejected(h):
    (h.bin / "uv").unlink()
    r = h.submit("setup")
    assert r.returncode == 2 and "uv" in r.stderr
    assert h.sbatch_calls() == []


def test_venv_missing_rejected(h):
    r = h.submit("gpucheck", "--venv", str(h.root / "novenv"))
    assert r.returncode == 2 and "venv missing" in r.stderr
    assert h.sbatch_calls() == []


def test_boost_input_missing_rejected(h):
    r = h.submit("boost")
    assert r.returncode == 2 and "boost input missing" in r.stderr
    assert h.sbatch_calls() == []


def _dry_lines(out):
    res = {}
    for line in out.splitlines():
        m = re.match(r"\[dry-run\] (\w+): (.*)$", line)
        if m:
            res[m.group(1)] = shlex.split(m.group(2))
    return res


def _opt(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def test_dry_run_pipeline(h):
    r = h.submit("pipeline", "--dry-run", "--account", "def-acct", account=None)
    assert r.returncode == 0, r.stderr
    lines = _dry_lines(r.stdout)
    assert list(lines) == ["setup", "prepare", "boost", "train", "eval"]
    assert h.sbatch_calls() == []
    assert not (h.runs / "x").exists(), "dry run must not claim the run dir"
    run_dir = str(h.runs / "x")
    prev = None
    for stage, argv in lines.items():
        assert argv[0] == "sbatch" and "--parsable" in argv
        assert _opt(argv, "--account") == "def-acct"
        assert _opt(argv, "--chdir") == run_dir
        assert argv[-1] == f"{run_dir}/code/slurm/jobs/{stage}.sbatch"
        assert re.fullmatch(rf"{re.escape(run_dir)}/logs/{stage}-{TS_RE}-%j\.out",
                            _opt(argv, "--output"))
        assert re.fullmatch(rf"{re.escape(run_dir)}/logs/{stage}-{TS_RE}-%j\.err",
                            _opt(argv, "--error"))
        if stage in ("boost", "train", "eval"):
            assert _opt(argv, "--gpus-per-node") == "a100:1"
            assert _opt(argv, "--mem") == "120G"
            assert _opt(argv, "--cpus-per-task") == "8"
        else:
            for f in ("--gpus-per-node", "--mem", "--cpus-per-task", "--gres", "--constraint"):
                assert f not in argv, (stage, f)
        if prev:
            assert _opt(argv, "--dependency") == f"afterok:DRYRUN_{prev}"
        else:
            assert "--dependency" not in argv
        prev = stage


def test_dry_run_resources(h):
    r = h.submit("pipeline", "--dry-run", "--stages", "prepare,boost,train",
                 "--train-gpus", "a100:4",
                 "--mem", "200G", "--cpu-mem", "16G", "--partition", "gpu",
                 "--cpu-partition", "cpu", "--constraint", "a100_80", "--qos", "q",
                 "--time", "1-00:00:00", "--depend-on", "77:78")
    assert r.returncode == 0, r.stderr
    lines = _dry_lines(r.stdout)
    p, b = lines["prepare"], lines["boost"]
    assert _opt(p, "--partition") == "cpu" and _opt(p, "--mem") == "16G"
    assert "--constraint" not in p and "--time" not in p and _opt(p, "--qos") == "q"
    assert _opt(p, "--dependency") == "afterok:77:78"
    assert _opt(b, "--gpus-per-node") == "a100:1" and _opt(b, "--mem") == "200G"
    assert _opt(lines["train"], "--gpus-per-node") == "a100:4"
    assert _opt(b, "--partition") == "gpu" and _opt(b, "--constraint") == "a100_80"
    assert _opt(b, "--time") == "1-00:00:00"


def test_dry_run_gpucheck_without_uv(h):
    (h.bin / "uv").unlink()  # only setup needs uv on the submission host
    r = h.submit("gpucheck", "--dry-run")
    assert r.returncode == 0, r.stderr
    argv = _dry_lines(r.stdout)["gpucheck"]
    assert _opt(argv, "--gpus-per-node") == "a100:1"


@pytest.mark.parametrize("stage", ["pipeline", "compare"])
def test_multi_gpu_gpus_refused_for_graphs(h, stage):
    r = h.submit(stage, "--dry-run", "--gpus", "a100:2")
    assert r.returncode == 2 and "--train-gpus" in r.stderr


@pytest.mark.parametrize("bad", ["4", "a100:9", "a100:0", "a100-2"])
def test_train_gpus_validated(h, bad):
    r = h.submit("train", "--dry-run", "--train-gpus", bad)
    assert r.returncode == 2 and "--train-gpus" in r.stderr


def test_train_gpus_only_for_training(h):
    r = h.submit("compare", "--dry-run", "--train-gpus", "a100:4")
    assert r.returncode == 0, r.stderr
    lines = _dry_lines(r.stdout)
    for stage, argv in lines.items():
        want = "a100:4" if stage in ("train", "train_sft") else (
            None if stage in ("setup", "prepare") else "a100:1")
        assert _opt(argv, "--gpus-per-node") == want, stage
    r = h.submit("train", "--dry-run", "--gpus", "a100:2")  # single stage: allowed
    assert r.returncode == 0 and _opt(_dry_lines(r.stdout)["train"], "--gpus-per-node") == \
        "a100:2"


# ------------------------------------------------------------------ separate verifier check
def _cfg(h, verifier_line):
    src = (h.project / "configs" / "a100" / "math_qwen35.yaml").read_text()
    out = re.sub(r"(?m)^(verifier:\n)  model: .*$", lambda m: m.group(1) + verifier_line, src)
    assert out != src or verifier_line in src
    p = h.root / "cfg.yaml"
    p.write_text(out)
    return p


def test_default_config_boost_needs_80gb(h):
    for stage in ("boost", "pipeline", "compare"):
        r = h.submit(stage, "--dry-run", min_gpu=False)
        assert r.returncode == 2 and "80 GB A100" in r.stderr, (stage, r.stderr)
        r = h.submit(stage, "--dry-run", "--min-gpu-mem-gb", "79.5", min_gpu=False)
        assert r.returncode == 0, (stage, r.stderr)
    for stage in ("train", "eval_base", "gpucheck", "train_sft"):
        r = h.submit(stage, "--dry-run", min_gpu=False)
        assert r.returncode == 0, (stage, r.stderr)


@pytest.mark.parametrize("line,ok", [
    ('  model: ""', True), ("  model: ''", True), ("  model:", True),
    ("  model: Qwen/Qwen3.5-4B   # self-verify", True),
    ("  model: Qwen/Qwen3.5-9B", False), ('  model: "Qwen/Qwen3.5-9B"', False),
])
def test_verifier_check_config_variants(h, line, ok):
    cfg = _cfg(h, line)
    r = h.submit("boost", "--dry-run", "--config", cfg, min_gpu=False)
    assert (r.returncode == 0) == ok, (line, r.stderr)


def test_verifier_check_before_claiming_run_dir(h):
    r = h.submit("boost", "--input", h.input, min_gpu=False)
    assert r.returncode == 2 and "80 GB A100" in r.stderr
    assert not (h.runs / "x").exists(), "rejected submission left a claimed run dir behind"
