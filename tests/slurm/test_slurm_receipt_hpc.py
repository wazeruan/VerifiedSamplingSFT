"""receipt.py marker rules (regression, terminal, older-job races, flock) and item 9: hpc.py."""

import json
import os
import stat
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest
from conftest import REAL_VENV, REPO, _clean_env, make_venv_shim

RECEIPT = REPO / "slurm" / "lib" / "receipt.py"


def rc(*args, job=None, umask=None, check=True):
    env = _clean_env()
    if job is not None:
        env["SLURM_JOB_ID"] = str(job)
    pre = (lambda: os.umask(umask)) if umask is not None else None
    p = subprocess.run(["/usr/bin/python3", str(RECEIPT), *map(str, args)], env=env,
                       capture_output=True, text=True, preexec_fn=pre)
    if check:
        assert p.returncode == 0, p.stderr
    return p


def state(run, stage="s"):
    return json.loads((run / "status" / f"{stage}.state").read_text())


def test_same_job_never_regresses(tmp_path):
    rc("mark", tmp_path, "s", "STARTED", job=5)
    rc("mark", tmp_path, "s", "SUBMITTED", job=5)  # wrapper's late SUBMITTED
    rc("mark", tmp_path, "s", "PREFLIGHT_OK", job=5)
    assert state(tmp_path)["state"] == "STARTED"
    assert (tmp_path / "status" / "s.SUBMITTED").exists()  # per-state file still written


@pytest.mark.parametrize("first,second", [("SUCCEEDED", "FAILED"), ("FAILED", "SUCCEEDED")])
def test_terminal_is_final(tmp_path, first, second):
    rc("mark", tmp_path, "s", first, job=5)
    rc("mark", tmp_path, "s", second, job=5)
    assert state(tmp_path)["state"] == first


def test_newer_job_replaces_and_older_job_does_not(tmp_path):
    rc("mark", tmp_path, "s", "SUCCEEDED", job=5)
    rc("mark", tmp_path, "s", "SUBMITTED", job=7)
    assert state(tmp_path)["state"] == "SUBMITTED" and state(tmp_path)["job_id"] == "7"
    rc("mark", tmp_path, "s", "FAILED", job=5)  # old job's late trap
    assert state(tmp_path)["job_id"] == "7" and state(tmp_path)["state"] == "SUBMITTED"
    rc("mark", tmp_path, "s", "SUCCEEDED", job=7)
    rc("mark", tmp_path, "s", "FAILED", job=6)
    assert state(tmp_path)["state"] == "SUCCEEDED" and state(tmp_path)["job_id"] == "7"


def test_state_filter_by_job(tmp_path):
    rc("mark", tmp_path, "s", "SUCCEEDED", job=5)
    assert rc("state", tmp_path, "s").stdout.strip() == "SUCCEEDED"
    assert rc("state", tmp_path, "s", "5").stdout.strip() == "SUCCEEDED"
    assert rc("state", tmp_path, "s", "6").stdout.strip() == ""
    assert rc("state", tmp_path, "nostage").stdout.strip() == ""


def test_unknown_state_rejected(tmp_path):
    p = rc("mark", tmp_path, "s", "DONE", job=1, check=False)
    assert p.returncode != 0 and not (tmp_path / "status" / "s.state").exists()


def test_concurrent_marks_serialized(tmp_path):
    states = ["SUBMITTED"] * 12 + ["PREFLIGHT_OK"] * 6 + ["STARTED"] * 6 + ["SUCCEEDED"]
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda s: rc("mark", tmp_path, "s", s, job=9), states))
    assert state(tmp_path)["state"] == "SUCCEEDED"
    leftovers = [p.name for p in (tmp_path / "status").iterdir() if p.name.startswith(".tmp")]
    assert leftovers == []


def test_concurrent_updates_lose_nothing(tmp_path):
    f = tmp_path / "r.json"
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda i: rc("update", f, f"k{i}={i}"), range(30)))
        list(ex.map(lambda i: rc("append", f, "jobs", f"j{i}"), range(30)))
    obj = json.loads(f.read_text())
    assert all(obj[f"k{i}"] == i for i in range(30))
    assert sorted(obj["jobs"]) == sorted(f"j{i}" for i in range(30))


def test_claim_is_exclusive(tmp_path):
    f = tmp_path / "receipts" / "a.json"
    rc("claim", f, "state=SUBMITTING")
    p = rc("claim", f, "state=OTHER", check=False)
    assert p.returncode != 0 and "already exists" in p.stderr
    assert json.loads(f.read_text())["state"] == "SUBMITTING"


@pytest.mark.parametrize("umask,mode", [(0o022, 0o644), (0o002, 0o664), (0o077, 0o600)])
def test_files_honour_umask(tmp_path, umask, mode):
    rc("update", tmp_path / "u.json", "a=1", umask=umask)
    rc("mark", tmp_path, "s", "STARTED", job=1, umask=umask)
    for p in (tmp_path / "u.json", tmp_path / "status" / "s.state"):
        assert stat.S_IMODE(p.stat().st_mode) == mode, (p, oct(p.stat().st_mode))


def test_get_and_typed_values(tmp_path):
    f = tmp_path / "r.json"
    rc("update", f, "job_id=123", "flag=true", "text=hello world", "d={\"a\": 1}")
    assert rc("get", f, "job_id").stdout.strip() == "123"
    assert rc("get", f, "text").stdout.strip() == "hello world"
    assert rc("get", f, "missing").stdout.strip() == ""
    assert rc("get", tmp_path / "nofile.json", "x").stdout.strip() == ""


# ------------------------------------------------------------------ 9. hpc.py
VPY = str(REAL_VENV / "bin" / "python")


def hpc(*args, env=None, py=VPY):
    e = _clean_env()
    e.update(env or {})
    return subprocess.run([py, "-m", "vsft.hpc", *map(str, args)], capture_output=True,
                          text=True, env=e, timeout=120)


def make_train(d, losses=(1.0, 0.5), final=True, raw_history=None):
    if final:
        (d / "final").mkdir(parents=True)
        (d / "final" / "config.json").write_text("{}")
        (d / "final" / "model.safetensors").write_bytes(b"x")
    (d / "train_result.json").write_text("{}")
    hist = raw_history if raw_history is not None else "".join(
        json.dumps({"step": i, "loss": x}) + "\n" for i, x in enumerate(losses))
    (d / "train_history.jsonl").write_text(hist)
    return d


def test_check_train_ok(tmp_path):
    p = hpc("check-train", "--dir", make_train(tmp_path))
    assert p.returncode == 0 and "training output OK" in p.stdout


@pytest.mark.parametrize("kw,msg", [
    (dict(final=False), "config.json missing"),
    (dict(losses=(1.0, float("nan"))), "non-finite"),
    (dict(losses=(float("inf"),)), "non-finite"),
    (dict(raw_history='{"step": 1}\n'), "missing"),
    (dict(raw_history='{"step": 1, "loss": null}\n'), "1 non-finite, missing or unreadable"),
    (dict(raw_history='{"step": 1, "loss": "0.5"}\n'), "unreadable loss record"),
    (dict(raw_history='{"step": 1, "loss": true}\n'), "unreadable loss record"),
    (dict(raw_history='[1, 2]\n'), "unreadable loss record"),
    (dict(raw_history='{"step": 1, "loss": 0.5}\n{"step": 2, "lo'),
     "1 non-finite, missing or unreadable"),
])
def test_check_train_failures(tmp_path, kw, msg):
    p = hpc("check-train", "--dir", make_train(tmp_path, **kw))
    assert p.returncode == 3, (p.returncode, p.stderr[-500:])
    assert "Traceback" not in p.stderr
    assert msg in p.stderr


def test_preflight_cpu_writes_preflight_ok_not_started(tmp_path):
    p = hpc("preflight", "--run-dir", tmp_path, "--stage", "prepare",
            env={"SLURM_JOB_ID": "4242"})
    assert p.returncode == 0, p.stderr
    m = json.loads((tmp_path / "status" / "prepare.PREFLIGHT_OK").read_text())
    assert m["job_id"] == "4242" and m["slurm"]["SLURM_JOB_ID"] == "4242"
    assert not (tmp_path / "status" / "prepare.STARTED").exists()
    assert json.loads((tmp_path / "status" / "prepare.state").read_text())["state"] == \
        "PREFLIGHT_OK"


def test_preflight_require_gpu_fails_on_mac(tmp_path):
    p = hpc("preflight", "--run-dir", tmp_path, "--stage", "boost", "--require-gpu",
            "--require-bf16", "--gpu-type", "A100", "--min-gpu-mem-gb", "39",
            env={"SLURM_JOB_ID": "1"})
    assert p.returncode == 2
    assert "no CUDA device" in p.stderr
    assert not (tmp_path / "status").exists() or not list((tmp_path / "status").iterdir())


def test_preflight_inputs_and_model(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    p = hpc("preflight", "--run-dir", tmp_path, "--stage", "boost", "--require-file", empty,
            "--require-file", tmp_path / "missing.jsonl", "--require-model", tmp_path,
            env={"HF_HUB_OFFLINE": "1", "HF_HOME": str(tmp_path / "hf")})
    assert p.returncode == 2
    assert p.stderr.count("required input missing or empty") == 2
    assert "has no config.json" in p.stderr


def test_preflight_require_lock(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text("lock-a\n")
    venv = make_venv_shim(tmp_path / "venv")
    py = str(venv / "bin" / "python")
    run = tmp_path / "run"
    run.mkdir()
    p = hpc("preflight", "--run-dir", run, "--stage", "x", "--require-lock", lock, py=py)
    assert p.returncode == 2 and "stamp missing" in p.stderr
    assert hpc("stamp-lock", "--lock", lock, py=py).returncode == 0
    assert (venv / ".vsft-uv-lock.sha256").is_file()
    assert not (REAL_VENV / ".vsft-uv-lock.sha256").exists() or \
        (REAL_VENV / ".vsft-uv-lock.sha256").read_text() != (venv /
                                                             ".vsft-uv-lock.sha256").read_text()
    p = hpc("preflight", "--run-dir", run, "--stage", "x", "--require-lock", lock, py=py)
    assert p.returncode == 0, p.stderr
    lock.write_text("lock-b\n")
    p = hpc("preflight", "--run-dir", run, "--stage", "x", "--require-lock", lock, py=py)
    assert p.returncode == 2 and "was not built from" in p.stderr


def test_config_get(tmp_path):
    cfg = REPO / "configs" / "a100" / "math_qwen3b.yaml"
    p = hpc("config-get", "--config", cfg, "model.name_or_path")
    assert p.returncode == 0 and p.stdout.strip()
    import yaml
    assert p.stdout.strip() == yaml.safe_load(cfg.read_text())["model"]["name_or_path"]
    p = hpc("config-get", "--config", cfg, "--set", "train.lora_r=8", "train.lora_r")
    assert p.stdout.strip() == "8"
    p = hpc("config-get", "--config", cfg, "train.nope")
    assert p.returncode != 0


def test_notify_started_writes_marker(tmp_path):
    env = {"VSFT_RUN_DIR": str(tmp_path), "VSFT_STAGE": "boost", "SLURM_JOB_ID": "77",
           "VSFT_RECEIPT_PY": str(RECEIPT)}
    p = subprocess.run([VPY, "-c", "from vsft.hpc import notify_started; notify_started(a=1)"],
                       env={**_clean_env(), **env}, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    m = json.loads((tmp_path / "status" / "boost.STARTED").read_text())
    assert m["job_id"] == "77" and m["a"] == 1
    # non-zero ranks never write
    p = subprocess.run([VPY, "-c", "from vsft.hpc import notify_started; notify_started()"],
                       env={**_clean_env(), **env, "VSFT_STAGE": "train", "RANK": "1"},
                       capture_output=True, text=True)
    assert p.returncode == 0 and not (tmp_path / "status" / "train.STARTED").exists()
