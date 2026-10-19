import hashlib
import json
import sys
from pathlib import Path

import pytest

import vsft.hpc as hpc
from vsft.hpc import main

ROOT = Path(__file__).resolve().parents[1]


def make_run(d, losses=(1.0, 0.5), result=True, weights=True):
    (d / "final").mkdir(parents=True)
    (d / "final" / "config.json").write_text("{}")
    if weights:
        (d / "final" / "model.safetensors").write_bytes(b"x")
    if result:
        (d / "train_result.json").write_text("{}")
    (d / "train_history.jsonl").write_text(
        "".join(json.dumps({"step": i, "loss": x}) + "\n" for i, x in enumerate(losses))
    )
    return d


def test_check_train_ok(tmp_path, capsys):
    make_run(tmp_path)
    assert main(["check-train", "--dir", str(tmp_path)]) == 0
    assert "training output OK" in capsys.readouterr().out


@pytest.mark.parametrize(
    "kw,msg",
    [
        (dict(losses=(1.0, float("nan"))), "has 1 non-finite, missing or unreadable loss"),
        (dict(losses=(float("inf"), float("-inf"))),
         "has 2 non-finite, missing or unreadable loss"),
        (dict(result=False), "train_result.json missing"),
        (dict(weights=False), "no *.safetensors"),
    ],
)
def test_check_train_failures(tmp_path, capsys, kw, msg):
    make_run(tmp_path, **kw)
    assert main(["check-train", "--dir", str(tmp_path)]) == 3
    assert msg in capsys.readouterr().err


def test_check_train_empty_dir(tmp_path, capsys):
    assert main(["check-train", "--dir", str(tmp_path)]) == 3
    assert "config.json missing" in capsys.readouterr().err


def test_config_get(tmp_path, capsys):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  name_or_path: some/model\ntrain:\n  learning_rate: 2e-5\n")
    assert main(["config-get", "--config", str(p), "model.name_or_path"]) == 0
    assert capsys.readouterr().out.strip() == "some/model"
    main(["config-get", "--config", str(p), "--set", "train.lora_r=8", "train.lora_r"])
    assert capsys.readouterr().out.strip() == "8"
    main(["config-get", "--config", str(p), "train.learning_rate"])
    assert float(capsys.readouterr().out) == pytest.approx(2e-5)
    with pytest.raises(AttributeError):
        main(["config-get", "--config", str(p), "train.nope"])


def test_preflight_cpu_only(tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    need = tmp_path / "in.jsonl"
    need.write_text("{}\n")
    assert main(["preflight", "--run-dir", str(run), "--stage", "boost",
                 "--require-file", str(need)]) == 0
    ok = json.loads((run / "status" / "boost.PREFLIGHT_OK").read_text())
    assert ok["stage"] == "boost" and ok["state"] == "PREFLIGHT_OK"
    assert "accelerator" in ok and "slurm" in ok
    assert json.loads((run / "status" / "boost.state").read_text())["state"] == "PREFLIGHT_OK"
    assert not (run / "status" / "boost.STARTED").exists()  # the application writes STARTED
    # missing input -> nonzero and no marker
    run2 = tmp_path / "run2"
    run2.mkdir()
    assert main(["preflight", "--run-dir", str(run2), "--stage", "b",
                 "--require-file", str(tmp_path / "missing")]) == 2
    assert not (run2 / "status").exists()


def test_check_train_missing_loss_key(tmp_path, capsys):
    make_run(tmp_path)
    with open(tmp_path / "train_history.jsonl", "a") as f:
        f.write('{"step": 9}\n')
    assert main(["check-train", "--dir", str(tmp_path)]) == 3
    assert ("train_history.jsonl has 1 non-finite, missing or unreadable loss record(s)"
            in capsys.readouterr().err)


@pytest.mark.parametrize(
    "lines,n_bad",
    [
        (['{"step": 1, "loss": null}'], 1),
        (['{"step": 1, "loss": true}', '{"step": 2, "loss": false}'], 2),
        (['{"step": 1, "loss": "0.5"}'], 1),
        (['{"step": 1, "loss": [0.5]}'], 1),
        (['[1, 2]'], 1),
        (['{"step": 1, "loss": 0.5}', '{"step": 2, "lo'], 1),  # torn last line
    ],
)
def test_check_train_bad_history_records(tmp_path, capsys, lines, n_bad):
    make_run(tmp_path, losses=(1.0,))
    with open(tmp_path / "train_history.jsonl", "a") as f:
        f.write("\n".join(lines))
    assert main(["check-train", "--dir", str(tmp_path)]) == 3
    assert (f"train_history.jsonl has {n_bad} non-finite, missing or unreadable loss record(s)"
            in capsys.readouterr().err)


def test_check_train_accepts_int_loss_and_blank_lines(tmp_path):
    make_run(tmp_path, losses=(1, 0.5))
    with open(tmp_path / "train_history.jsonl", "a") as f:
        f.write("\n\n")
    assert main(["check-train", "--dir", str(tmp_path)]) == 0


# ------------------------------------------------------------- preflight: model / lock


def _preflight(run, *extra):
    return main(["preflight", "--run-dir", str(run), "--stage", "s", *extra])


def test_require_model_local_dir(tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    good = tmp_path / "m"
    good.mkdir()
    (good / "config.json").write_text("{}")
    assert _preflight(run, "--require-model", str(good)) == 0
    bad = tmp_path / "empty"
    bad.mkdir()
    assert _preflight(run, "--require-model", str(bad)) == 2
    assert "has no config.json" in capsys.readouterr().err


def _fake_hf_cache(cache, repo, files):
    d = cache / f"models--{repo.replace('/', '--')}"
    commit = "a" * 40
    (d / "refs").mkdir(parents=True)
    (d / "refs" / "main").write_text(commit)
    snap = d / "snapshots" / commit
    snap.mkdir(parents=True)
    for f in files:
        (snap / f).write_text("{}")


def test_require_model_hf_cache(tmp_path, monkeypatch, capsys):
    import huggingface_hub.constants as C

    cache = tmp_path / "hub"
    _fake_hf_cache(cache, "org/full", ["config.json", "model.safetensors"])
    _fake_hf_cache(cache, "org/noweights", ["config.json"])
    monkeypatch.setattr(C, "HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert hpc.model_available("org/full") is None
    assert "no *.safetensors" in hpc.model_available("org/noweights")
    assert "not in the HF cache" in hpc.model_available("org/missing")
    run = tmp_path / "run"
    run.mkdir()
    assert _preflight(run, "--require-model", "org/full") == 0
    assert _preflight(run, "--require-model", "org/full", "--require-model", "org/missing") == 2
    err = capsys.readouterr().err
    assert "org/missing" in err and "run the prepare stage" in err


def test_require_model_real_qwen_cache(monkeypatch):
    import huggingface_hub.constants as C

    cache = ROOT / ".scratch" / "hf" / "hub"
    if not (cache / "models--Qwen--Qwen3.5-0.8B").is_dir():
        pytest.skip("Qwen not in .scratch/hf")
    monkeypatch.setattr(C, "HF_HUB_CACHE", str(cache))
    assert hpc.model_available("Qwen/Qwen3.5-0.8B") is None


def test_stamp_lock_and_require_lock(tmp_path, monkeypatch, capsys):
    venv = tmp_path / "venv"
    venv.mkdir()
    monkeypatch.setattr(sys, "prefix", str(venv))  # never touch the real project venv
    lock = tmp_path / "uv.lock"
    lock.write_text("version = 1\n")
    run = tmp_path / "run"
    run.mkdir()
    # no stamp yet
    assert _preflight(run, "--require-lock", str(lock)) == 2
    assert "stamp missing" in capsys.readouterr().err
    assert main(["stamp-lock", "--lock", str(lock)]) == 0
    stamp = (venv / hpc.LOCK_STAMP).read_text().strip()
    assert stamp == hashlib.sha256(lock.read_bytes()).hexdigest()
    assert _preflight(run, "--require-lock", str(lock)) == 0
    # lock changed after the venv was built
    lock.write_text("version = 2\n")
    assert _preflight(run, "--require-lock", str(lock)) == 2
    assert "run the setup stage" in capsys.readouterr().err


# ------------------------------------------------------- notify_started / stop_requested


def test_notify_started_requires_env(tmp_path):
    hpc.notify_started(x=1)  # no env: no-op, no error
    assert not list(tmp_path.iterdir())


def test_notify_started_writes_started(tmp_path, monkeypatch):
    monkeypatch.setenv("VSFT_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("VSFT_STAGE", "train")
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    hpc.notify_started(num_examples=3)
    m = json.loads((tmp_path / "status" / "train.STARTED").read_text())
    assert m["state"] == "STARTED" and m["num_examples"] == 3 and m["job_id"] == "42"
    assert json.loads((tmp_path / "status" / "train.state").read_text())["state"] == "STARTED"


@pytest.mark.parametrize(
    "env,writes",
    [({"RANK": "1"}, False), ({"LOCAL_RANK": "1"}, False), ({"RANK": "0", "LOCAL_RANK": "1"}, True),
     ({"RANK": "0"}, True), ({"LOCAL_RANK": "0"}, True)],
)
def test_notify_started_rank_gate(tmp_path, monkeypatch, env, writes):
    monkeypatch.setenv("VSFT_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("VSFT_STAGE", "boost")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    hpc.notify_started()
    assert (tmp_path / "status" / "boost.STARTED").exists() is writes


def test_notify_started_needs_both_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("VSFT_RUN_DIR", str(tmp_path))
    hpc.notify_started()
    monkeypatch.delenv("VSFT_RUN_DIR")
    monkeypatch.setenv("VSFT_STAGE", "x")
    hpc.notify_started()
    assert not (tmp_path / "status").exists()


def test_stop_requested(tmp_path, monkeypatch):
    assert hpc.stop_requested() is False
    f = tmp_path / "STOP"
    monkeypatch.setenv("VSFT_STOP_FILE", str(f))
    assert hpc.stop_requested() is False
    f.write_text("")
    assert hpc.stop_requested() is True
    monkeypatch.setenv("VSFT_STOP_FILE", "")
    assert hpc.stop_requested() is False


def test_receipt_override_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VSFT_RECEIPT_PY", str(tmp_path / "missing.py"))
    with pytest.raises(FileNotFoundError):
        hpc.write_marker(tmp_path, "s", "STARTED")


# ---------------------------------------------------------------- state ordering rules


def _state(run, stage="st"):
    return json.loads((run / "status" / f"{stage}.state").read_text())


def test_state_never_regresses_for_same_job(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    for s in ("SUBMITTED", "PREFLIGHT_OK", "STARTED"):
        hpc.write_marker(tmp_path, "st", s)
        assert _state(tmp_path)["state"] == s
    # the wrapper's late SUBMITTED / a repeated preflight cannot move it back
    hpc.write_marker(tmp_path, "st", "SUBMITTED")
    hpc.write_marker(tmp_path, "st", "PREFLIGHT_OK")
    assert _state(tmp_path)["state"] == "STARTED"
    # but every per-state file is still written
    for s in ("SUBMITTED", "PREFLIGHT_OK", "STARTED"):
        assert (tmp_path / "status" / f"st.{s}").is_file()
    hpc.write_marker(tmp_path, "st", "STARTED", extra=1)  # same rank: replaces
    assert _state(tmp_path)["extra"] == 1
    hpc.write_marker(tmp_path, "st", "SUCCEEDED")
    assert _state(tmp_path)["state"] == "SUCCEEDED"
    # terminal states are final for the job
    for s in ("FAILED", "STARTED", "SUBMITTED"):
        hpc.write_marker(tmp_path, "st", s)
        assert _state(tmp_path)["state"] == "SUCCEEDED"
    hpc.write_marker(tmp_path, "st", "SUCCEEDED", again=True)
    assert _state(tmp_path)["state"] == "SUCCEEDED"


def test_failed_is_terminal(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    hpc.write_marker(tmp_path, "st", "STARTED")
    hpc.write_marker(tmp_path, "st", "FAILED")
    hpc.write_marker(tmp_path, "st", "SUCCEEDED")
    assert _state(tmp_path)["state"] == "FAILED"


def test_new_job_replaces_state(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    hpc.write_marker(tmp_path, "st", "SUCCEEDED")
    monkeypatch.setenv("SLURM_JOB_ID", "2")
    hpc.write_marker(tmp_path, "st", "SUBMITTED")
    st = _state(tmp_path)
    assert st["state"] == "SUBMITTED" and st["job_id"] == "2"
    # stages are independent
    hpc.write_marker(tmp_path, "other", "STARTED")
    assert _state(tmp_path)["state"] == "SUBMITTED"
    assert _state(tmp_path, "other")["state"] == "STARTED"


def test_unknown_state_rejected(tmp_path):
    with pytest.raises(SystemExit, match="unknown state"):
        hpc.write_marker(tmp_path, "st", "RUNNING")
    assert not (tmp_path / "status" / "st.state").exists()


# ------------------------------------------------------- SIGTERM -> graceful stop


def test_request_stop_sets_flag_and_touches_file(tmp_path, monkeypatch):
    f = tmp_path / "STOP"
    monkeypatch.setenv("VSFT_STOP_FILE", str(f))
    assert not hpc.stop_requested()
    hpc.request_stop()
    assert hpc._STOP_FLAG is True and f.exists() and hpc.stop_requested()
    f.unlink()
    assert hpc.stop_requested()  # the in-process flag persists


def test_request_stop_without_stop_file(monkeypatch):
    hpc.request_stop(15, None)  # signal-handler signature
    assert hpc.stop_requested()


def test_request_stop_unwritable_stop_file(tmp_path, monkeypatch):
    monkeypatch.setenv("VSFT_STOP_FILE", str(tmp_path / "no" / "such" / "dir" / "STOP"))
    hpc.request_stop()
    assert hpc.stop_requested()


def test_flag_reset_between_tests():
    assert hpc._STOP_FLAG is False and not hpc.stop_requested()


def test_install_term_handler_and_sigterm(tmp_path, monkeypatch):
    import os
    import signal

    original = signal.getsignal(signal.SIGTERM)
    hpc.install_term_handler()  # no VSFT_STAGE: unchanged
    assert signal.getsignal(signal.SIGTERM) is original
    f = tmp_path / "STOP"
    monkeypatch.setenv("VSFT_STAGE", "train")
    monkeypatch.setenv("VSFT_STOP_FILE", str(f))
    try:
        hpc.install_term_handler()
        assert signal.getsignal(signal.SIGTERM) is hpc.request_stop
        os.kill(os.getpid(), signal.SIGTERM)  # handled: the process keeps running
        assert hpc.stop_requested() and f.exists()
    finally:
        signal.signal(signal.SIGTERM, original)


def test_record_stopped(tmp_path, monkeypatch):
    hpc.record_stopped(written=1)  # no path, no env: no-op
    assert not list(tmp_path.iterdir())
    monkeypatch.setenv("SLURM_JOB_ID", "9")
    p = tmp_path / "a" / "rec.json"
    hpc.record_stopped(p, step=3, state_dir="x")
    assert json.loads(p.read_text()) == {"job_id": "9", "step": 3, "state_dir": "x"}
    monkeypatch.setenv("VSFT_STOPPED_RECORD", str(tmp_path / "env.json"))
    monkeypatch.delenv("SLURM_JOB_ID")
    hpc.record_stopped(written=2, total=5)
    assert json.loads((tmp_path / "env.json").read_text()) == {
        "job_id": None, "written": 2, "total": 5}


# ------------------------------------------------------------------ A100 math flags


@pytest.mark.parametrize("cap,expect", [((8, 0), True), ((9, 0), True), ((7, 0), False)])
def test_enable_a100_math(monkeypatch, cap, expect):
    import torch

    mm, cudnn = torch.backends.cuda.matmul, torch.backends.cudnn
    before = (mm.allow_tf32, cudnn.allow_tf32)
    # monkeypatch restores both flags at teardown, whatever enable_a100_math did
    monkeypatch.setattr(mm, "allow_tf32", False)
    monkeypatch.setattr(cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: cap)
    hpc.enable_a100_math()
    assert mm.allow_tf32 is expect and cudnn.allow_tf32 is expect
    monkeypatch.undo()
    assert (mm.allow_tf32, cudnn.allow_tf32) == before


def test_enable_a100_math_noop_without_cuda(monkeypatch):
    import torch

    mm, cudnn = torch.backends.cuda.matmul, torch.backends.cudnn
    monkeypatch.setattr(mm, "allow_tf32", False)
    monkeypatch.setattr(cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def no_device(*a, **k):
        raise AssertionError("capability must not be queried without CUDA")

    monkeypatch.setattr(torch.cuda, "get_device_capability", no_device)
    hpc.enable_a100_math()
    assert mm.allow_tf32 is False and cudnn.allow_tf32 is False


def test_enable_a100_math_is_wired_in():
    import inspect

    import vsft.backends.hf as hf
    import vsft.train as tr

    t = inspect.getsource(tr.train)
    assert "enable_a100_math()" in t and t.index("set_seed(t.seed)") < t.index(
        "enable_a100_math()")
    assert "enable_a100_math()" in inspect.getsource(hf.HFBackend.__init__)
