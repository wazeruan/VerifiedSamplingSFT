"""Item 10: status.sh, results.sh, logs.sh on synthetic run dirs; remote.sh with fake
ssh/rsync; fetch_online.sh argument validation and failure marking with fake uv/python."""

import json
import os
import shlex
import time

import pytest


def mkrun(h, name="run"):
    run = h.root / name
    for d in ("status", "receipts", "logs", "eval/m-final-1", "sft"):
        (run / d).mkdir(parents=True, exist_ok=True)
    (run / "manifest.json").write_text("{}")
    return run


def mark(run, stage, state, job, **kw):
    obj = {"stage": stage, "state": state, "job_id": job, "time_utc": "2026-10-19T00:00:00Z",
           **kw}
    (run / "status" / f"{stage}.state").write_text(json.dumps(obj))


# ------------------------------------------------------------------ status.sh
def test_status(h):
    run = mkrun(h)
    h.add_job(1001, "RUNNING|0:0|None@0")
    h.add_job(1002, "COMPLETED|0:0|None@0")
    mark(run, "boost", "STARTED", "1001")
    mark(run, "prepare", "SUCCEEDED", 1002)
    mark(run, "setup", "SUCCEEDED", None)
    (run / "receipts" / "boost-20261019T000000Z-abcdef12.json").write_text(
        json.dumps({"log_out": "/x/boost-1001.out"}))
    r = h.run("status.sh", run)
    assert r.returncode == 0, r.stderr
    rows = {ln.split()[0]: ln.split() for ln in r.stdout.splitlines()[1:]}
    assert rows["boost"][1:4] == ["STARTED", "1001", "RUNNING"]
    assert rows["boost"][-1] == "/x/boost-1001.out"
    assert rows["prepare"][1:5] == ["SUCCEEDED", "1002", "COMPLETED", "0:0"]
    assert rows["setup"][1:4] == ["SUCCEEDED", "null", "-"] or rows["setup"][3] == "-"
    assert "train" not in rows


def test_status_usage(h):
    assert h.run("status.sh").returncode == 2
    assert h.run("status.sh", h.root / "nope").returncode == 2
    (h.root / "plain").mkdir()
    assert h.run("status.sh", h.root / "plain").returncode == 2


# ------------------------------------------------------------------ results.sh
def test_results(h):
    run = mkrun(h)
    (run / "boosted.summary.json").write_text(json.dumps(
        {"num_rows": 10, "accept_rate": 0.5, "mean_rounds": 1.5, "status": {"accepted": 5},
         "mean_logprob_gain_vs_expert": None}))
    td = run / "sft"
    (run / "status" / "train.dir").write_text(f"{td}\n")
    (td / "train_result.json").write_text(json.dumps({"final_step": 7, "elapsed_s": 3.0}))
    (td / "train_history.jsonl").write_text('{"step": 7, "loss": null}\n')
    (run / "eval" / "m-final-1" / "eval_summary.json").write_text(
        json.dumps({"accuracy": {"math500_test": 0.25}}))
    mark(run, "train", "SUCCEEDED", 1)
    r = h.run("results.sh", run)
    assert r.returncode == 0, r.stderr
    assert "rows=10" in r.stdout and "final_step=7" in r.stdout
    assert "math500_test=0.25" in r.stdout and "[train] state=SUCCEEDED" in r.stdout


def test_results_empty_run(h):
    run = mkrun(h)
    r = h.run("results.sh", run)
    assert r.returncode == 0 and "[boost] state=-" in r.stdout


# ------------------------------------------------------------------ logs.sh
def test_logs(h):
    run = mkrun(h)
    old = run / "logs" / "boost-20261018T000000Z-1.out"
    old.write_text("old\n")
    os.utime(old, (time.time() - 100, time.time() - 100))
    new = run / "logs" / "boost-20261019T000000Z-2.out"
    new.write_text("".join(f"line{i}\n" for i in range(300)))
    (run / "logs" / "boost-20261019T000000Z-2.err").write_text("errline\n")
    r = h.run("logs.sh", run, "boost", "5")
    assert r.returncode == 0, r.stderr
    assert "line299" in r.stdout and "line294" not in r.stdout and "errline" in r.stdout
    assert "old" not in r.stdout
    assert h.run("logs.sh", run, "boost", "x").returncode == 2
    assert h.run("logs.sh", run, "eval").returncode == 1
    assert h.run("logs.sh").returncode == 2


# ------------------------------------------------------------------ remote.sh
def remote(h, *args, env=None):
    return h.run("remote.sh", *args, env=env)


@pytest.mark.parametrize("args,msg", [
    (["bogus"], "unknown command"),
    (["status", "--remote-dir", "d", "r"], "--host is required"),
    (["status", "--host", "h", "r"], "--remote-dir is required"),
    (["status", "--host", "h", "--remote-dir", "it's", "r"], "quotes"),
    (["status", "--host", "h", "--remote-dir", "a\nb", "r"], "quotes"),
    (["sync", "--host", "h", "--remote-dir", "my proj", "r"], "whitespace"),
    (["sync", "--host", "h", "--remote-dir", "~/a\tb"], "whitespace"),
    (["status", "--host", "h", "--remote-dir", "/p/x y", "r"], "whitespace"),
    (["status", "--host", "h", "--remote-dir", "d"], "needs RUN_DIR"),
    (["status", "--host", "h", "--remote-dir", "d", "a", "b"], "needs RUN_DIR"),
    (["tail", "--host", "h", "--remote-dir", "d"], "LOG_PATH"),
    (["tail", "--host", "h", "--remote-dir", "d", "f", "1x"], "integer"),
    (["submit", "--host", "h", "--remote-dir", "d"], "submit needs"),
    (["sync", "--host", "h", "--remote-dir", "d", "--bogus"], "unknown option"),
    (["sync", "--host", "h", "--remote-dir", "/"], "top-level"),
    (["sync", "--host", "h", "--remote-dir", "~"], "top-level"),
    (["sync", "--host", "h", "--remote-dir", "~/"], "top-level"),
    (["sync", "--host", "h", "--remote-dir", "/scratch/"], "top-level"),
])
def test_remote_validation(h, args, msg):
    r = remote(h, *args)
    assert r.returncode == 2, r.stderr
    assert msg in r.stderr
    assert h.log("ssh.log") == [] and h.log("rsync.log") == []


def test_remote_help(h):
    r = remote(h, "--help")
    assert r.returncode == 0 and "remote.sh sync" in r.stdout


def _remote_home(h):
    home = h.root / "remotehome"
    home.mkdir()
    return home


def _stub_submit(d):
    (d / "slurm").mkdir(parents=True, exist_ok=True)
    stub = d / "slurm" / "submit.sh"
    stub.write_text('#!/bin/bash\npython3 -c \'import json,sys; print("ARGV=" + '
                    'json.dumps(sys.argv[1:]))\' "$@"\nexit "${STUB_RC:-0}"\n')
    stub.chmod(0o755)
    for name in ("status.sh", "results.sh"):
        (d / "slurm" / name).write_text(stub.read_text())
        (d / "slurm" / name).chmod(0o755)


def test_remote_submit_quoting_and_exit_code(h):
    home = _remote_home(h)
    rdir = home / "my-proj"
    env = {"FAKE_SSH_EXEC": "1", "FAKE_REMOTE_HOME": home, "STUB_RC": "10"}
    weird = ["pipeline", "--account", "def-x", "--run-dir", "runs/a b", "--",
             "--set", "x=$HOME;rm -rf /", "it's", "", "*", "a\nb"]
    # first sync creates the dir + marker, then the stub submit.sh is installed
    r = remote(h, "sync", "--host", "cl", "--remote-dir", str(rdir), env=env)
    assert r.returncode == 0, r.stderr
    assert (rdir / ".vsft-checkout").is_file()
    _stub_submit(rdir)
    r = remote(h, "submit", "--host", "cl", "--remote-dir", str(rdir), "--", *weird, env=env)
    assert r.returncode == 10, r.stderr
    argv = json.loads(r.stdout.split("ARGV=", 1)[1].splitlines()[0])
    assert argv == weird
    for c in h.log("ssh.log"):
        assert c["host"] == "cl" and "BatchMode=yes" in c["opts"]


def test_remote_status_tail_home_relative(h):
    home = _remote_home(h)
    rdir = home / "vsft"
    rdir.mkdir()
    (rdir / ".vsft-checkout").write_text("")
    _stub_submit(rdir)
    (rdir / "log with space.out").write_text("".join(f"l{i}\n" for i in range(10)))
    env = {"FAKE_SSH_EXEC": "1", "FAKE_REMOTE_HOME": home}
    r = remote(h, "status", "--host", "cl", "--remote-dir", "~/vsft", "runs/x y", env=env)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.split("ARGV=", 1)[1]) == ["runs/x y"]
    r = remote(h, "tail", "--host", "cl", "--remote-dir", "~/vsft", "log with space.out", "3",
               env=env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["l7", "l8", "l9"]


def test_remote_sync_refusals(h):
    home = _remote_home(h)
    (home / "dotfile").write_text("x")
    other = home / "other"
    other.mkdir()
    (other / "precious").write_text("x")
    env = {"FAKE_SSH_EXEC": "1", "FAKE_REMOTE_HOME": home}
    r = remote(h, "sync", "--host", "cl", "--remote-dir", "~/other", env=env)
    assert r.returncode == 2 and "not a vsft checkout" in r.stderr
    r = remote(h, "sync", "--host", "cl", "--remote-dir", str(home), env=env)
    assert r.returncode == 2 and "home" in r.stderr
    r = remote(h, "sync", "--host", "cl", "--remote-dir", "~/.", env=env)
    assert r.returncode == 2
    assert h.log("rsync.log") == []
    assert (other / "precious").exists() and not (other / ".vsft-checkout").exists()


def test_remote_sync_rsync_argv(h):
    home = _remote_home(h)
    env = {"FAKE_SSH_EXEC": "1", "FAKE_REMOTE_HOME": home}
    r = remote(h, "sync", "--host", "cl", "--remote-dir", "~/vsft", env=env)
    assert r.returncode == 0, r.stderr
    calls = [c["argv"] for c in h.log("rsync.log")]
    local = str(h.project)
    trees = {}
    tops = None
    for a in calls:
        src, dst = a[-2], a[-1]
        if src.endswith("/") and not src.endswith(f"{local}/"):
            trees[src[len(local) + 1:-1]] = (a, dst)
        else:
            tops = a
    assert set(trees) <= {"src", "slurm", "configs", "tests"}
    assert {"src", "slurm", "configs"} <= set(trees)
    for t, (a, dst) in trees.items():
        assert dst == f"cl:~/vsft/{t}/", dst
        assert "--delete-after" in a or "--delete" in a
    assert tops is not None and not any(x.startswith("--delete") for x in tops)
    assert tops[-1] == "cl:~/vsft/"
    assert {os.path.basename(x) for x in tops[1:-1] if not x.startswith("-")} >= {
        "pyproject.toml", "uv.lock"}
    for a in calls:  # never sync the whole checkout root with --delete
        assert not (a[-2] == f"{local}/" and any(x.startswith("--delete") for x in a))


def test_remote_ssh_failure_reported(h):
    r = remote(h, "sync", "--host", "cl", "--remote-dir", "d", env={"FAKE_SSH_RC": "255"})
    assert r.returncode == 2 and "ssh to cl failed" in r.stderr
    assert h.log("rsync.log") == []


# ------------------------------------------------------------------ fetch_online.sh
def fetch(h, *args, env=None):
    return h.run("fetch_online.sh", *args, env=env, timeout=120)


@pytest.mark.parametrize("args,msg", [
    ([], "--i-confirm-site-policy-allows-this"),
    (["--run-dir", "r"], "--i-confirm-site-policy-allows-this"),
    (["--i-confirm-site-policy-allows-this"], "--run-dir or --resume"),
    (["--i-confirm-site-policy-allows-this", "--run-dir", "a", "--resume", "b"], "exclusive"),
    (["--i-confirm-site-policy-allows-this", "--run-dir", "a", "--datasets", "aime"],
     "unknown dataset"),
    (["--i-confirm-site-policy-allows-this", "--bogus"], "unknown argument"),
    (["--i-confirm-site-policy-allows-this", "--resume", "/nonexistent"], "not a run dir"),
])
def test_fetch_validation(h, args, msg):
    r = fetch(h, *args)
    assert r.returncode == 2, r.stderr
    assert msg in r.stderr


def test_fetch_requires_uv(h):
    (h.bin / "uv").unlink()
    r = fetch(h, "--i-confirm-site-policy-allows-this", "--run-dir", h.runs / "f")
    assert r.returncode == 2 and "uv not on PATH" in r.stderr


def _fake_tools(h, uv_rc=1, prepare_ok=True):
    """Fake uv + a fake venv python, so nothing real is synced or downloaded."""
    (h.bin / "uv").unlink()
    calls = h.root / "calls.log"
    (h.bin / "uv").write_text(f'#!/bin/sh\necho "uv $*" >> {calls}\nexit {uv_rc}\n')
    (h.bin / "uv").chmod(0o755)
    venv = h.root / "fakevenv"
    (venv / "bin").mkdir(parents=True)
    write = 'echo x >' if prepare_ok else ': >'
    out_line = ('  for a in "$@"; do if [ "$prev" = --out-dir ]; then mkdir -p "$a"; '
                f'{write} "$a/math_test.jsonl"; fi; prev="$a"; done\n')
    (venv / "bin" / "python").write_text(
        "#!/bin/bash\n"
        f'echo "python $*" >> {calls}\n'
        'if [ "$2" = vsft.hpc ] && [ "$3" = stamp-lock ]; then '
        f'echo stamped > {venv}/.vsft-uv-lock.sha256; fi\n'
        'if [ "$2" = vsft.cli ] && [ "$3" = prepare ]; then\n' + out_line + "fi\nexit 0\n")
    (venv / "bin" / "python").chmod(0o755)
    return venv, calls


def _wait_log(run):
    time.sleep(1.0)  # tee runs asynchronously after `exec > >(tee)`
    return "".join(p.read_text() for p in (run / "logs").glob("fetch-*.log"))


def test_fetch_failing_uv_marks_setup_failed(h):
    venv, calls = _fake_tools(h, uv_rc=1)
    r = fetch(h, "--i-confirm-site-policy-allows-this", "--run-dir", h.runs / "f",
              "--venv", venv, "--cache-root", h.root / "cache")
    assert r.returncode != 0, r.stdout + r.stderr
    run = h.runs / "f"
    st = json.loads((run / "status" / "setup.state").read_text())
    assert st["state"] == "FAILED" and st["exit_code"] == 1
    assert not (run / "status" / "setup.SUCCEEDED").exists()
    assert not (run / "status" / "prepare.state").exists()
    assert not (venv / ".vsft-uv-lock.sha256").exists()
    assert "uv sync" in calls.read_text() and "python" not in calls.read_text()
    assert "FAILED during setup" in _wait_log(run) + r.stderr + r.stdout


def test_fetch_empty_dataset_marks_prepare_failed(h):
    venv, calls = _fake_tools(h, uv_rc=0, prepare_ok=False)
    r = fetch(h, "--i-confirm-site-policy-allows-this", "--run-dir", h.runs / "f",
              "--venv", venv, "--cache-root", h.root / "cache")
    assert r.returncode != 0
    run = h.runs / "f"
    assert json.loads((run / "status" / "setup.state").read_text())["state"] == "SUCCEEDED"
    assert json.loads((run / "status" / "prepare.state").read_text())["state"] == "FAILED"


def test_fetch_success_with_fakes(h):
    venv, calls = _fake_tools(h, uv_rc=0, prepare_ok=True)
    r = fetch(h, "--i-confirm-site-policy-allows-this", "--run-dir", h.runs / "f",
              "--venv", venv, "--cache-root", h.root / "cache", "--datasets", "math")
    assert r.returncode == 0, r.stdout + r.stderr
    run = h.runs / "f"
    for s in ("setup", "prepare"):
        st = json.loads((run / "status" / f"{s}.state").read_text())
        assert st["state"] == "SUCCEEDED" and st["via"] == "fetch_online"
    text = calls.read_text()
    assert f"uv sync --project {run}/code --frozen --extra vllm" in text
    assert "python -m vsft.cli prepare math --out-dir" in text
    assert (run / "code" / "uv.lock").is_file()
    log = _wait_log(run)
    assert "Done. Now run the GPU stages offline" in log + r.stdout
    # the printed follow-up command must be valid for submit.sh
    tail = (log + r.stdout).split("Done. Now run the GPU stages offline:", 1)[1]
    cmd = shlex.split(tail.replace("\\\n", " ").strip().splitlines()[0])
    assert cmd[0] == "slurm/submit.sh" and "--resume" in cmd
