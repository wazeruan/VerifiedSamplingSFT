"""Items 7-8: the bounded startup monitor and in-job behaviour of job_common.sh, by actually
running stages through the fake scheduler (FAKE_SBATCH_RUN=1)."""

import json
import os
import re
import signal
import subprocess
import time

import pytest
from conftest import BASH

RUN = {"FAKE_SBATCH_RUN": "1"}


def receipt(h, stage, run="x"):
    recs = sorted((h.runs / run / "receipts").glob(f"{stage}-*.json"))
    return h.rjson(recs[-1])


def status(h, stage, state, run="x"):
    p = h.runs / run / "status" / f"{stage}.{state}"
    return h.rjson(p) if p.exists() else None


def logs(h, jid, run="x"):
    out = "".join(p.read_text() for p in sorted((h.runs / run / "logs").glob(f"*-{jid}.*")))
    return out


# a workload that initialises (notify_started), then works for FAKE_SLEEP seconds
STARTED_JOB = r'''
vsft_init
vsft_preflight
vsft_run "$VSFT_PY" -c 'import os, time, vsft
from vsft.hpc import notify_started
print("VSFT_FILE", vsft.__file__, flush=True)
notify_started(probe=1)
time.sleep(float(os.environ.get("FAKE_SLEEP", "30")))'
vsft_succeed "probe=done"
'''


# ------------------------------------------------------------------ 7. monitor outcomes
def test_monitor_started(h):
    h.job_script("gpucheck", STARTED_JOB)
    r = h.submit("gpucheck", "--monitor-timeout", "60", env={**RUN, "FAKE_SLEEP": "12"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OUTCOME: STARTED" in r.stderr
    jid = h.sbatch_calls()[0]["job_id"]
    assert receipt(h, "gpucheck")["monitor"] == "STARTED"
    assert str(status(h, "gpucheck", "STARTED")["job_id"]) == jid
    assert str(status(h, "gpucheck", "PREFLIGHT_OK")["job_id"]) == jid
    j = h.wait_job(jid)
    assert j["state"] == "COMPLETED", logs(h, jid)
    st = status(h, "gpucheck", "state")
    assert st["state"] == "SUCCEEDED" and str(st["job_id"]) == jid
    # the job ran the per-run code snapshot, not the checkout
    m = re.search(r"VSFT_FILE (\S+)", logs(h, jid))
    assert m and m.group(1).startswith(str(h.runs / "x" / "code" / "src")), logs(h, jid)
    for line in ("RUN_DIR=", "JOB_IDS=", "RECEIPT=", "LOG_PATH=", "slurm/status.sh"):
        assert line in r.stdout


def test_monitor_pending_timeout(h):
    t0 = time.time()
    r = h.submit("gpucheck", "--monitor-timeout", "5",
                 env={"FAKE_JOB_TIMELINE": "PENDING|0:0|Priority@0"})
    assert r.returncode == 10, r.stderr
    assert "PENDING" in r.stderr and "Priority" in r.stderr
    rec = receipt(h, "gpucheck")
    assert rec["monitor"] == "PENDING" and rec["pending_reason"] == "Priority"
    assert time.time() - t0 < 40


def test_monitor_pending_via_sacct_when_squeue_down(h):
    r = h.submit("gpucheck", "--monitor-timeout", "5",
                 env={"FAKE_JOB_TIMELINE": "PENDING|0:0|Resources@0", "FAKE_SQUEUE_FAIL": "1"})
    assert r.returncode == 10, r.stderr
    assert receipt(h, "gpucheck")["pending_reason"] == "Resources"


def test_monitor_failed_before_started_with_log_tail(h):
    r = h.submit("gpucheck", env={"FAKE_JOB_TIMELINE": "FAILED|1:0|NonZeroExitCode@0",
                                  "FAKE_JOB_LOG_TEXT": "Traceback: CUDA driver exploded"})
    assert r.returncode == 1, r.stderr
    assert "reached FAILED before STARTED" in r.stderr
    assert "----- tail" in r.stderr and "CUDA driver exploded" in r.stderr
    rec = receipt(h, "gpucheck")
    assert rec["monitor"] == "FAILED" and rec["scheduler_state"].startswith("FAILED|1:0")


def test_monitor_completed_without_success_record_is_failed(h):
    r = h.submit("gpucheck", env={"FAKE_JOB_TIMELINE": "COMPLETED|0:0|None@0"})
    assert r.returncode == 1, r.stderr
    assert "OUTCOME: FAILED" in r.stderr


@pytest.mark.parametrize("show_done", ["0", "1"])
def test_monitor_short_job_succeeded_between_polls(h, show_done):
    h.job_script("gpucheck", "vsft_init\nvsft_succeed quick=1\n")
    r = h.submit("gpucheck", env={**RUN, "FAKE_PENDING_SECS": "2",
                                  "FAKE_SQUEUE_SHOW_DONE": show_done})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "COMPLETED|0:0 and recorded SUCCEEDED" in r.stderr
    assert receipt(h, "gpucheck")["monitor"] == "COMPLETED"


def test_monitor_started_then_failed(h):
    h.job_script("gpucheck", r'''
vsft_init
vsft_run "$VSFT_PY" -c 'from vsft.hpc import notify_started; notify_started(); raise SystemExit(1)'
vsft_succeed
''')
    r = h.submit("gpucheck", env={**RUN, "FAKE_PENDING_SECS": "3"})
    assert r.returncode == 1, r.stdout + r.stderr
    assert "started, then ended FAILED" in r.stderr
    assert status(h, "gpucheck", "state")["state"] == "FAILED"


@pytest.mark.parametrize("squeue_gone", ["error", "empty"])
def test_monitor_gone_then_visible(h, squeue_gone):
    # real squeue answers "Invalid job id specified" (exit 1) for a job it no longer knows
    h.job_script("gpucheck", "vsft_init\nvsft_succeed\n")
    r = h.submit("gpucheck", "--monitor-timeout", "5",
                 env={**RUN, "FAKE_GONE_SECS": "15", "FAKE_SQUEUE_GONE": squeue_gone})
    assert r.returncode == 0, r.stdout + r.stderr


def test_monitor_gone_forever_is_unknown(h):
    t0 = time.time()
    r = h.submit("gpucheck", "--monitor-timeout", "0", env={"FAKE_JOB_TIMELINE": "GONE@0"},
                 timeout=300)
    took = time.time() - t0
    assert r.returncode == 3, r.stderr
    assert "UNKNOWN/UNVERIFIED" in r.stderr
    assert receipt(h, "gpucheck")["monitor"] == "UNVERIFIED"
    assert 110 < took < 200, took  # waited the ~120 s accounting grace, then gave up


def test_monitor_scheduler_down_timeout(h):
    r = h.submit("gpucheck", "--monitor-timeout", "5",
                 env={"FAKE_SQUEUE_FAIL": "1", "FAKE_SACCT_FAIL": "1"})
    assert r.returncode == 3, r.stderr
    assert receipt(h, "gpucheck")["monitor"] in ("UNVERIFIED", "UNKNOWN")


def test_monitor_scheduler_down_repeatedly(h):
    r = h.submit("gpucheck", env={"FAKE_SQUEUE_FAIL": "1", "FAKE_SACCT_FAIL": "1"}, timeout=300)
    assert r.returncode == 3, r.stderr
    assert "scheduler queries keep failing" in r.stderr and "Do NOT resubmit" in r.stderr
    assert receipt(h, "gpucheck")["monitor"] == "UNKNOWN"


# ------------------------------------------------------------------ 8. real stages
def test_real_gpucheck_fails_preflight_on_mac(h):
    r = h.submit("gpucheck", env=RUN)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "before STARTED" in r.stderr and "PREFLIGHT FAILED" in r.stderr
    assert "no CUDA device" in r.stderr
    jid = h.sbatch_calls()[0]["job_id"]
    j = h.wait_job(jid)
    assert j["state"] == "FAILED" and j["rc"] == 2
    failed = status(h, "gpucheck", "FAILED")
    assert str(failed["job_id"]) == jid and failed["reason"] == "exit" and failed["exit_code"] == 2
    assert status(h, "gpucheck", "STARTED") is None
    assert status(h, "gpucheck", "PREFLIGHT_OK") is None
    assert receipt(h, "gpucheck")["state"] == "FAILED"


def test_real_boost_fails_preflight_with_fake_nvidia_smi(h):
    smi = h.bin / "nvidia-smi"
    smi.write_text("#!/bin/sh\necho 'index, name, memory.total [MiB], driver_version'\n"
                   "echo '0, NVIDIA H100 80GB HBM3, 81920 MiB, 535.00'\n")
    smi.chmod(0o755)
    r = h.submit("boost", "--input", h.input, env=RUN)
    assert r.returncode == 1, r.stdout + r.stderr
    jid = h.sbatch_calls()[0]["job_id"]
    h.wait_job(jid)
    text = logs(h, jid)
    assert "NVIDIA H100 80GB HBM3" in text  # nvidia-smi ran inside the job
    assert "no CUDA device" in text and "not in the HF cache" in text
    assert status(h, "boost", "FAILED")["exit_code"] == 2
    assert status(h, "boost", "STARTED") is None


@pytest.mark.parametrize("stage", ["prepare", "setup"])
def test_network_stages_exit_69_without_internet(h, stage):
    (h.bin / "uv").unlink()
    fake_uv = h.bin / "uv"
    fake_uv.write_text(f"#!/bin/sh\necho \"$@\" >> {h.root}/uv.calls\nexit 1\n")
    fake_uv.chmod(0o755)
    dead = "http://127.0.0.1:9"
    r = h.submit(stage, env={**RUN, "https_proxy": dead, "HTTPS_PROXY": dead,
                             "http_proxy": dead, "no_proxy": "", "NO_PROXY": ""})
    assert r.returncode == 1, r.stdout + r.stderr
    jid = h.sbatch_calls()[0]["job_id"]
    j = h.wait_job(jid)
    assert j["rc"] == 69
    assert "no internet access" in logs(h, jid)
    assert status(h, stage, "FAILED")["exit_code"] == 69
    assert status(h, stage, "STARTED") is None
    assert not (h.root / "uv.calls").exists()


def _wait_started(h, stage="gpucheck"):
    """STARTED is marked before vsft_run launches the child; wait for the child too."""
    st = h.wait_for(lambda: status(h, stage, "STARTED"), 60, f"{stage}.STARTED")
    jid = str(st["job_id"])
    h.wait_for(lambda: "exec:" in logs(h, jid), 30, "workload launch")
    time.sleep(0.5)
    return st


SLEEP_JOB = '''
vsft_init
vsft_mark_started_cpu
vsft_run sleep 60
vsft_succeed
'''


@pytest.mark.parametrize("sig", ["TERM", "USR1", "INT"])
def test_signal_marks_failed(h, sig):
    h.job_script("gpucheck", SLEEP_JOB)
    r = h.submit("gpucheck", "--no-monitor", "--preempt-grace", "2", env=RUN)
    assert r.returncode == 3, r.stderr
    jid = h.sbatch_calls()[0]["job_id"]
    _wait_started(h)
    t0 = time.time()
    os.kill(h.job(jid)["pid"], getattr(signal, f"SIG{sig}"))
    j = h.wait_job(jid, timeout=40)
    assert time.time() - t0 < 20, "workload was not stopped after the grace period"
    assert j["state"] == "FAILED" and j["rc"] == 143, j
    f = status(h, "gpucheck", "FAILED")
    assert f["reason"] == "signal" and f["exit_code"] == 143 and f["signal"] == sig
    assert str(f["job_id"]) == jid
    assert status(h, "gpucheck", "state")["state"] == "FAILED"
    assert (h.runs / "x" / "status" / f"gpucheck.{jid}.STOP").exists()


STOPPING_JOB = r'''
vsft_init
vsft_mark_started_cpu
vsft_run "$VSFT_PY" -c 'import json, os, time
while not os.path.exists(os.environ["VSFT_STOP_FILE"]):
    time.sleep(0.1)
if os.environ.get("FAKE_WRITE_RECORD") == "1":
    with open(os.environ["VSFT_STOPPED_RECORD"], "w") as f:
        json.dump({"job_id": os.environ["SLURM_JOB_ID"], "state_dir": "x"}, f)
raise SystemExit(75)'
vsft_succeed
'''


@pytest.mark.parametrize("record,rc,reason", [("1", 75, "stopped_resumable"),
                                              ("0", 143, "signal")])
def test_graceful_stop(h, record, rc, reason):
    h.job_script("gpucheck", STOPPING_JOB)
    r = h.submit("gpucheck", "--no-monitor", "--preempt-grace", "8",
                 env={**RUN, "FAKE_WRITE_RECORD": record})
    assert r.returncode == 3
    jid = h.sbatch_calls()[0]["job_id"]
    _wait_started(h)
    os.kill(h.job(jid)["pid"], signal.SIGUSR1)
    j = h.wait_job(jid, timeout=40)
    assert j["rc"] == rc, (j, logs(h, jid))
    f = status(h, "gpucheck", "FAILED")
    assert f["reason"] == reason and f["exit_code"] == rc


def test_stale_stop_record_from_other_job_ignored(h):
    h.job_script("gpucheck", STOPPING_JOB)
    r = h.submit("gpucheck", "--no-monitor", "--preempt-grace", "8", env=RUN)
    jid = h.sbatch_calls()[0]["job_id"]
    _wait_started(h)
    rec = h.runs / "x" / "status" / f"gpucheck.{jid}.STOPPED.json"
    rec.write_text(json.dumps({"job_id": "1"}))  # right path, wrong job id
    os.kill(h.job(jid)["pid"], signal.SIGUSR1)
    j = h.wait_job(jid, timeout=40)
    assert r.returncode == 3 and j["rc"] == 143
    assert status(h, "gpucheck", "FAILED")["reason"] == "signal"


def test_script_without_succeed_is_failed(h):
    h.job_script("gpucheck", "vsft_init\necho done-without-success\n")
    r = h.submit("gpucheck", env=RUN)
    assert r.returncode == 1, r.stdout + r.stderr
    jid = h.sbatch_calls()[0]["job_id"]
    j = h.wait_job(jid)
    assert j["state"] == "FAILED" and j["rc"] == 70
    f = status(h, "gpucheck", "FAILED")
    assert f["exit_code"] == 70 and f["reason"] == "exit"


def test_stage_args_reach_job_intact(h):
    h.job_script("gpucheck", '''
vsft_init
for a in ${VSFT_STAGE_ARGS[@]+"${VSFT_STAGE_ARGS[@]}"}; do printf 'ARG<%s>\\n' "$a"; done
printf 'NARGS=%s\\n' "${#VSFT_STAGE_ARGS[@]}"
vsft_succeed
''')
    args = ["--limit", "3", "with space", "", "a\nb", "$HOME;x", "--x=*"]
    r = h.submit("gpucheck", "--no-monitor", "--", *args, env=RUN)
    assert r.returncode == 3
    jid = h.sbatch_calls()[0]["job_id"]
    j = h.wait_job(jid)
    text = logs(h, jid)
    assert j["rc"] == 0, text
    assert "NARGS=7" in text
    got = re.findall(r"ARG<(.*?)>\n", text, flags=re.S)
    assert got == args


def test_no_extra_args_ok_on_bash32(h):
    h.job_script("gpucheck", '''
vsft_init
printf 'NARGS=%s\\n' "${#VSFT_STAGE_ARGS[@]}"
vsft_succeed
''')
    r = h.submit("gpucheck", "--no-monitor", env=RUN)
    jid = h.sbatch_calls()[0]["job_id"]
    j = h.wait_job(jid)
    assert r.returncode == 3 and j["rc"] == 0 and "NARGS=0" in logs(h, jid)


def test_eval_job_refuses_without_success_record(h):
    run = h.runs / "x"
    r = h.submit("gpucheck", "--no-monitor")  # create the run
    td = run / "sft"
    (td / "final").mkdir(parents=True)
    (run / "status" / "train.dir").write_text(f"{td}\n")
    (td / "SUCCEEDED.json").write_text('{"job_id": "1"}')
    h.make_test_sets(run)
    r = h.submit("eval", "--no-monitor", "--resume", run, run_dir=None,
                 env={**RUN, "FAKE_PENDING_SECS": "4"})
    assert r.returncode == 3, r.stderr
    (td / "SUCCEEDED.json").unlink()  # removed while the job is queued
    jid = h.sbatch_calls()[-1]["job_id"]
    j = h.wait_job(jid)
    assert j["rc"] == 64
    assert "no SUCCEEDED.json" in logs(h, jid)


def test_job_outside_slurm_refuses(h):
    r = h.submit("gpucheck", "--no-monitor")
    env = dict(h.env)
    env.update(h.sbatch_calls()[0]["env"])
    p = subprocess.run([BASH, h.sbatch_calls()[0]["script"]], env=env, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 3 and p.returncode == 64
    assert "not inside a Slurm allocation" in p.stdout


# ------------------------------------------------------------------ preemption grace
def _grace(h, stage, override=None):
    run = h.root / "g"
    (run / "status").mkdir(parents=True, exist_ok=True)
    env = dict(h.env, VSFT_PROJECT_DIR=str(h.project), VSFT_CODE_DIR=str(h.project),
               VSFT_RUN_DIR=str(run), VSFT_STAGE=stage, VSFT_RECEIPT=str(run / "r.json"),
               VSFT_VENV=str(h.venv))
    env.pop("VSFT_PREEMPT_GRACE", None)
    if override is not None:
        env["VSFT_PREEMPT_GRACE"] = str(override)
    code = ('source "$VSFT_CODE_DIR/slurm/lib/job_common.sh"; VSFT_DONE=1; '
            'echo "$(_signal_lead) $(_default_grace)"')
    p = subprocess.run([BASH, "-c", code], env=env, capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr
    lead, grace = map(int, p.stdout.split())
    return lead, grace


@pytest.mark.parametrize("stage,lead,grace", [
    ("boost", 600, 500), ("train", 600, 500), ("eval", 300, 240),
    ("setup", 120, 60), ("prepare", 120, 60), ("gpucheck", 60, 10),
    ("train_sft", 600, 500), ("eval_sft", 300, 240), ("eval_base", 300, 240),
])
def test_default_grace(h, stage, lead, grace):
    assert _grace(h, stage) == (lead, grace)
    lead_, g = _grace(h, stage, 500)
    assert g == min(500, max(lead - 60, 10)) and g < lead
    assert _grace(h, stage, 5)[1] == 5


def test_eval_grace_capped(h):
    assert _grace(h, "eval", 500) == (300, 240)
    assert _grace(h, "boost", 539) == (600, 539)


def test_signal_lead_matches_sbatch_directives(h):
    script = {"train_sft": "train", "eval_sft": "eval", "eval_base": "eval"}
    for stage in ("setup", "prepare", "boost", "train", "eval", "gpucheck", "train_sft",
                  "eval_sft", "eval_base"):
        text = (h.project / "slurm" / "jobs" / f"{script.get(stage, stage)}.sbatch").read_text()
        m = re.search(r"#SBATCH --signal=B:USR1@(\d+)", text)
        lead, grace = _grace(h, stage)
        if m:
            assert int(m.group(1)) == lead, stage
            assert grace <= lead - 60, stage
