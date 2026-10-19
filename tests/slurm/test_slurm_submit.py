"""Items 3-6: run-dir claiming, pipeline chaining, dependency reconciliation, rejection,
ambiguous sbatch output, duplicate guard, --resume, receipts and args files."""

import json
import re

from conftest import TS_RE


def _opt(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


# ------------------------------------------------------------------ 3. claiming
def test_fresh_run_claiming(h):
    r1 = h.submit("gpucheck", "--no-monitor")
    assert r1.returncode == 3, r1.stderr  # --no-monitor -> UNVERIFIED
    x = h.runs / "x"
    manifest1 = (x / "manifest.json").read_text()
    receipts1 = sorted((x / "receipts").iterdir())
    r2 = h.submit("gpucheck", "--no-monitor", "--allow-duplicate")
    assert r2.returncode == 3, r2.stderr
    x1 = h.runs / "x-001"
    assert x1.is_dir() and (x1 / "manifest.json").is_file()
    # the first run dir was not reused or touched
    assert (x / "manifest.json").read_text() == manifest1
    assert sorted((x / "receipts").iterdir()) == receipts1
    calls = h.sbatch_calls()
    assert [_opt(c["argv"], "--chdir") for c in calls] == [str(x), str(x1)]


def test_existing_dir_is_never_reused(h):
    (h.runs / "x").mkdir(parents=True)
    (h.runs / "x" / "keep.txt").write_text("user data")
    r = h.submit("gpucheck", "--no-monitor")
    assert r.returncode == 3, r.stderr
    assert sorted(p.name for p in (h.runs / "x").iterdir()) == ["keep.txt"]
    assert (h.runs / "x-001" / "manifest.json").is_file()


def test_code_snapshot(h):
    r = h.submit("gpucheck", "--no-monitor")
    assert r.returncode == 3, r.stderr
    run = h.runs / "x"
    code = run / "code"
    for p in ("src/vsft/hpc.py", "slurm/lib/job_common.sh", "slurm/jobs/gpucheck.sbatch",
              "pyproject.toml", "uv.lock"):
        assert (code / p).is_file(), p
    assert not list(code.rglob("__pycache__"))
    assert (run / "config.yaml").read_bytes() == (
        h.project / "configs/a100/math_qwen3b.yaml").read_bytes()
    call = h.sbatch_calls()[0]
    assert call["script"] == str(code / "slurm/jobs/gpucheck.sbatch")
    assert _opt(call["argv"], "--chdir") == str(run)
    m = h.rjson(run / "manifest.json")
    assert m["code_dir"] == str(code) and m["config"] == str(run / "config.yaml")


# ------------------------------------------------------------------ 4. pipeline
def test_pipeline_chaining_and_records(h):
    r = h.submit("pipeline", "--no-monitor")
    assert r.returncode == 3, r.stderr
    run = h.runs / "x"
    calls = h.sbatch_calls()
    stages = ["setup", "prepare", "boost", "train", "eval"]
    assert [c["env"]["VSFT_STAGE"] for c in calls] == stages
    ids = [c["job_id"] for c in calls]
    assert len(set(ids)) == 5
    for i, c in enumerate(calls):
        assert _opt(c["argv"], "--account") == "def-test"
        if i == 0:
            assert "--dependency" not in c["argv"]
        else:
            assert _opt(c["argv"], "--dependency") == f"afterok:{ids[i - 1]}"
            assert "--kill-on-invalid-dep=yes" in c["argv"]
        assert c["env"]["VSFT_OFFLINE"] == "1"  # GPU stages offline by default
    m = h.rjson(run / "manifest.json")
    expected = [[s, i] for s, i in zip(stages, ids, strict=True)]
    assert [j.split(":")[:2] for j in m["jobs"]] == expected
    assert (run / "config.yaml").is_file()
    for s, jid in zip(stages, ids, strict=True):
        recs = list((run / "receipts").glob(f"{s}-*.json"))
        assert len(recs) == 1
        assert re.fullmatch(rf"{s}-{TS_RE}-[0-9a-f]{{8}}\.json", recs[0].name)
        rec = h.rjson(recs[0])
        assert str(rec["job_id"]) == jid and rec["state"] == "SUBMITTED"
        assert rec["env"]["VSFT_STAGE"] == s and rec["env"]["VSFT_RUN_DIR"] == str(run)
        assert rec["log_out"].endswith(f"-{jid}.out")
        sub = h.rjson(run / "status" / f"{s}.SUBMITTED")
        assert str(sub["job_id"]) == jid
        st = h.rjson(run / "status" / f"{s}.state")
        assert st["state"] == "SUBMITTED" and str(st["job_id"]) == jid


def test_online_flag_exported(h):
    r = h.submit("gpucheck", "--no-monitor", "--online")
    assert r.returncode == 3
    assert h.sbatch_calls()[0]["env"]["VSFT_OFFLINE"] == "0"


def test_pipeline_rereconciles_failed_predecessor(h):
    # every submitted job fails immediately: the second stage must not be submitted
    r = h.submit("pipeline", "--stages", "boost,train", "--input", h.input, "--no-monitor",
                 env={"FAKE_JOB_TIMELINE": "FAILED|1:0|NonZeroExitCode@0"})
    assert r.returncode == 2, r.stderr
    calls = h.sbatch_calls()
    assert len(calls) == 1
    assert "FAILED" in r.stderr and calls[0]["job_id"] in r.stderr  # "already submitted" note


def test_pipeline_completed_predecessor_dropped(h):
    r = h.submit("pipeline", "--stages", "boost,train", "--input", h.input, "--no-monitor",
                 env={"FAKE_JOB_TIMELINE": "COMPLETED|0:0|None@0"})
    assert r.returncode == 3, r.stderr
    calls = h.sbatch_calls()
    assert len(calls) == 2 and "--dependency" not in calls[1]["argv"]


# ------------------------------------------------------------------ 5. --depend-on
def _deps(h):
    h.add_job(500, "RUNNING|0:0|None@0")
    h.add_job(501, "COMPLETED|0:0|None@0")
    h.add_job(502, "FAILED|1:0|NonZeroExitCode@0")
    h.add_job(503, "GONE@0")
    h.add_job(504, "COMPLETED|2:0|None@0")
    h.add_job(505, "PENDING|0:0|Dependency@0")
    h.add_job(506, "CANCELLED|0:15|None@0")


def test_depend_on_active_kept(h):
    _deps(h)
    r = h.submit("gpucheck", "--no-monitor", "--depend-on", "500:505")
    assert r.returncode == 3, r.stderr
    assert _opt(h.sbatch_calls()[0]["argv"], "--dependency") == "afterok:500:505"


def test_depend_on_completed_omitted(h):
    _deps(h)
    r = h.submit("gpucheck", "--no-monitor", "--depend-on", "500:501")
    assert r.returncode == 3, r.stderr
    assert _opt(h.sbatch_calls()[0]["argv"], "--dependency") == "afterok:500"
    assert "501 already COMPLETED" in r.stderr


def test_depend_on_all_completed_no_dependency(h):
    _deps(h)
    r = h.submit("gpucheck", "--no-monitor", "--depend-on", "501")
    assert r.returncode == 3, r.stderr
    assert "--dependency" not in h.sbatch_calls()[0]["argv"]


def test_depend_on_completed_works_with_squeue_empty_mode(h):
    _deps(h)
    r = h.submit("gpucheck", "--no-monitor", "--depend-on", "501:500",
                 env={"FAKE_SQUEUE_GONE": "empty"})
    assert r.returncode == 3, r.stderr
    assert _opt(h.sbatch_calls()[0]["argv"], "--dependency") == "afterok:500"


def test_depend_on_bad_aborts_before_claim(h):
    _deps(h)
    for dep, msg in (("502", "FAILED"), ("503", "unknown to the scheduler"),
                     ("504", "completed with exit 2:0"), ("500:506", "CANCELLED")):
        r = h.submit("gpucheck", "--no-monitor", "--depend-on", dep)
        assert r.returncode == 2, (dep, r.stderr)
        assert any(m in r.stderr for m in msg.split("|")), (dep, r.stderr)
    assert h.sbatch_calls() == []
    assert not (h.runs / "x").exists(), "run dir must not be claimed when a dependency is bad"


def test_depend_on_scheduler_down_aborts(h):
    _deps(h)
    r = h.submit("gpucheck", "--no-monitor", "--depend-on", "500",
                 env={"FAKE_SQUEUE_FAIL": "1", "FAKE_SACCT_FAIL": "1"})
    assert r.returncode == 2 and "scheduler query failed" in r.stderr
    assert h.sbatch_calls() == [] and not (h.runs / "x").exists()


def test_depend_on_skips_existing_input_check(h):
    _deps(h)
    # boost input is produced by the dependency, so it need not exist yet
    r = h.submit("boost", "--no-monitor", "--depend-on", "500")
    assert r.returncode == 3, r.stderr


# ------------------------------------------------------------------ 6. sbatch responses
def test_rejected_submission(h):
    r = h.submit("gpucheck", "--no-monitor", env={"FAKE_SBATCH_MODE": "reject"})
    assert r.returncode == 4, r.stderr
    assert "SUBMISSION_REJECTED" in r.stderr and "Invalid account" in r.stderr
    run = h.runs / "x"
    rec = h.rjson(next((run / "receipts").glob("gpucheck-*.json")))
    assert rec["state"] == "SUBMISSION_REJECTED" and rec["sbatch_exit"] == 1
    assert "Invalid account" in rec["sbatch_stderr"]
    assert not (run / "status" / "gpucheck.SUBMITTED").exists()
    assert len(h.sbatch_calls()) == 1


def test_rejection_mid_pipeline_reports_submitted(h):
    r = h.submit("pipeline", "--stages", "boost,train", "--input", h.input, "--no-monitor",
                 env={"FAKE_SBATCH_REJECT_ON": "2"})
    assert r.returncode == 4
    first = h.sbatch_calls()[0]["job_id"]
    assert f"already submitted: {first}" in r.stderr


def test_cluster_suffix_parsed(h):
    r = h.submit("gpucheck", "--no-monitor", env={"FAKE_SBATCH_MODE": "cluster"})
    assert r.returncode == 3, r.stderr
    rec = h.rjson(next((h.runs / "x" / "receipts").glob("gpucheck-*.json")))
    assert str(rec["job_id"]) == h.sbatch_calls()[0]["job_id"]


def test_ambiguous_output_reconciled_by_token(h):
    h.add_job(900, "RUNNING|0:0|None@0", comment="vsft-someoneelse")
    r = h.submit("gpucheck", "--no-monitor", env={"FAKE_SBATCH_MODE": "garbage"})
    assert r.returncode == 3, r.stderr
    call = h.sbatch_calls()[0]
    rec = h.rjson(next((h.runs / "x" / "receipts").glob("gpucheck-*.json")))
    assert str(rec["job_id"]) == call["job_id"] != "900"
    assert rec["submission_token"] == _opt(call["argv"], "--comment")
    assert "reconciled job id" in r.stderr
    assert len(h.sbatch_calls()) == 1


def test_ambiguous_output_unresolvable(h):
    for extra in ({}, {"FAKE_SQUEUE_FAIL": "1"}):
        r = h.submit("gpucheck", "--no-monitor", "--allow-duplicate",
                     env={"FAKE_SBATCH_MODE": "hidden", **extra})
        assert r.returncode == 3, r.stderr
        assert "UNKNOWN/UNVERIFIED" in r.stderr and "Do NOT resubmit" in r.stderr
    assert len(h.sbatch_calls()) == 2  # one per invocation, never retried
    for d in ("x", "x-001"):
        rec = h.rjson(next((h.runs / d / "receipts").glob("gpucheck-*.json")))
        assert rec["state"] == "UNKNOWN"


def test_ambiguous_in_pipeline_stops(h):
    r = h.submit("pipeline", "--stages", "boost,train", "--input", h.input, "--no-monitor",
                 env={"FAKE_SBATCH_MODE": "hidden"})
    assert r.returncode == 3
    assert len(h.sbatch_calls()) == 1


# ------------------------------------------------------------------ resume / duplicates
def test_duplicate_guard_and_resume(h):
    r = h.submit("gpucheck", "--no-monitor")
    assert r.returncode == 3
    run = h.runs / "x"
    jid = h.sbatch_calls()[0]["job_id"]
    r = h.submit("gpucheck", "--no-monitor", "--resume", run, run_dir=None)
    assert r.returncode == 2 and "already PENDING" in r.stderr and jid in r.stderr
    assert len(h.sbatch_calls()) == 1
    r = h.submit("gpucheck", "--no-monitor", "--resume", run, "--allow-duplicate", run_dir=None)
    assert r.returncode == 3, r.stderr
    calls = h.sbatch_calls()
    assert len(calls) == 2
    assert calls[1]["script"] == str(run / "code/slurm/jobs/gpucheck.sbatch")
    # once the old jobs are finished, resubmission is allowed again
    for c in calls:
        h.set_timeline(c["job_id"], "FAILED|1:0|x@0")
    r = h.submit("gpucheck", "--no-monitor", "--resume", run, run_dir=None)
    assert r.returncode == 3, r.stderr
    assert len(list((run / "receipts").glob("gpucheck-*.json"))) == 3
    assert not (h.runs / "x-001").exists()


def test_resume_requires_snapshot(h):
    bogus = h.runs / "bogus"
    bogus.mkdir(parents=True)
    (bogus / "manifest.json").write_text("{}")
    (bogus / "config.yaml").write_text("{}")
    r = h.submit("gpucheck", "--resume", bogus, run_dir=None)
    assert r.returncode == 2 and "code snapshot" in r.stderr
    assert h.sbatch_calls() == []


def test_resume_uses_snapshot_not_project(h):
    r = h.submit("gpucheck", "--no-monitor")
    run = h.runs / "x"
    h.set_timeline(h.sbatch_calls()[0]["job_id"], "FAILED|1:0|x@0")
    # later edits to the checkout must not change what a resumed run executes
    (h.project / "slurm" / "jobs" / "gpucheck.sbatch").write_text("#!/bin/bash\nexit 99\n")
    r = h.submit("gpucheck", "--no-monitor", "--resume", run, run_dir=None)
    assert r.returncode == 3, r.stderr
    script = h.sbatch_calls()[1]["script"]
    assert script.startswith(str(run / "code"))
    assert "exit 99" not in open(script).read()


# ------------------------------------------------------------------ args file / env
def test_extra_args_nul_separated(h):
    args = ["--limit", "3", "with space", "", "multi\nline", "--set", "a=$HOME;b"]
    r = h.submit("gpucheck", "--no-monitor", "--", *args)
    assert r.returncode == 3, r.stderr
    call = h.sbatch_calls()[0]
    data = open(call["env"]["VSFT_ARGS_FILE"], "rb").read()
    assert data.split(b"\0")[:-1] == [a.encode() for a in args]
    tag = call["env"]["VSFT_ARGS_FILE"].rsplit("/", 1)[1]
    assert re.fullmatch(rf"gpucheck-{TS_RE}-[0-9a-f]{{8}}\.args", tag)


def test_eval_data_repeatable_and_model_id(h):
    run_ok = h.submit("eval", "--no-monitor", "--eval-model", "Qwen/Qwen2.5-0.5B",
                      "--eval-data", "a b.jsonl", "--eval-data", "/abs/c.jsonl")
    assert run_ok.returncode == 3, run_ok.stderr
    env = h.sbatch_calls()[0]["env"]
    assert env["VSFT_EVAL_MODEL"] == "Qwen/Qwen2.5-0.5B"  # hub id is not made absolute
    assert env["VSFT_EVAL_DATA"].splitlines() == [str(h.project / "a b.jsonl"), "/abs/c.jsonl"]


def test_eval_requires_trained_model(h):
    r = h.submit("eval", "--no-monitor")
    assert r.returncode == 2 and "trained model" in r.stderr
    assert h.sbatch_calls() == []
    run = h.runs / "x"
    td = run / "sft"
    (td / "final").mkdir(parents=True)
    (run / "status" / "train.dir").write_text(f"{td}\n")
    r = h.submit("eval", "--no-monitor", "--resume", run, run_dir=None)
    assert r.returncode == 2 and "trained model" in r.stderr  # no SUCCEEDED.json yet
    (td / "SUCCEEDED.json").write_text('{"job_id": "1"}')
    r = h.submit("eval", "--no-monitor", "--resume", run, run_dir=None)
    assert r.returncode == 3, r.stderr
    assert len(h.sbatch_calls()) == 1


def test_receipts_honour_umask(h):
    r = h.submit("gpucheck", "--no-monitor", env={})
    run = h.runs / "x"
    rec = next((run / "receipts").glob("gpucheck-*.json"))
    import os
    import stat
    mode = stat.S_IMODE(os.stat(rec).st_mode)
    assert mode & 0o044, oct(mode)  # group/other readable under the default umask
    assert json.loads(rec.read_text())["state"] == "SUBMITTED"
    assert r.returncode == 3


def test_depend_on_completed_lingering_in_squeue(h):
    # slurmctld shows a just-finished job as COMPLETED|-|None during MinJobAge
    _deps(h)
    r = h.submit("gpucheck", "--no-monitor", "--depend-on", "501:500",
                 env={"FAKE_SQUEUE_SHOW_DONE": "1"})
    assert r.returncode == 3, r.stderr
    assert _opt(h.sbatch_calls()[0]["argv"], "--dependency") == "afterok:500"
    r = h.submit("gpucheck", "--no-monitor", "--depend-on", "504", run_dir="y",
                 env={"FAKE_SQUEUE_SHOW_DONE": "1"})
    assert r.returncode == 2 and "exit 2:0" in r.stderr


# ------------------------------------------------------------------ venv resolution
def test_default_venv_is_per_lock_and_checked(h):
    import hashlib

    from conftest import make_venv_shim

    r = h.submit("gpucheck", "--no-monitor", venv=False)
    assert r.returncode == 2, r.stderr
    sha = hashlib.sha256((h.project / "uv.lock").read_bytes()).hexdigest()[:12]
    expected = h.project / ".cache" / "venvs" / sha
    assert f"venv missing at {expected}" in r.stderr
    assert h.sbatch_calls() == []
    # a venv at that path stamped with a different lock is refused
    other = h.root / "other.lock"
    other.write_text("different\n")
    make_venv_shim(expected, stamp_lock=other)
    r = h.submit("gpucheck", "--no-monitor", venv=False, run_dir="y")
    assert r.returncode == 2 and "was not built from this run's uv.lock" in r.stderr
    make_venv_shim(expected, stamp_lock=h.project / "uv.lock")
    r = h.submit("gpucheck", "--no-monitor", venv=False, run_dir="z")
    assert r.returncode == 3, r.stderr
    assert h.sbatch_calls()[0]["env"]["VSFT_VENV"] == str(expected)


def test_setup_skips_venv_check(h):
    r = h.submit("setup", "--no-monitor", venv=False)
    assert r.returncode == 3, r.stderr


def test_preempt_grace_bounds(h):
    r = h.submit("gpucheck", "--dry-run", "--preempt-grace", "540")
    assert r.returncode == 2
    r = h.submit("gpucheck", "--no-monitor", "--preempt-grace", "539")
    assert r.returncode == 3, r.stderr
    assert h.sbatch_calls()[0]["env"]["VSFT_PREEMPT_GRACE"] == "539"


def test_accelerate_config_snapshotted(h):
    cfg = h.project / "configs" / "accelerate" / "fsdp_a100.yaml"
    r = h.submit("train", "--no-monitor", "--train-data", h.input, "--accelerate-config", cfg)
    assert r.returncode == 3, r.stderr
    run = h.runs / "x"
    env = h.sbatch_calls()[0]["env"]
    assert env["VSFT_ACCELERATE_CONFIG"] == str(run / "code" / "accelerate.yaml")
    assert (run / "code" / "accelerate.yaml").read_bytes() == cfg.read_bytes()


def test_pipeline_survives_squeue_outage_with_accounting_lag(h):
    # squeue times out and slurmdbd has not recorded the just-submitted job yet: the
    # predecessor we just submitted must not be reported as "unknown to the scheduler".
    r = h.submit("pipeline", "--stages", "boost,train", "--input", h.input, "--no-monitor",
                 env={"FAKE_SQUEUE_FAIL": "1", "FAKE_SACCT_LAG_SECS": "60"})
    calls = h.sbatch_calls()
    assert r.returncode == 3, r.stderr
    assert len(calls) == 2
    assert _opt(calls[1]["argv"], "--dependency") == f"afterok:{calls[0]['job_id']}"


def test_run_dir_with_spaces_end_to_end(h):
    h.job_script("gpucheck", "vsft_init\nvsft_succeed\n")
    r = h.submit("gpucheck", run_dir="my run", env={"FAKE_SBATCH_RUN": "1",
                                                     "FAKE_PENDING_SECS": "2"})
    assert r.returncode == 0, r.stdout + r.stderr
    run = h.runs / "my run"
    assert h.rjson(run / "status" / "gpucheck.state")["state"] == "SUCCEEDED"
    assert list((run / "logs").glob("gpucheck-*.out"))


def test_receipt_env_records_multiline_eval_data(h):
    r = h.submit("eval", "--no-monitor", "--eval-model", "Qwen/Qwen2.5-0.5B",
                 "--eval-data", "/d/a.jsonl", "--eval-data", "/d/b.jsonl")
    assert r.returncode == 3, r.stderr
    rec = h.rjson(next((h.runs / "x" / "receipts").glob("eval-*.json")))
    assert rec["env"]["VSFT_EVAL_DATA"].splitlines() == ["/d/a.jsonl", "/d/b.jsonl"]


def test_resume_accelerate_config_must_match_snapshot(h):
    a = h.root / "a.yaml"
    a.write_text("distributed_type: MULTI_GPU\n")
    b = h.root / "b.yaml"
    b.write_text("distributed_type: FSDP\n")
    r = h.submit("train", "--no-monitor", "--train-data", h.input, "--accelerate-config", a)
    assert r.returncode == 3, r.stderr
    run = h.runs / "x"
    h.set_timeline(h.sbatch_calls()[0]["job_id"], "FAILED|1:0|x@0")
    r = h.submit("train", "--no-monitor", "--train-data", h.input, "--accelerate-config", b,
                 "--resume", run, run_dir=None)
    assert r.returncode == 2 and "differs from this run's snapshot" in r.stderr, r.stderr
    assert len(h.sbatch_calls()) == 1
    assert (run / "code" / "accelerate.yaml").read_text() == a.read_text()
    same = h.root / "a-copy.yaml"
    same.write_text(a.read_text())
    r = h.submit("train", "--no-monitor", "--train-data", h.input, "--accelerate-config", same,
                 "--resume", run, run_dir=None)
    assert r.returncode == 3, r.stderr
    assert h.sbatch_calls()[1]["env"]["VSFT_ACCELERATE_CONFIG"] == str(
        run / "code" / "accelerate.yaml")


def test_unverifiable_foreign_dependency_still_aborts(h):
    # a user-supplied dependency that cannot be verified is never kept blindly
    h.add_job(700, "PENDING|0:0|Priority@0")
    r = h.submit("gpucheck", "--no-monitor", "--depend-on", "700",
                 env={"FAKE_SQUEUE_FAIL": "1", "FAKE_SACCT_LAG_SECS": "60"})
    assert r.returncode == 2, r.stderr
    assert h.sbatch_calls() == [] and not (h.runs / "x").exists()
