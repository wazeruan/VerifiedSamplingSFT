"""compare mode and the train_sft / eval_sft / eval_base stages (graph dependencies, validation,
real jobs through the fake scheduler with an argv-logging fake venv, results.sh table)."""

import hashlib
import json
import re
import shlex
import time

import pytest

GRAPH = {"setup": None, "prepare": "setup", "boost": "prepare", "train": "boost",
         "eval": "train", "train_sft": "boost", "eval_sft": "train_sft",
         "eval_base": "prepare"}
SCRIPT = {"train_sft": "train", "eval_sft": "eval", "eval_base": "eval"}
GPU = {"boost", "train", "eval", "train_sft", "eval_sft", "eval_base"}


def _opt(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def _dry(out):
    res = {}
    for line in out.splitlines():
        m = re.match(r"\[dry-run\] (\w+): (.*)$", line)
        if m:
            res[m.group(1)] = shlex.split(m.group(2))
    return res


# ------------------------------------------------------------------ dry-run graph
def test_compare_dry_run_full_graph(h):
    r = h.submit("compare", "--dry-run")
    assert r.returncode == 0, r.stderr
    lines = _dry(r.stdout)
    assert list(lines) == list(GRAPH)
    run = str(h.runs / "x")
    for stage, argv in lines.items():
        pred = GRAPH[stage]
        dep = _opt(argv, "--dependency")
        assert dep == (f"afterok:DRYRUN_{pred}" if pred else None), (stage, dep)
        assert _opt(argv, "--job-name") == f"vsft-{stage}"
        assert argv[-1] == f"{run}/code/slurm/jobs/{SCRIPT.get(stage, stage)}.sbatch"
        assert re.search(rf"/logs/{stage}-\d{{8}}T\d{{6}}Z-%j\.out$", _opt(argv, "--output"))
        assert ("--gpus-per-node" in argv) == (stage in GPU), stage
        assert _opt(argv, "--account") == "def-test"


def test_compare_dry_run_subset_with_depend_on(h):
    r = h.submit("compare", "--dry-run", "--depend-on", "77:78",
                 "--stages", "boost,train,eval,train_sft,eval_sft,eval_base")
    assert r.returncode == 0, r.stderr
    deps = {s: _opt(a, "--dependency") for s, a in _dry(r.stdout).items()}
    assert deps == {"boost": "afterok:77:78", "train": "afterok:DRYRUN_boost",
                    "eval": "afterok:DRYRUN_train", "train_sft": "afterok:DRYRUN_boost",
                    "eval_sft": "afterok:DRYRUN_train_sft", "eval_base": "afterok:77:78"}


def test_compare_dry_run_sft_only(h):
    r = h.submit("compare", "--dry-run", "--stages", "train_sft,eval_sft,eval_base")
    assert r.returncode == 0, r.stderr
    deps = {s: _opt(a, "--dependency") for s, a in _dry(r.stdout).items()}
    assert deps == {"train_sft": None, "eval_sft": "afterok:DRYRUN_train_sft",
                    "eval_base": None}


@pytest.mark.xfail(strict=True, reason="design: a stage whose direct predecessor is not in "
                   "--stages ignores earlier ancestors that are (boost does not wait for setup)")
def test_compare_dry_run_skipped_predecessor_uses_ancestor(h):
    r = h.submit("compare", "--dry-run", "--stages", "setup,boost")
    assert r.returncode == 0, r.stderr
    assert _opt(_dry(r.stdout)["boost"], "--dependency") == "afterok:DRYRUN_setup"


@pytest.mark.parametrize("stage", ["train_sft", "eval_sft", "eval_base"])
def test_single_new_stage_dry_run(h, stage):
    r = h.submit(stage, "--dry-run")
    assert r.returncode == 0, r.stderr
    argv = _dry(r.stdout)[stage]
    assert argv[-1].endswith(f"/slurm/jobs/{SCRIPT[stage]}.sbatch")
    assert _opt(argv, "--job-name") == f"vsft-{stage}"
    assert _opt(argv, "--gpus-per-node") == "a100:1" and _opt(argv, "--mem") == "120G"


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize("args,msg", [
    (["compare", "--stages", "train,boost"], "boost must come before train"),
    (["compare", "--stages", "eval_sft,train_sft"], "train_sft must come before eval_sft"),
    (["compare", "--stages", "eval_base,prepare"], "prepare must come before eval_base"),
    (["compare", "--stages", "boost,gpucheck"], "invalid compare stage"),
    (["compare", "--stages", ","], "stage"),
    (["compare", "--eval-model", "m"], "not compare"),
    (["compare", "--train-data", "d.jsonl"], "not compare"),
    (["compare", "--train-resume", "r"], "--train-resume"),
    (["compare", "--", "--limit", "3"], "single stage"),
    (["pipeline", "--stages", "train_sft"], "invalid pipeline stage"),
    (["eval_sft", "--train-resume", "r"], "train or train_sft"),
    (["eval_base", "--stages", "eval_base"], "only valid with pipeline or compare"),
])
def test_compare_validation(h, args, msg):
    r = h.submit(*args)
    assert r.returncode == 2, r.stderr
    assert msg in r.stderr, r.stderr
    assert h.sbatch_calls() == []


def test_train_sft_needs_data_or_prepare(h):
    r = h.submit("train_sft", "--no-monitor")
    assert r.returncode == 2 and "plain-SFT data missing" in r.stderr
    assert h.sbatch_calls() == []
    r = h.submit("train_sft", "--no-monitor", "--train-data", h.input, run_dir="y")
    assert r.returncode == 3, r.stderr
    assert h.sbatch_calls()[0]["env"]["VSFT_TRAIN_DATA"] == str(h.input)


def test_eval_sft_needs_trained_plain_sft(h):
    r = h.submit("eval_sft", "--no-monitor", "--eval-data", h.input)
    assert r.returncode == 2 and "plain-SFT model" in r.stderr
    run = h.runs / "x"
    td = run / "sft-baseline"
    td.mkdir(parents=True)
    (run / "status" / "train_sft.dir").write_text(f"{td}\n")
    r = h.submit("eval_sft", "--no-monitor", "--resume", run, run_dir=None)
    assert r.returncode == 2  # no SUCCEEDED.json yet
    (td / "SUCCEEDED.json").write_text("{}")
    h.make_test_sets(run)
    # the *sampling* model's record must not satisfy eval_sft
    (run / "status" / "train.dir").write_text(f"{run / 'sft'}\n")
    r = h.submit("eval_sft", "--no-monitor", "--resume", run, run_dir=None)
    assert r.returncode == 3, r.stderr


def test_eval_stages_need_test_sets(h):
    for i, stage in enumerate(("eval_base", "eval", "eval_sft")):
        extra = ["--eval-model", "Qwen/X"] if stage != "eval_base" else []
        r = h.submit(stage, "--no-monitor", *extra, run_dir=f"r{i}")
        assert r.returncode == 2 and f"{stage}: test set missing" in r.stderr, r.stderr
    assert h.sbatch_calls() == []
    run = h.runs / "r0"
    h.make_test_sets(run)
    r = h.submit("eval_base", "--no-monitor", "--resume", run, run_dir=None)
    assert r.returncode == 3, r.stderr
    # explicit --eval-data, or a dependency that will produce data, skip the check
    r = h.submit("eval_base", "--no-monitor", "--eval-data", h.input, run_dir="r5")
    assert r.returncode == 3, r.stderr
    h.add_job(900, "RUNNING|0:0|None@0")
    r = h.submit("eval_base", "--no-monitor", "--depend-on", "900", run_dir="r6")
    assert r.returncode == 3, r.stderr
    # compare including prepare needs nothing yet
    r = h.submit("compare", "--no-monitor", "--allow-duplicate", run_dir="r7")
    assert r.returncode == 3, r.stderr


def test_train_resume_allowed_for_train_sft(h):
    r = h.submit("train_sft", "--no-monitor", "--train-data", h.input)
    run = h.runs / "x"
    h.set_timeline(h.sbatch_calls()[0]["job_id"], "FAILED|1:0|x@0")
    state = run / "sft-baseline" / "state-step10"
    state.mkdir(parents=True)
    (state / "trainer_state.json").write_text("{}")
    r = h.submit("train_sft", "--no-monitor", "--train-data", h.input, "--resume", run,
                 "--train-resume", state, run_dir=None)
    assert r.returncode == 3, r.stderr
    assert h.sbatch_calls()[1]["env"]["VSFT_TRAIN_RESUME"] == str(state)


def test_duplicate_guard_per_new_stage(h):
    r = h.submit("train_sft", "--no-monitor", "--train-data", h.input)
    assert r.returncode == 3, r.stderr
    run = h.runs / "x"
    h.make_test_sets(run)
    again = ["--no-monitor", "--train-data", h.input, "--resume", run]
    r = h.submit("train_sft", *again, run_dir=None)
    assert r.returncode == 2 and "stage train_sft of this run is already PENDING" in r.stderr
    # other stages (including the sampling `train`) are not blocked by train_sft
    r = h.submit("train", *again, run_dir=None)
    assert r.returncode == 3, r.stderr
    r = h.submit("eval_base", "--no-monitor", "--resume", run, run_dir=None)
    assert r.returncode == 3, r.stderr
    r = h.submit("eval_base", "--no-monitor", "--resume", run, run_dir=None)
    assert r.returncode == 2 and "eval_base" in r.stderr
    # compare refuses if any of its stages is active
    h.make_test_sets(run)
    (run / "boosted.jsonl").write_text('{"id": 1}\n')
    r = h.submit("compare", "--no-monitor", "--resume", run, "--stages", "train_sft,eval_sft",
                 run_dir=None)
    assert r.returncode == 2 and "train_sft" in r.stderr
    st = {c["env"]["VSFT_STAGE"] for c in h.sbatch_calls()}
    assert st == {"train_sft", "train", "eval_base"}


# ------------------------------------------------------------------ real jobs (fake venv)
FAKE_PY = r'''#!/bin/bash
line="$(printf '%q ' "$@")"  # one write per call: concurrent jobs share the log
printf '%s #job=%s stage=%s\n' "$line" "$SLURM_JOB_ID" "$VSFT_STAGE" >> "$FAKE_ARGV_LOG"
get() { local want="$1"; shift; local prev=""; for a in "$@"; do
  if [ "$prev" = "$want" ]; then echo "$a"; return; fi; prev="$a"; done; }
if [ "$1" = -c ]; then echo 1; exit 0; fi
if [ "$2" = vsft.cli ] && [ "$VSFT_STAGE" = "${FAKE_FAIL_STAGE:-}" ]; then exit 1; fi
case "$2 $3" in
  "vsft.hpc config-get") if [ "${@: -1}" = models ]; then
       printf 'Qwen/Qwen3.5-fake\nQwen/Qwen3.5-verifier\n'; else echo "Qwen/Qwen3.5-fake"; fi
     exit 0 ;;
  "vsft.hpc fla-check") exit 0 ;;
  "vsft.hpc preflight") exit 0 ;;
  "vsft.hpc check-train") exit 0 ;;
  "vsft.cli boost") o="$(get --output "$@")"; echo '{"id":1}' > "$o"
     echo '{"num_rows": 1}' > "${o%.jsonl}.summary.json"; exit 0 ;;
  "vsft.cli train") d="$(get --output-dir "$@")"; mkdir -p "$d/final"
     echo '{}' > "$d/final/config.json"; echo x > "$d/final/model.safetensors"
     echo '{"final_step": 3}' > "$d/train_result.json"; exit 0 ;;
  "vsft.cli eval") o="$(get --output-dir "$@")"; l="$(get --label "$@")"; mkdir -p "$o"
     case "$l" in base) a=0.10 ;; sft) a=0.20 ;; sampling) a=0.30 ;; *) a=0.0 ;; esac
     printf '{"model": "%s", "label": "%s", "accuracy": {"math500_test": %s, "gsm8k_test": %s}}\n' \
       "$(get --model "$@")" "$l" "$a" "$a" > "$o/eval_summary.json"; exit 0 ;;
esac
echo "fake python: unhandled $*" >&2; exit 9
'''


def fake_venv(h):
    venv = h.root / "fakevenv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text(FAKE_PY)
    (venv / "bin" / "python").chmod(0o755)
    sha = hashlib.sha256((h.project / "uv.lock").read_bytes()).hexdigest()
    (venv / ".vsft-uv-lock.sha256").write_text(sha + "\n")
    return venv


def prepared_run(h, venv):
    """A run dir (code snapshot etc.) with prepared data, no active jobs."""
    r = h.submit("gpucheck", "--no-monitor", "--venv", venv)
    assert r.returncode == 3, r.stderr
    run = h.runs / "x"
    h.set_timeline(h.sbatch_calls()[0]["job_id"], "FAILED|1:0|x@0")
    h.make_test_sets(run)
    (run / "data" / "math_l345_train.jsonl").write_text('{"id": 1}\n')
    (run / "boosted.jsonl").write_text('{"id": 1, "expert": "e", "response": "r"}\n')
    return run


def argv_log(h):
    out = []
    for line in (h.root / "argv.log").read_text().splitlines():
        cmd, meta = line.rsplit(" #", 1)
        m = dict(kv.split("=", 1) for kv in meta.split())
        out.append((m["stage"], m["job"], shlex.split(cmd)))
    return out


def test_compare_real_jobs(h):
    venv = fake_venv(h)
    run = prepared_run(h, venv)
    env = {"FAKE_SBATCH_RUN": "1", "FAKE_PENDING_SECS": "6",
           "FAKE_ARGV_LOG": h.root / "argv.log"}
    stages = ["boost", "train", "eval", "train_sft", "eval_sft", "eval_base"]
    r = h.submit("compare", "--no-monitor", "--resume", run, "--venv", venv,
                 "--stages", ",".join(stages), run_dir=None, env=env)
    assert r.returncode == 3, r.stderr
    calls = {c["env"]["VSFT_STAGE"]: c for c in h.sbatch_calls()[1:]}
    assert list(calls) == stages
    ids = {s: c["job_id"] for s, c in calls.items()}
    deps = {s: _opt(c["argv"], "--dependency") for s, c in calls.items()}
    assert deps == {"boost": None, "train": f"afterok:{ids['boost']}",
                    "eval": f"afterok:{ids['train']}", "train_sft": f"afterok:{ids['boost']}",
                    "eval_sft": f"afterok:{ids['train_sft']}", "eval_base": None}
    for s in stages:
        j = h.wait_job(ids[s], timeout=120)
        assert j["state"] == "COMPLETED", (s, j, (run / "logs").iterdir())
    for s in stages:
        st = h.rjson(run / "status" / f"{s}.state")
        assert st["state"] == "SUCCEEDED" and str(st["job_id"]) == ids[s], s

    def cli(stage, sub):
        hits = [a for s, j, a in argv_log(h)
                if s == stage and j == ids[stage] and a[:3] == ["-m", "vsft.cli", sub]]
        assert len(hits) == 1, (stage, sub, argv_log(h))
        return hits[0]

    tr = cli("train", "train")
    assert _opt(tr, "--data") == str(run / "boosted.jsonl")
    assert _opt(tr, "--output-dir") == str(run / "sft")
    assert "train.source=expert" not in tr
    ts = cli("train_sft", "train")
    assert _opt(ts, "--data") == str(run / "boosted.jsonl")  # same rows as the sampling arm
    assert _opt(ts, "--output-dir") == str(run / "sft-baseline")
    assert ts[-2:] == ["--set", "train.source=expert"]
    assert (run / "status" / "train.dir").read_text().strip() == str(run / "sft")
    assert (run / "status" / "train_sft.dir").read_text().strip() == str(run / "sft-baseline")
    for d in ("sft", "sft-baseline"):
        assert h.rjson(run / d / "SUCCEEDED.json")["final_model"] == str(run / d / "final")
    expect = {"eval": ("sampling", str(run / "sft" / "final")),
              "eval_sft": ("sft", str(run / "sft-baseline" / "final")),
              "eval_base": ("base", "Qwen/Qwen3.5-fake")}
    for s, (label, model) in expect.items():
        a = cli(s, "eval")
        assert _opt(a, "--label") == label and _opt(a, "--model") == model, (s, a)
        assert _opt(a, "--output-dir") == str(run / "eval" / f"{label}-{ids[s]}")
        assert h.rjson(run / "status" / f"{s}.SUCCEEDED")["label"] == label
    # results.sh prints the comparison table from the run's eval summaries
    res = h.run("results.sh", run)
    assert res.returncode == 0, res.stderr
    out = res.stdout
    assert "[train] Sampling SFT: state=SUCCEEDED" in out
    assert "[train_sft] plain SFT baseline: state=SUCCEEDED" in out
    for s in ("eval_base", "eval_sft", "eval"):
        assert f"[{s}] state=SUCCEEDED" in out
    assert "| Initial model | 10.0 | 10.0 | 10.0 |" in out
    assert "| Plain SFT (expert traces) | 20.0 | 20.0 | 20.0 |" in out
    assert "| Sampling SFT | 30.0 | 30.0 | 30.0 |" in out
    assert "| Sampling - Plain SFT (pts) | +10.0 | +10.0 | +10.0 |" in out
    # boost preflight required the student and the separate verifier
    pf = [a for s, j, a in argv_log(h) if s == "boost" and a[:3] == ["-m", "vsft.hpc",
                                                                      "preflight"]]
    assert len(pf) == 1
    models = [pf[0][i + 1] for i, x in enumerate(pf[0]) if x == "--require-model"]
    assert models == ["Qwen/Qwen3.5-fake", "Qwen/Qwen3.5-verifier"]


def test_compare_failed_branch_cancels_dependents_only(h):
    venv = fake_venv(h)
    run = prepared_run(h, venv)
    # break the sampling arm's training: its eval is cancelled, the plain-SFT arm finishes
    env = {"FAKE_SBATCH_RUN": "1", "FAKE_PENDING_SECS": "6", "FAKE_FAIL_STAGE": "train",
           "FAKE_ARGV_LOG": h.root / "argv.log"}
    r = h.submit("compare", "--no-monitor", "--resume", run, "--venv", venv,
                 "--stages", "boost,train,eval,train_sft,eval_sft", run_dir=None, env=env)
    assert r.returncode == 3, r.stderr
    ids = {c["env"]["VSFT_STAGE"]: c["job_id"] for c in h.sbatch_calls()[1:]}
    assert h.wait_job(ids["boost"], timeout=120)["state"] == "COMPLETED"
    assert h.wait_job(ids["train"], timeout=120)["state"] == "FAILED"
    assert h.wait_job(ids["eval"], timeout=120)["state"] == "CANCELLED"
    assert h.wait_job(ids["eval_sft"], timeout=120)["state"] == "COMPLETED"
    assert h.rjson(run / "status" / "eval_sft.state")["state"] == "SUCCEEDED"


def test_train_sft_extra_args_keep_source_last(h):
    venv = fake_venv(h)
    run = prepared_run(h, venv)
    env = {"FAKE_SBATCH_RUN": "1", "FAKE_ARGV_LOG": h.root / "argv.log"}
    r = h.submit("train_sft", "--no-monitor", "--resume", run, "--venv", venv, run_dir=None,
                 env=env)
    assert r.returncode == 3, r.stderr
    jid = h.sbatch_calls()[-1]["job_id"]
    h.wait_job(jid)
    r = h.submit("train_sft", "--no-monitor", "--resume", run, "--venv", venv, "--",
                 "--set", "train.source=response", "--limit", "3", run_dir=None, env=env)
    assert r.returncode == 3, r.stderr
    jid2 = h.sbatch_calls()[-1]["job_id"]
    assert h.wait_job(jid2)["state"] == "COMPLETED"
    a = [a for s, j, a in argv_log(h) if j == jid2 and a[:3] == ["-m", "vsft.cli", "train"]][0]
    assert a[-6:] == ["--set", "train.source=response", "--limit", "3",
                      "--set", "train.source=expert"]
    # second attempt claims a fresh directory and becomes the recorded plain-SFT model
    assert _opt(a, "--output-dir") == str(run / "sft-baseline-001")
    assert (run / "status" / "train_sft.dir").read_text().strip() == str(
        run / "sft-baseline-001")


def test_eval_base_model_override(h):
    venv = fake_venv(h)
    run = prepared_run(h, venv)
    env = {"FAKE_SBATCH_RUN": "1", "FAKE_ARGV_LOG": h.root / "argv.log"}
    r = h.submit("eval_base", "--no-monitor", "--resume", run, "--venv", venv,
                 "--eval-model", "Qwen/Other", run_dir=None, env=env)
    assert r.returncode == 3, r.stderr
    jid = h.sbatch_calls()[-1]["job_id"]
    assert h.wait_job(jid)["state"] == "COMPLETED"
    a = [a for s, j, a in argv_log(h) if j == jid and a[:3] == ["-m", "vsft.cli", "eval"]][0]
    assert _opt(a, "--model") == "Qwen/Other" and _opt(a, "--label") == "base"
    assert not any(j == jid and "config-get" in a for s, j, a in argv_log(h))


def test_eval_sft_job_refuses_without_success_record(h):
    venv = fake_venv(h)
    run = prepared_run(h, venv)
    td = run / "sft-baseline"
    (td / "final").mkdir(parents=True)
    (td / "SUCCEEDED.json").write_text("{}")
    (run / "status" / "train_sft.dir").write_text(f"{td}\n")
    env = {"FAKE_SBATCH_RUN": "1", "FAKE_PENDING_SECS": "4",
           "FAKE_ARGV_LOG": h.root / "argv.log"}
    r = h.submit("eval_sft", "--no-monitor", "--resume", run, "--venv", venv, run_dir=None,
                 env=env)
    assert r.returncode == 3, r.stderr
    (td / "SUCCEEDED.json").unlink()
    jid = h.sbatch_calls()[-1]["job_id"]
    assert h.wait_job(jid)["rc"] == 64
    time.sleep(0.2)
    assert h.rjson(run / "status" / "eval_sft.FAILED")["exit_code"] == 64


# ------------------------------------------------------------------ results.sh table only
def test_results_comparison_table_synthetic(h):
    run = h.root / "run"
    (run / "status").mkdir(parents=True)
    (run / "manifest.json").write_text("{}")
    meta = {"datasets": {"math500_test": {"sha256": "ab" * 32, "n": 500}},
            "decoding": {"temperature": 0.0, "max_new_tokens": 1856}}
    for label, jid, acc in (("sampling", 0, 0.01), ("base", 1, 0.1), ("sft", 2, 0.25),
                            ("sampling", 3, 0.4)):
        d = run / "eval" / f"{label}-{jid}"
        d.mkdir(parents=True)
        (d / "eval_summary.json").write_text(json.dumps(
            {"model": f"m-{label}", "label": label, "accuracy": {"math500_test": acc},
             **meta}))
        time.sleep(0.02)
    (run / "status" / "eval_sft.state").write_text(json.dumps({"state": "SUCCEEDED"}))
    r = h.run("results.sh", run)
    assert r.returncode == 0, r.stderr
    assert "| Method | math500_test | Avg |" in r.stdout
    assert "| Sampling SFT | 40.0 | 40.0 |" in r.stdout  # newest summary per label wins
    assert "| Sampling - Plain SFT (pts) | +15.0 | +15.0 |" in r.stdout
    assert "[eval_sft] state=SUCCEEDED" in r.stdout
    assert "not comparable" not in r.stdout


def test_results_reports_incomparable_rows_and_arm_mismatch(h):
    run = h.root / "run"
    (run / "status").mkdir(parents=True)
    for label, sha, n in (("sft", "aa", 500), ("sampling", "bb", 499)):
        d = run / "eval" / f"{label}-1"
        d.mkdir(parents=True)
        (d / "eval_summary.json").write_text(json.dumps(
            {"label": label, "accuracy": {"math500_test": 0.5},
             "datasets": {"math500_test": {"sha256": sha * 32, "n": n}}, "decoding": {}}))
    for stage, n, steps in (("train", 1000, 62), ("train_sft", 900, 56)):
        td = run / f"dir-{stage}"
        td.mkdir()
        (run / "status" / f"{stage}.dir").write_text(f"{td}\n")
        (td / "train_manifest.json").write_text(json.dumps(
            {"data": str(run / "boosted.jsonl"),
             "geometry": {"num_examples": n, "total_steps": steps}}))
    r = h.run("results.sh", run)
    assert r.returncode == 0, r.stderr
    out = r.stdout + r.stderr
    assert "examples=1000 total_steps=62" in out and "examples=900 total_steps=56" in out
    assert "WARNING: the two SFT arms trained on different example counts" in out
    assert "different test data" in out and "rows may not be comparable" in out


def test_results_flags_failed_comparison(h):
    """compare.py exits 2 on mismatched rows; results.sh should say the table is unusable."""
    run = h.root / "run"
    (run / "status").mkdir(parents=True)
    for label, sha in (("sft", "aa"), ("sampling", "bb")):
        d = run / "eval" / f"{label}-1"
        d.mkdir(parents=True)
        (d / "eval_summary.json").write_text(json.dumps(
            {"label": label, "accuracy": {"m": 0.5},
             "datasets": {"m": {"sha256": sha * 32, "n": 5}}, "decoding": {}}))
    r = h.run("results.sh", run)
    assert r.returncode == 0
    assert "comparison incomplete or not comparable" in r.stdout + r.stderr


def test_results_without_eval_dir(h):
    run = h.root / "run"
    (run / "status").mkdir(parents=True)
    r = h.run("results.sh", run)
    assert r.returncode == 0 and "Method" not in r.stdout
