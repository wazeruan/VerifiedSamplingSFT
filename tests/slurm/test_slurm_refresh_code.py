"""submit.sh --refresh-code: replaces a resumed run's code snapshot, refused while any job of
the run is active, and only logs under --dry-run."""

import json
import re

import pytest
from conftest import TS_RE


def _fresh_run(h, timeline="FAILED|1:0|x@0"):
    r = h.submit("gpucheck", "--no-monitor")
    assert r.returncode == 3, r.stderr
    run = h.runs / "x"
    jid = h.sbatch_calls()[0]["job_id"]
    assert list((run / "status").glob("*.state")), "expected a status/*.state file"
    h.set_timeline(jid, timeline)
    return run, jid


def _prev_dirs(run):
    return sorted(p for p in run.glob("code.prev-*") if p.is_dir())


def test_refresh_code_requires_resume(h):
    r = h.submit("gpucheck", "--refresh-code", "--no-monitor")
    assert r.returncode == 2, r.stderr
    assert "--refresh-code needs --resume" in r.stderr
    assert h.sbatch_calls() == []


def test_refresh_code_replaces_snapshot(h):
    run, _jid = _fresh_run(h)
    (run / "code" / "marker_old.txt").write_text("old snapshot")
    (run / "code" / "accelerate.yaml").write_text("mixed_precision: bf16\n")
    (h.project / "slurm" / "lib" / "marker_new.txt").write_text("new checkout")

    r = h.submit("gpucheck", "--no-monitor", "--resume", run, "--refresh-code", run_dir=None)
    assert r.returncode == 3, r.stderr

    # the new snapshot reflects the checkout
    assert (run / "code" / "slurm" / "lib" / "marker_new.txt").read_text() == "new checkout"
    assert not (run / "code" / "marker_old.txt").exists()
    # the old snapshot moved aside intact
    prevs = _prev_dirs(run)
    assert len(prevs) == 1, prevs
    prev = prevs[0]
    assert re.fullmatch(r"code\.prev-" + TS_RE, prev.name), prev.name
    assert (prev / "marker_old.txt").read_text() == "old snapshot"
    # accelerate.yaml carried over into the new snapshot
    assert (run / "code" / "accelerate.yaml").read_text() == "mixed_precision: bf16\n"
    # manifest records the refresh
    m = json.loads((run / "manifest.json").read_text())
    refreshes = m["code_refreshes"]
    assert len(refreshes) == 1
    assert re.match(TS_RE + r":", refreshes[0]), refreshes[0]
    assert refreshes[0].endswith("previous=" + str(prev)), refreshes[0]


@pytest.mark.parametrize("state", ["PENDING", "RUNNING"])
def test_refresh_code_refused_while_job_active(h, state):
    timeline = f"{state}|0:0|None@0"
    run, jid = _fresh_run(h, timeline=timeline)
    (run / "code" / "marker_old.txt").write_text("old snapshot")
    manifest_before = (run / "manifest.json").read_text()

    r = h.submit("gpucheck", "--no-monitor", "--resume", run, "--refresh-code", run_dir=None)
    assert r.returncode == 2, r.stderr
    assert f"job {jid}" in r.stderr and state in r.stderr
    # code/ untouched, nothing moved aside, manifest unchanged, no new sbatch call
    assert (run / "code" / "marker_old.txt").read_text() == "old snapshot"
    assert _prev_dirs(run) == []
    assert (run / "manifest.json").read_text() == manifest_before
    assert len(h.sbatch_calls()) == 1


def test_refresh_code_dry_run_changes_nothing(h):
    run, _jid = _fresh_run(h)
    (run / "code" / "marker_old.txt").write_text("old snapshot")
    (h.project / "slurm" / "lib" / "marker_new.txt").write_text("new checkout")
    manifest_before = (run / "manifest.json").read_text()

    r = h.submit("gpucheck", "--no-monitor", "--resume", run, "--refresh-code", "--dry-run",
                 run_dir=None)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "would replace" in r.stdout + r.stderr
    assert (run / "code" / "marker_old.txt").read_text() == "old snapshot"
    assert not (run / "code" / "marker_new.txt").exists()
    assert not (run / "code" / "slurm" / "lib" / "marker_new.txt").exists()
    assert _prev_dirs(run) == []
    assert (run / "manifest.json").read_text() == manifest_before


def test_refresh_code_refused_when_scheduler_unreachable(h):
    run, jid = _fresh_run(h, timeline="PENDING|0:0|Priority@0")
    (run / "code" / "marker_old.txt").write_text("old snapshot")
    r = h.submit("gpucheck", "--no-monitor", "--resume", run, "--refresh-code", run_dir=None,
                 env={"FAKE_SQUEUE_FAIL": "1", "FAKE_SACCT_FAIL": "1"}, timeout=400)
    assert r.returncode == 2, r.stderr
    assert "cannot verify job" in r.stderr
    assert (run / "code" / "marker_old.txt").read_text() == "old snapshot"
    assert _prev_dirs(run) == []
