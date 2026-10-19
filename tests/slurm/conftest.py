"""Fake-Slurm harness for testing slurm/*.sh end to end (see fakebin/fakeslurm.py for knobs).

Each test gets a scratch *project* (a copy of src/, slurm/, configs/, pyproject.toml, uv.lock
taken at session start) plus a shim venv at PROJECT/.venv whose python is the real project
interpreter but whose sys.prefix is the shim (so `stamp-lock` never touches the real .venv).
Scripts run with $VSFT_TEST_BASH (default /bin/bash, i.e. 3.2 on macOS).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import stat
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FAKEBIN = HERE / "fakebin"
SCRATCH = REPO / ".scratch" / "slurmtest"
BASH = os.environ.get("VSFT_TEST_BASH", "/bin/bash")
UV = shutil.which("uv") or "/opt/homebrew/bin/uv"
REAL_VENV = REPO / ".venv"
TS_RE = r"\d{8}T\d{6}Z"


def _ignore(_d, names):
    return [n for n in names if n in ("__pycache__", ".pytest_cache")]


def make_venv_shim(dst: Path, stamp_lock: Path | None = None) -> Path:
    """A venv directory whose bin/python is the real interpreter (sys.prefix = dst)."""
    (dst / "bin").mkdir(parents=True, exist_ok=True)
    shutil.copy(REAL_VENV / "pyvenv.cfg", dst / "pyvenv.cfg")
    if not (dst / "lib").exists():
        (dst / "lib").symlink_to(REAL_VENV / "lib")
    py = dst / "bin" / "python"
    if not py.exists():
        py.symlink_to(REAL_VENV / "bin" / "python")
    if stamp_lock is not None:
        subprocess.run([str(py), "-m", "vsft.hpc", "stamp-lock", "--lock", str(stamp_lock)],
                       check=True, capture_output=True, env=_clean_env())
    return dst


def _clean_env():
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("VSFT_", "FAKE_", "SLURM_", "VIRTUAL_ENV", "PYTHONPATH", "UV_"))}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


@pytest.fixture(scope="session")
def template(tmp_path_factory):
    """Snapshot of the repository's code taken once per session."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    t = tmp_path_factory.mktemp("template") / "project"
    t.mkdir()
    for name in ("src", "slurm", "configs"):
        shutil.copytree(REPO / name, t / name, ignore=_ignore)
    for name in ("pyproject.toml", "uv.lock", "README.md"):
        if (REPO / name).exists():
            shutil.copy(REPO / name, t / name)
    make_venv_shim(t / ".venv", stamp_lock=t / "uv.lock")
    return t


def rmtree_force(p: Path):
    def onerr(func, path, _exc):
        try:
            os.chmod(os.path.dirname(path), 0o755)
            os.chmod(path, 0o755)
            func(path)
        except OSError:
            pass

    if p.exists():
        for root, dirs, _files in os.walk(p):
            for d in dirs:
                try:
                    os.chmod(os.path.join(root, d), 0o755)
                except OSError:
                    pass
        shutil.rmtree(p, onerror=onerr)


class Harness:
    def __init__(self, root: Path, template: Path):
        self.root = root
        self.project = root / "project"
        shutil.copytree(template, self.project, symlinks=True)
        self.state = root / "fakeslurm"
        self.state.mkdir()
        self.bin = root / "bin"
        self.bin.mkdir()
        (self.bin / "bash").symlink_to(BASH)
        (self.bin / "uv").symlink_to(UV)
        (root / "tmp").mkdir()
        self.runs = root / "runs"
        self.venv = self.project / ".venv"  # shim venv stamped with the snapshot's uv.lock
        self.env = _clean_env()
        self.env.update(
            PATH=f"{FAKEBIN}:{self.bin}:/usr/bin:/bin:/usr/sbin:/sbin",
            FAKE_SLURM_DIR=str(self.state),
            TMPDIR=str(root / "tmp"),
            LC_ALL="C",
        )
        self.input = root / "input.jsonl"
        self.input.write_text('{"id": 1}\n')

    # -------------------------------------------------------------- running scripts
    def script(self, name: str) -> Path:
        return self.project / "slurm" / name

    def run(self, name, *args, env=None, timeout=300, cwd=None):
        e = dict(self.env)
        e.update({k: str(v) for k, v in (env or {}).items()})
        return subprocess.run([BASH, str(self.script(name)), *map(str, args)],
                              capture_output=True, text=True, env=e, timeout=timeout,
                              cwd=cwd or self.root)

    def submit(self, stage, *args, account="def-test", run_dir="x", env=None, timeout=400,
               min_free=True, venv=True):
        a = [stage]
        if venv is True and "--venv" not in map(str, args):
            a += ["--venv", str(self.venv)]
        elif venv not in (True, False, None):
            a += ["--venv", str(venv)]
        if account is not None:
            a += ["--account", account]
        if run_dir is not None:
            a += ["--run-dir", str(self.runs / run_dir)]
        if min_free:
            a += ["--min-free-gb", "1"]
        return self.run("submit.sh", *a, *args, env=env, timeout=timeout)

    # -------------------------------------------------------------- job script overrides
    def job_script(self, stage, body):
        p = self.project / "slurm" / "jobs" / f"{stage}.sbatch"
        p.write_text(
            "#!/usr/bin/env bash\n"
            'source "${VSFT_CODE_DIR:?}/slurm/lib/job_common.sh"\n' + body.strip() + "\n")
        p.chmod(0o755)
        return p

    # -------------------------------------------------------------- fake scheduler
    def sbatch_calls(self):
        p = self.state / "sbatch.log"
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

    def log(self, name):
        p = self.state / name
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

    def job(self, jid):
        p = self.state / "jobs" / f"{jid}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def add_job(self, jid, timeline, comment=None, name="vsft-old"):
        (self.state / "jobs").mkdir(exist_ok=True)
        (self.state / "jobs" / f"{jid}.json").write_text(json.dumps({
            "id": str(jid), "name": name, "comment": comment, "submit_time": time.time() - 1,
            "run": False, "timeline": timeline}))

    def set_timeline(self, jid, timeline):
        j = self.job(jid)
        j["timeline"] = timeline
        j["submit_time"] = time.time()
        (self.state / "jobs" / f"{jid}.json").write_text(json.dumps(j))

    def wait_job(self, jid, timeout=90, states=("COMPLETED", "FAILED", "CANCELLED")):
        end = time.time() + timeout
        while time.time() < end:
            j = self.job(jid)
            if j and j.get("state") in states and "rc" in j:
                return j
            time.sleep(0.2)
        raise AssertionError(f"job {jid} did not finish: {self.job(jid)}")

    def wait_for(self, pred, timeout=60, what="condition"):
        end = time.time() + timeout
        while time.time() < end:
            v = pred()
            if v:
                return v
            time.sleep(0.2)
        raise AssertionError(f"timed out waiting for {what}")

    def kill_all(self):
        for f in (self.state / "jobs").glob("*.json") if (self.state / "jobs").exists() else []:
            try:
                j = json.loads(f.read_text())
            except ValueError:
                continue
            for key in ("pid", "runner_pid"):
                pid = j.get(key)
                if pid and "rc" not in j:
                    for sig in (signal.SIGKILL,):
                        try:
                            os.killpg(os.getpgid(pid), sig)
                        except (ProcessLookupError, PermissionError):
                            pass

    # -------------------------------------------------------------- run-dir helpers
    @staticmethod
    def rjson(p):
        return json.loads(Path(p).read_text())

    def only_run_dir(self, name="x"):
        return self.runs / name


@pytest.fixture
def h(tmp_path, template):
    hh = Harness(tmp_path, template)
    yield hh
    hh.kill_all()
    # make read-only code snapshots removable by pytest's basetemp cleanup
    for root, dirs, files in os.walk(tmp_path):
        for n in dirs + files:
            p = os.path.join(root, n)
            try:
                if not os.path.islink(p):
                    os.chmod(p, os.stat(p).st_mode | stat.S_IWUSR)
            except OSError:
                pass
