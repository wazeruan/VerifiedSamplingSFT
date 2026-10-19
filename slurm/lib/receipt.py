#!/usr/bin/env python3
"""Stdlib-only receipt/marker helper (works on login nodes without the project venv).

  receipt.py update <file.json> key=value ...     merge keys (locked, atomic)
  receipt.py append <file.json> key value          append to a list key (locked, atomic)
  receipt.py get <file.json> key                   print a value ('' if missing)
  receipt.py mark <run_dir> <stage> <STATE> key=value ...
        write <run_dir>/status/<stage>.<STATE> and advance <stage>.state
  receipt.py state <run_dir> <stage> [job_id]      latest state ('' if none / other job)
  receipt.py claim <file.json> key=value ...       create a new file exclusively (fails if exists)

`<stage>.state` never moves backwards for the same job (SUBMITTED < PREFLIGHT_OK <
STARTED < SUCCEEDED|FAILED), so a fast job's STARTED/SUCCEEDED cannot be overwritten by
the wrapper's late SUBMITTED. A newer job id replaces the state; an older (numerically
smaller) job id never does, so a late update from a superseded job cannot hide a newer one.
Read-modify-write updates are serialized with flock on a sidecar lock file (skipped if
the filesystem does not support flock). Files honour the umask (group-readable on
shared project space) instead of mkstemp's 0600.
"""

import contextlib
import json
import os
import socket
import sys
import tempfile
import time

RANK = {"SUBMITTED": 0, "PREFLIGHT_OK": 1, "STARTED": 2, "SUCCEEDED": 3, "FAILED": 3}
_UMASK = os.umask(0)
os.umask(_UMASK)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@contextlib.contextmanager
def locked(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd = os.open(path + ".lock", os.O_CREAT | os.O_RDWR, 0o666 & ~_UMASK)
    try:
        try:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass  # no flock on this filesystem: best effort
        yield
    finally:
        os.close(fd)


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return {}


def save(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp.", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.chmod(tmp, 0o666 & ~_UMASK)
    os.replace(tmp, path)


def parse_kv(items):
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"expected key=value, got {it!r}")
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def mark(run_dir, stage, state, extra):
    if state not in RANK:
        raise SystemExit(f"unknown state {state!r}; expected one of {sorted(RANK)}")
    info = {
        "stage": stage,
        "state": state,
        "time_utc": utc(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "host": socket.gethostname(),
        "run_dir": os.path.abspath(run_dir),
    }
    info.update(extra)
    status = os.path.join(run_dir, "status")
    state_file = os.path.join(status, f"{stage}.state")
    with locked(state_file):
        save(os.path.join(status, f"{stage}.{state}"), info)
        cur = load(state_file)
        same_job = str(cur.get("job_id")) == str(info.get("job_id")) and cur.get("job_id")
        cur_id, new_id = str(cur.get("job_id") or ""), str(info.get("job_id") or "")
        if not same_job and cur_id.isdigit() and new_id.isdigit() and int(new_id) < int(cur_id):
            return  # an older job's late update never replaces a newer job's state
        if same_job and RANK.get(cur.get("state"), -1) > RANK[state]:
            return  # never regress the same job's state
        if same_job and RANK.get(cur.get("state"), -1) == 3 and cur.get("state") != state:
            return  # terminal state is final
        save(state_file, info)


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    cmd = argv[1]
    if cmd == "update":
        with locked(argv[2]):
            obj = load(argv[2])
            obj.update(parse_kv(argv[3:]))
            obj["updated_utc"] = utc()
            save(argv[2], obj)
    elif cmd == "append":
        with locked(argv[2]):
            obj = load(argv[2])
            obj.setdefault(argv[3], []).append(argv[4])
            obj["updated_utc"] = utc()
            save(argv[2], obj)
    elif cmd == "claim":
        path = argv[2]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666 & ~_UMASK)
        except FileExistsError:
            raise SystemExit(f"already exists: {path}") from None
        os.close(fd)
        obj = parse_kv(argv[3:])
        obj["created_utc"] = utc()
        with locked(path):
            save(path, obj)
    elif cmd == "get":
        v = load(argv[2]).get(argv[3], "")
        print(v if isinstance(v, str) else json.dumps(v))
    elif cmd == "mark":
        mark(argv[2], argv[3], argv[4], parse_kv(argv[5:]))
    elif cmd == "state":
        cur = load(os.path.join(argv[2], "status", f"{argv[3]}.state"))
        if len(argv) > 4 and str(cur.get("job_id")) != argv[4]:
            print("")
        else:
            print(cur.get("state", ""))
    else:
        raise SystemExit(f"unknown command {cmd!r}\n{__doc__}")


if __name__ == "__main__":
    main(sys.argv)
