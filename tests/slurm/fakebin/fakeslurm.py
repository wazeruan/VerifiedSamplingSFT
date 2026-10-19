#!/usr/bin/env python3
"""Fake Slurm/SSH backend for testing slurm/*.sh (stdlib only).

Every shim in this directory (sbatch, squeue, sacct, scontrol, ssh, rsync) is a symlink-free
tiny wrapper that calls main(<tool>, argv). All state lives under $FAKE_SLURM_DIR:

  counter              next job id
  sbatch.log           JSONL: one record per sbatch call (argv, parsed options, VSFT_* env)
  calls.log            JSONL: every squeue/sacct/scontrol call (tool, argv, rc)
  jobs/<id>.json       job record (see Job below)
  ssh.log, rsync.log   JSONL argv records

Knobs (environment variables, read at call time):

  sbatch
    FAKE_SBATCH_MODE      ok (prints "ID") | cluster ("ID;cluster") | garbage (job registered,
                          stdout unparsable) | hidden (garbage stdout, job NOT registered) |
                          reject (stderr + exit 1, no job)
    FAKE_SBATCH_REJECT_ON N   reject only the N-th sbatch call (1-based); others use the mode
    FAKE_SBATCH_RUN=1     actually run the job script (detached runner): PENDING for
                          FAKE_PENDING_SECS (default 0.3), then RUNNING, then COMPLETED|0:0,
                          FAILED|rc:0, or CANCELLED|0:sig when killed by a signal
    FAKE_JOB_TIMELINE     non-run jobs: "STATE|EXIT|REASON@t;STATE|EXIT|REASON@t2" where t is
                          seconds since submission (default "PENDING|0:0|Priority@0").
                          STATE GONE = unknown to both squeue and sacct.
    FAKE_JOB_LOG_TEXT     non-run jobs: written to the job's --output and --error files
    FAKE_GONE_SECS        run jobs: invisible to squeue+sacct for the first N seconds
  squeue
    FAKE_SQUEUE_FAIL=1        always fail (exit 1, "Socket timed out")
    FAKE_SQUEUE_FAIL_FIRST=N  fail the first N squeue calls only
    FAKE_SQUEUE_SHOW_DONE=1   show finished (terminal) jobs as "STATE|-|None" like slurmctld
                              does during MinJobAge
    FAKE_SQUEUE_GONE          error (default; real behaviour: "Invalid job id", exit 1) |
                              empty (exit 0, no output) for jobs that left the queue
  sacct
    FAKE_SACCT_FAIL=1         always fail
    FAKE_SACCT_LAG_SECS=N     accounting delay: jobs younger than N s are not in sacct
  ssh
    FAKE_SSH_EXEC=1           run the remote command with bash -c in $FAKE_REMOTE_HOME
                              (HOME=$FAKE_REMOTE_HOME); exit with its status
    FAKE_SSH_RC               exit status when not executing (default 0)
  rsync
    FAKE_RSYNC_RC             exit status (default 0)
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time

ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED"}
VALUE_OPTS = {
    "--account", "--job-name", "--chdir", "--output", "--error", "--comment", "--export",
    "--gpus-per-node", "--cpus-per-task", "--mem", "--partition", "--time", "--qos",
    "--reservation", "--constraint", "--dependency", "--gres", "--nodes", "--ntasks",
    "-A", "-J", "-D", "-o", "-e", "-p", "-t", "-d",
}


def sdir() -> str:
    d = os.environ.get("FAKE_SLURM_DIR")
    if not d:
        sys.stderr.write("fakeslurm: FAKE_SLURM_DIR not set\n")
        sys.exit(99)
    os.makedirs(os.path.join(d, "jobs"), exist_ok=True)
    return d


class Locked:
    def __init__(self, name="global"):
        self.path = os.path.join(sdir(), f".{name}.lock")

    def __enter__(self):
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *a):
        os.close(self.fd)


def _atomic_write(path, obj):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def append_log(name, rec):
    with Locked("log"), open(os.path.join(sdir(), name), "a") as f:
        f.write(json.dumps(rec) + "\n")


def job_path(jid):
    return os.path.join(sdir(), "jobs", f"{jid}.json")


def load_job(jid):
    try:
        with open(job_path(jid)) as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return None


def save_job(job):
    with Locked("jobs"):
        _atomic_write(job_path(job["id"]), job)


def update_job(jid, **kw):
    with Locked("jobs"):
        job = load_job(jid) or {"id": str(jid)}
        job.update(kw)
        _atomic_write(job_path(jid), job)
        return job


def next_id():
    with Locked("counter"):
        p = os.path.join(sdir(), "counter")
        try:
            n = int(open(p).read().strip())
        except (FileNotFoundError, ValueError):
            n = int(os.environ.get("FAKE_SLURM_START_ID", "1000"))
        with open(p, "w") as f:
            f.write(str(n + 1))
        return str(n)


def bump(name):
    with Locked("counter"):
        p = os.path.join(sdir(), f"count.{name}")
        try:
            n = int(open(p).read().strip())
        except (FileNotFoundError, ValueError):
            n = 0
        n += 1
        with open(p, "w") as f:
            f.write(str(n))
        return n


def parse_timeline(spec):
    out = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        st, _, t = part.rpartition("@")
        if not _:
            st, t = part, "0"
        fields = (st.split("|") + ["", "", ""])[:3]
        out.append((float(t), fields[0], fields[1] or "0:0", fields[2] or "None"))
    return sorted(out, key=lambda x: x[0])


def current(job):
    """(state, exit, reason) as the scheduler sees it now; state GONE = invisible."""
    if job is None:
        return ("GONE", "", "")
    age = time.time() - job.get("submit_time", time.time())
    if job.get("gone_secs") and age < float(job["gone_secs"]):
        return ("GONE", "", "")
    if job.get("timeline") and not job.get("run"):
        cur = ("GONE", "", "")
        for t, st, ex, rs in parse_timeline(job["timeline"]):
            if age >= t:
                cur = (st, ex, rs)
        return cur
    return (job.get("state", "PENDING"), job.get("exit", "0:0"), job.get("reason", "None"))


def fmt(fmtstr, job, st):
    def rep(m):
        c = m.group(1)
        return {
            "i": job["id"], "T": st[0], "r": st[2], "k": job.get("comment") or "(null)",
            "j": job.get("name", ""), "R": st[2], "M": "0:01",
        }.get(c, m.group(0))

    return re.sub(r"%\.?\d*([A-Za-z])", rep, fmtstr)


# ----------------------------------------------------------------------------- tools
def parse_sbatch(argv):
    opts, script, script_args, flags = {}, None, [], []
    i = 0
    while i < len(argv):
        a = argv[i]
        if script is None and a.startswith("-"):
            if "=" in a and a.startswith("--"):
                k, v = a.split("=", 1)
                opts[k] = v
            elif a in VALUE_OPTS:
                opts[a] = argv[i + 1]
                i += 1
            else:
                flags.append(a)
        elif script is None:
            script = a
        else:
            script_args.append(a)
        i += 1
    return opts, flags, script, script_args


def subst(path, jid, name, chdir):
    if not path:
        return None
    p = path.replace("%j", jid).replace("%x", name or "")
    if not os.path.isabs(p):
        p = os.path.join(chdir or os.getcwd(), p)
    return p


def sbatch(argv):
    opts, flags, script, script_args = parse_sbatch(argv)
    n = bump("sbatch")
    mode = os.environ.get("FAKE_SBATCH_MODE", "ok")
    if os.environ.get("FAKE_SBATCH_REJECT_ON"):
        mode = "reject" if n == int(os.environ["FAKE_SBATCH_REJECT_ON"]) else "ok"
    rec = {
        "n": n, "argv": argv, "opts": opts, "flags": flags, "script": script,
        "script_args": script_args, "mode": mode, "cwd": os.getcwd(),
        "env": {k: v for k, v in os.environ.items() if k.startswith("VSFT_")},
    }
    if mode == "reject":
        rec["job_id"] = None
        append_log("sbatch.log", rec)
        sys.stderr.write("sbatch: error: Batch job submission failed: Invalid account or "
                         "account/partition combination specified\n")
        return 1
    if script is None or not os.path.isfile(script):
        append_log("sbatch.log", rec)
        sys.stderr.write(f"sbatch: error: Unable to open file {script}\n")
        return 1
    jid = next_id()
    rec["job_id"] = jid if mode != "hidden" else None
    append_log("sbatch.log", rec)
    name = opts.get("--job-name") or os.path.basename(script)
    chdir = opts.get("--chdir") or os.getcwd()
    run = os.environ.get("FAKE_SBATCH_RUN") == "1"
    job = {
        "id": jid, "name": name, "comment": opts.get("--comment"), "script": script,
        "script_args": script_args, "chdir": chdir,
        "out": subst(opts.get("--output") or "slurm-%j.out", jid, name, chdir),
        "err": subst(opts.get("--error") or opts.get("--output") or "slurm-%j.out", jid, name,
                     chdir),
        "submit_time": time.time(), "run": run, "state": "PENDING", "exit": "0:0",
        "reason": "None", "dependency": opts.get("--dependency"),
        "timeline": None if run else os.environ.get("FAKE_JOB_TIMELINE",
                                                     "PENDING|0:0|Priority@0"),
        "gone_secs": os.environ.get("FAKE_GONE_SECS"),
    }
    if mode != "hidden":
        save_job(job)
        text = os.environ.get("FAKE_JOB_LOG_TEXT")
        if text and not run:
            for p in {job["out"], job["err"]}:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "a") as f:
                    f.write(text + "\n")
        if run:
            subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "__run__", jid],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True, env=dict(os.environ),
            )
    if mode == "ok":
        print(jid if "--parsable" in flags else f"Submitted batch job {jid}")
    elif mode == "cluster":
        print(f"{jid};fakecluster")
    else:  # garbage / hidden
        print("sbatch: warning: slurmctld response delayed\nSubmitted batch job (pending)")
    return 0


def runner(jid):
    job = load_job(jid)
    update_job(jid, runner_pid=os.getpid())
    time.sleep(float(os.environ.get("FAKE_PENDING_SECS", "0.3")))
    env = dict(os.environ)
    env.update(SLURM_JOB_ID=jid, SLURM_JOB_NAME=job["name"], SLURM_SUBMIT_DIR=job["chdir"],
               SLURM_JOB_NODELIST="fakenode01", SLURM_CPUS_PER_TASK="2")
    for p in (job["out"], job["err"]):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    out = open(job["out"], "a")
    err = out if job["err"] == job["out"] else open(job["err"], "a")
    proc = subprocess.Popen([job["script"], *job["script_args"]], cwd=job["chdir"], env=env,
                            stdin=subprocess.DEVNULL, stdout=out, stderr=err)
    update_job(jid, state="RUNNING", pid=proc.pid, start_time=time.time())
    rc = proc.wait()
    if rc == 0:
        update_job(jid, state="COMPLETED", exit="0:0", rc=rc, end_time=time.time())
    elif rc < 0:
        update_job(jid, state="CANCELLED", exit=f"0:{-rc}", rc=rc, end_time=time.time())
    else:
        update_job(jid, state="FAILED", exit=f"{rc}:0", reason="NonZeroExitCode", rc=rc,
                   end_time=time.time())
    return 0


def _flagval(argv, *names):
    for i, a in enumerate(argv):
        for n in names:
            if a == n and i + 1 < len(argv):
                return argv[i + 1]
            if a.startswith(n + "="):
                return a.split("=", 1)[1]
    return None


def all_jobs():
    d = os.path.join(sdir(), "jobs")
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            j = load_job(f[:-5])
            if j:
                out.append(j)
    return out


def squeue(argv):
    n = bump("squeue")
    if os.environ.get("FAKE_SQUEUE_FAIL") == "1" or n <= int(
        os.environ.get("FAKE_SQUEUE_FAIL_FIRST", "0") or 0
    ):
        sys.stderr.write("squeue: error: slurm_load_jobs error: Socket timed out on send/recv\n")
        return 1
    fmtstr = _flagval(argv, "-o", "--format") or "%i %j %T %r"
    header = "-h" not in argv and "--noheader" not in argv
    ids = _flagval(argv, "-j", "--job", "--jobs")
    lines = []
    if ids:
        for jid in ids.split(","):
            j = load_job(jid)
            st = current(j)
            if st[0] in ACTIVE:
                lines.append(fmt(fmtstr, j, st))
            elif st[0] != "GONE" and os.environ.get("FAKE_SQUEUE_SHOW_DONE") == "1":
                # slurmctld keeps finished jobs for MinJobAge: state only, no exit code
                lines.append(fmt(fmtstr, j, (st[0], "", "None")))
            elif os.environ.get("FAKE_SQUEUE_GONE", "error") == "error":
                sys.stderr.write("slurm_load_jobs error: Invalid job id specified\n")
                return 1
    else:
        for j in all_jobs():
            st = current(j)
            if st[0] in ACTIVE:
                lines.append(fmt(fmtstr, j, st))
    if header and lines:
        print("JOBID STATE")
    for line in lines:
        print(line)
    return 0


def sacct(argv):
    bump("sacct")
    if os.environ.get("FAKE_SACCT_FAIL") == "1":
        sys.stderr.write("sacct: error: Problem talking to the database: Connection refused\n")
        return 1
    ids = _flagval(argv, "-j", "--jobs") or ""
    fields = (_flagval(argv, "-o", "--format") or "JobID,State,ExitCode").split(",")
    for jid in ids.split(","):
        j = load_job(jid)
        st = current(j)
        if st[0] == "GONE":
            continue
        lag = float(os.environ.get("FAKE_SACCT_LAG_SECS", "0") or 0)
        if lag and time.time() - j.get("submit_time", 0) < lag:
            continue  # slurmdbd has not recorded the job yet
        vals = {"JobID": jid, "State": st[0], "ExitCode": st[1], "Reason": st[2],
                "JobName": j.get("name", ""), "Elapsed": "00:00:01"}
        print("|".join(vals.get(f.split("%")[0], "") for f in fields))
    return 0


def scontrol(argv):
    bump("scontrol")
    jid = argv[-1] if argv else ""
    j = load_job(jid)
    if j is None:
        sys.stderr.write("slurm_load_jobs error: Invalid job id specified\n")
        return 1
    st = current(j)
    print(f"JobId={jid} JobName={j.get('name')} JobState={st[0]} Reason={st[2]} "
          f"ExitCode={st[1]} Comment={j.get('comment')}")
    return 0


def ssh(argv):
    i, opts = 0, []
    while i < len(argv) and argv[i].startswith("-"):
        if argv[i] in ("-o", "-p", "-i", "-l", "-F", "-J"):
            opts += argv[i:i + 2]
            i += 2
        else:
            opts.append(argv[i])
            i += 1
    host = argv[i] if i < len(argv) else None
    cmd = " ".join(argv[i + 1:])
    rec = {"argv": argv, "host": host, "cmd": cmd, "opts": opts}
    rc = int(os.environ.get("FAKE_SSH_RC", "0"))
    if os.environ.get("FAKE_SSH_EXEC") == "1":
        home = os.environ["FAKE_REMOTE_HOME"]
        env = dict(os.environ, HOME=home)
        p = subprocess.run(["bash", "-c", cmd], cwd=home, env=env)
        rc = p.returncode
    rec["rc"] = rc
    append_log("ssh.log", rec)
    return rc


def rsync(argv):
    append_log("rsync.log", {"argv": argv})
    return int(os.environ.get("FAKE_RSYNC_RC", "0"))


TOOLS = {"sbatch": sbatch, "squeue": squeue, "sacct": sacct, "scontrol": scontrol,
         "ssh": ssh, "rsync": rsync}


def main(tool, argv):
    if tool == "__run__":
        return runner(argv[0])
    rc = TOOLS[tool](argv)
    if tool in ("squeue", "sacct", "scontrol"):
        append_log("calls.log", {"tool": tool, "argv": argv, "rc": rc, "t": time.time()})
    return rc


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "__run__":
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        sys.exit(runner(sys.argv[2]))
    sys.exit(main(os.path.basename(sys.argv[1]), sys.argv[2:]))
