"""In-allocation helpers for Slurm jobs (run inside the project venv on the compute node).

    python -m vsft.hpc preflight --run-dir R --stage S [--require-gpu --gpu-type A100 ...]
    python -m vsft.hpc download-model --config C
    python -m vsft.hpc config-get --config C model.name_or_path
    python -m vsft.hpc check-train --dir D

`preflight` verifies the accelerator before the application starts and only then writes the
atomic STARTED marker; any failure exits nonzero without writing it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

from vsft.io import write_json_atomic


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_marker(run_dir: Path, stage: str, state: str, **info) -> Path:
    path = run_dir / "status" / f"{stage}.{state}"
    payload = {
        "stage": stage,
        "state": state,
        "time_utc": _utc(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "host": socket.gethostname(),
        "run_dir": str(run_dir),
        **info,
    }
    write_json_atomic(path, payload)
    write_json_atomic(run_dir / "status" / f"{stage}.state", payload)  # latest state
    return path


def gpu_report(require_gpu: bool, gpu_type: str | None, min_mem_gb: float, need_bf16: bool):
    import torch

    devices = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            devices.append(
                {
                    "index": i,
                    "name": p.name,
                    "total_gb": round(total / 2**30, 1),
                    "free_gb": round(free / 2**30, 1),
                    "capability": f"{p.major}.{p.minor}",
                    "bf16": p.major >= 8,
                }
            )
    report = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "devices": devices,
    }
    problems = []
    if require_gpu:
        if not devices:
            problems.append("no CUDA device visible in this allocation")
        for d in devices:
            if gpu_type and gpu_type.lower() not in d["name"].lower():
                problems.append(f"GPU {d['index']} is {d['name']!r}, expected {gpu_type}")
            if d["total_gb"] < min_mem_gb:
                problems.append(f"GPU {d['index']} has {d['total_gb']} GB < {min_mem_gb} GB")
            if need_bf16 and not d["bf16"]:
                problems.append(f"GPU {d['index']} lacks bf16 (capability {d['capability']})")
        if devices and not problems:
            # Tiny bf16 matmul proves the driver/runtime actually work.
            a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
            v = (a @ a).float().abs().mean().item()
            if not math.isfinite(v):
                problems.append("bf16 matmul produced a non-finite value")
            torch.cuda.synchronize()
    return report, problems


def cmd_preflight(a) -> int:
    run_dir = Path(a.run_dir).resolve()
    problems = []
    if not run_dir.is_dir():
        problems.append(f"run dir {run_dir} does not exist")
    elif not os.access(run_dir, os.W_OK):
        problems.append(f"run dir {run_dir} is not writable")
    for p in a.require_file:
        if not Path(p).is_file() or Path(p).stat().st_size == 0:
            problems.append(f"required input missing or empty: {p}")
    report, gpu_problems = gpu_report(a.require_gpu, a.gpu_type, a.min_gpu_mem_gb, a.require_bf16)
    problems += gpu_problems
    print(json.dumps({"preflight": report, "problems": problems}, indent=2), flush=True)
    if problems:
        print("PREFLIGHT FAILED:\n  " + "\n  ".join(problems), file=sys.stderr, flush=True)
        return 2
    write_marker(
        run_dir,
        a.stage,
        "STARTED",
        slurm={
            k: os.environ.get(k)
            for k in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NODELIST",
                "SLURM_CPUS_PER_TASK",
                "SLURM_MEM_PER_NODE",
                "SLURM_GPUS_ON_NODE",
                "SLURM_JOB_PARTITION",
            )
        },
        accelerator=report,
    )
    return 0


def cmd_config_get(a) -> int:
    from vsft.config import load_config

    obj = load_config(a.config, a.overrides)
    for part in a.key.split("."):
        obj = getattr(obj, part)
    print(obj)
    return 0


def cmd_download_model(a) -> int:
    from huggingface_hub import snapshot_download

    from vsft.config import load_config

    name = a.model or load_config(a.config, a.overrides).model.name_or_path
    if Path(name).exists():
        print(f"model {name} is a local path; nothing to download")
        return 0
    path = snapshot_download(
        name,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.tiktoken"],
    )
    print(f"downloaded {name} -> {path}")
    return 0


def cmd_check_train(a) -> int:
    """Verify a training output before evaluation: final model + finite history."""
    d = Path(a.dir)
    final = d / "final"
    problems = []
    if not (final / "config.json").is_file():
        problems.append(f"{final}/config.json missing")
    if not list(final.glob("*.safetensors")):
        problems.append(f"no *.safetensors in {final}")
    if not (d / "train_result.json").is_file():
        problems.append(f"{d}/train_result.json missing (training did not finish)")
    hist = d / "train_history.jsonl"
    if hist.is_file():
        losses = [
            json.loads(line).get("loss", float("nan"))
            for line in hist.read_text().splitlines()
            if line.strip()
        ]
        if any(not math.isfinite(x) for x in losses):
            problems.append("train_history.jsonl contains a non-finite or missing loss")
    if problems:
        print("CHECK FAILED:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 3
    print(f"training output OK: {final}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m vsft.hpc")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("preflight")
    f.add_argument("--run-dir", required=True)
    f.add_argument("--stage", required=True)
    f.add_argument("--require-gpu", action="store_true")
    f.add_argument("--gpu-type", default=None, help="substring of the device name, e.g. A100")
    f.add_argument("--min-gpu-mem-gb", type=float, default=0.0)
    f.add_argument("--require-bf16", action="store_true")
    f.add_argument("--require-file", action="append", default=[])
    f.set_defaults(func=cmd_preflight)

    for name, fn in (("config-get", cmd_config_get), ("download-model", cmd_download_model)):
        c = sub.add_parser(name)
        c.add_argument("--config")
        c.add_argument("--set", dest="overrides", action="append", default=[])
        if name == "config-get":
            c.add_argument("key")
        else:
            c.add_argument("--model")
        c.set_defaults(func=fn)

    t = sub.add_parser("check-train")
    t.add_argument("--dir", required=True)
    t.set_defaults(func=cmd_check_train)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
