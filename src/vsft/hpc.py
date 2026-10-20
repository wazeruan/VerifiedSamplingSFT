"""In-allocation helpers for Slurm jobs (run inside the project venv on the compute node).

    python -m vsft.hpc preflight --run-dir R --stage S [--require-gpu --gpu-type H100 ...]
    python -m vsft.hpc download-model --config C
    python -m vsft.hpc config-get --config C model.name_or_path
    python -m vsft.hpc check-train --dir D

Status protocol (shared with slurm/lib/receipt.py, which owns the on-disk format):
  PREFLIGHT_OK  written by `preflight` after accelerator/input/environment checks pass;
  STARTED       written by the application itself (`notify_started`) once its model or
                backend is initialized, so engine/model-load failures never count as started.
A job asks the workload to stop gracefully by creating $VSFT_STOP_FILE (`stop_requested`).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

LOCK_STAMP = ".vsft-uv-lock.sha256"  # written into the venv by the setup stage


def _receipt_module():
    """Load slurm/lib/receipt.py (stdlib-only) so markers have exactly one implementation."""
    path = os.environ.get("VSFT_RECEIPT_PY") or str(
        Path(__file__).resolve().parents[2] / "slurm" / "lib" / "receipt.py"
    )
    spec = importlib.util.spec_from_file_location("vsft_receipt", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"receipt helper not found at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_marker(run_dir: Path, stage: str, state: str, **info) -> Path:
    """Rank-aware marker write (never regresses the same job's state)."""
    _receipt_module().mark(str(run_dir), stage, state, info)
    return Path(run_dir) / "status" / f"{stage}.{state}"


def notify_started(**info) -> None:
    """Called by the application after initialization when running under a Slurm stage."""
    run_dir, stage = os.environ.get("VSFT_RUN_DIR"), os.environ.get("VSFT_STAGE")
    if not run_dir or not stage:
        return
    if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) or 0) != 0:
        return
    write_marker(Path(run_dir), stage, "STARTED", **info)
    print(f"[vsft] {stage}: application initialized; STARTED recorded", flush=True)


def request_stop(*_args) -> None:
    """Signal handler: create the stop file (or set an in-process flag) for a graceful stop."""
    global _STOP_FLAG
    _STOP_FLAG = True
    f = os.environ.get("VSFT_STOP_FILE")
    if f:
        try:
            Path(f).touch()
        except OSError:
            pass


def install_term_handler() -> None:
    """Make SIGTERM request a graceful stop instead of killing the process outright."""
    import signal

    if os.environ.get("VSFT_STAGE"):  # only inside a Slurm stage; plain CLI use is unchanged
        signal.signal(signal.SIGTERM, request_stop)


def record_stopped(path: Path | str | None = None, **info) -> None:
    """Write the 'stopped resumably' record (with this job id) that the job script checks."""
    target = path or os.environ.get("VSFT_STOPPED_RECORD")
    if not target:
        return
    from vsft.io import write_json_atomic

    write_json_atomic(Path(target), {"job_id": os.environ.get("SLURM_JOB_ID"), **info})


_STOP_FLAG = False


def stop_requested() -> bool:
    """True once the job script asked for a graceful stop (preemption/time-limit warning)."""
    if _STOP_FLAG:
        return True
    f = os.environ.get("VSFT_STOP_FILE")
    return bool(f) and os.path.exists(f)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Files a run needs from a model repo. The download and the cache check must use the same
# patterns: huggingface_hub treats a snapshot as incomplete if files outside the patterns
# are missing, which made a finished download look broken.
MODEL_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.tiktoken"]


def model_available(name: str) -> str | None:
    """None if the model is a local dir or fully in the HF cache, else a problem string."""
    if Path(name).is_dir():
        return None if (Path(name) / "config.json").is_file() else f"{name} has no config.json"
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(name, local_files_only=True, allow_patterns=MODEL_PATTERNS)
    except Exception as e:  # noqa: BLE001 - any cache miss is reported, not raised
        return f"model {name!r} not in the HF cache ({type(e).__name__}); run the prepare stage"
    if not list(Path(path).glob("*.safetensors")):
        return f"model {name!r} cache at {path} has no *.safetensors"
    return None


def gpu_report(
    require_gpu: bool,
    gpu_type: str | None,
    min_mem_gb: float,
    need_bf16: bool,
    capability: str | None = None,
):
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
        "cuda_arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        "driver": _nvidia_driver_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "devices": devices,
    }
    problems = []
    if require_gpu:
        if torch.version.cuda is None:
            problems.append(
                f"torch {torch.__version__} is a CPU-only build; a CUDA build is required"
            )
        if not devices:
            problems.append("no CUDA device visible in this allocation (driver/runtime mismatch?)")
        if capability and devices:
            # e.g. H100 = sm_90: the torch build must ship kernels for it.
            arch = "sm_" + capability.replace(".", "")
            if arch not in report["cuda_arch_list"]:
                problems.append(f"torch build lacks {arch} kernels: {report['cuda_arch_list']}")
        for d in devices:
            if capability and d["capability"] != capability:
                problems.append(
                    f"GPU {d['index']} has compute capability {d['capability']}, "
                    f"expected {capability}"
                )
            if gpu_type and gpu_type.lower() not in d["name"].lower():
                problems.append(f"GPU {d['index']} is {d['name']!r}, expected {gpu_type}")
            if d["total_gb"] < min_mem_gb:
                problems.append(f"GPU {d['index']} has {d['total_gb']} GB < {min_mem_gb} GB")
            if need_bf16 and not d["bf16"]:
                problems.append(f"GPU {d['index']} lacks bf16 (capability {d['capability']})")
        if devices and not problems:
            # Tiny bf16 and TF32 matmuls on every visible device prove the driver/runtime work.
            for d in devices:
                dev = f"cuda:{d['index']}"
                a = torch.randn(256, 256, device=dev, dtype=torch.bfloat16)
                b = torch.randn(256, 256, device=dev, dtype=torch.float32)
                v = (a @ a).float().abs().mean().item() + (b @ b).abs().mean().item()
                torch.cuda.synchronize(dev)
                if not math.isfinite(v):
                    problems.append(f"matmul on {dev} produced a non-finite value")
    return report, problems


def _nvidia_driver_version() -> str | None:
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            .stdout.strip()
            .splitlines()
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out[0] if out else None


def enable_tf32_math() -> None:
    """Ampere+ (H100): allow TF32 for fp32 matmuls/convolutions. No-op without CUDA."""
    import torch

    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


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
    for m in a.require_model:
        msg = model_available(m)
        if msg:
            problems.append(msg)
    if a.require_lock:
        stamp = Path(sys.prefix) / LOCK_STAMP
        want = _sha256(Path(a.require_lock))
        have = stamp.read_text().strip() if stamp.is_file() else ""
        if have != want:
            problems.append(
                f"venv {sys.prefix} was not built from {a.require_lock} "
                f"(stamp {have[:12] or 'missing'} != lock {want[:12]}); run the setup stage"
            )
    report, gpu_problems = gpu_report(
        a.require_gpu, a.gpu_type, a.min_gpu_mem_gb, a.require_bf16, a.gpu_capability
    )
    problems += gpu_problems
    print(json.dumps({"preflight": report, "problems": problems}, indent=2), flush=True)
    if problems:
        print("PREFLIGHT FAILED:\n  " + "\n  ".join(problems), file=sys.stderr, flush=True)
        return 2
    write_marker(
        run_dir,
        a.stage,
        "PREFLIGHT_OK",
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

    cfg = load_config(a.config, a.overrides)
    if a.key == "models":  # one per line: student, then the verifier if separate
        print("\n".join(config_models(cfg)))
        return 0
    obj = cfg
    for part in a.key.split("."):
        obj = getattr(obj, part)
    print(obj)
    return 0


def cmd_download_model(a) -> int:
    from huggingface_hub import snapshot_download

    from vsft.config import load_config

    names = [a.model] if a.model else config_models(load_config(a.config, a.overrides))
    for name in names:
        if Path(name).exists():
            print(f"model {name} is a local path; nothing to download")
            continue
        path = snapshot_download(
            name,
            allow_patterns=MODEL_PATTERNS,
        )
        print(f"downloaded {name} -> {path}")
    return 0


def config_models(cfg) -> list[str]:
    """Every model a run needs: the student, plus the verifier when it is a separate model."""
    names = [cfg.model.name_or_path]
    if cfg.verifier.model and cfg.verifier.model not in names:
        names.append(cfg.verifier.model)
    return names


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
        bad = 0
        for line in hist.read_text().splitlines():
            if not line.strip():
                continue
            try:
                x = json.loads(line).get("loss")
            except (ValueError, AttributeError):
                bad += 1  # torn/partial line (e.g. killed mid-write) or not an object
                continue
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
                bad += 1
        if bad:
            problems.append(
                f"train_history.jsonl has {bad} non-finite, missing or unreadable loss record(s)"
            )
    if problems:
        print("CHECK FAILED:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 3
    print(f"training output OK: {final}")
    return 0


def cmd_fla_check(_a) -> int:
    """Launch one flash-linear-attention Gated-DeltaNet kernel on the GPU (Qwen3.5 path)."""
    import torch

    if not torch.cuda.is_available():
        print("FLA CHECK FAILED: no CUDA device", file=sys.stderr)
        return 2
    try:
        import fla
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except ImportError as e:
        print(f"FLA CHECK FAILED: cannot import flash-linear-attention: {e}", file=sys.stderr)
        return 2
    b, t, h, d = 1, 128, 2, 64
    dev, dt = "cuda", torch.bfloat16
    q = torch.nn.functional.normalize(torch.randn(b, t, h, d, device=dev), dim=-1).to(dt)
    k = torch.nn.functional.normalize(torch.randn(b, t, h, d, device=dev), dim=-1).to(dt)
    v = torch.randn(b, t, h, d, device=dev, dtype=dt)
    g = -torch.nn.functional.softplus(torch.randn(b, t, h, device=dev)).to(dt)  # log decay
    beta = torch.sigmoid(torch.randn(b, t, h, device=dev)).to(dt)
    try:
        out = chunk_gated_delta_rule(q, k, v, g, beta)
        o = out[0] if isinstance(out, tuple) else out
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001 - report any kernel/compile failure clearly
        print(f"FLA CHECK FAILED: kernel launch error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    if not torch.isfinite(o.float()).all():
        print("FLA CHECK FAILED: non-finite kernel output", file=sys.stderr)
        return 2
    print(f"fla {getattr(fla, '__version__', '?')}: chunk_gated_delta_rule OK on {dev}")
    return 0


def gpu_memory_used() -> str | None:
    """'used/total MiB' per GPU from nvidia-smi (covers vLLM engine-core subprocesses)."""
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out.replace("\n", "; ") or None


def _stamp(lock: str) -> int:
    (Path(sys.prefix) / LOCK_STAMP).write_text(_sha256(Path(lock)) + "\n")
    print(f"stamped {sys.prefix} with {lock}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m vsft.hpc")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("preflight")
    f.add_argument("--run-dir", required=True)
    f.add_argument("--stage", required=True)
    f.add_argument("--require-gpu", action="store_true")
    f.add_argument("--gpu-type", default=None, help="substring of the device name, e.g. H100")
    f.add_argument("--min-gpu-mem-gb", type=float, default=0.0)
    f.add_argument("--require-bf16", action="store_true")
    f.add_argument("--gpu-capability", default=None, help="exact compute capability, e.g. 9.0")
    f.add_argument("--require-file", action="append", default=[])
    f.add_argument("--require-model", action="append", default=[], help="HF id or local dir")
    f.add_argument("--require-lock", help="uv.lock the venv must have been built from")
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

    fc = sub.add_parser("fla-check", help="run one flash-linear-attention kernel on the GPU")
    fc.set_defaults(func=cmd_fla_check)

    st = sub.add_parser("stamp-lock", help="record the uv.lock hash in the current venv")
    st.add_argument("--lock", required=True)
    st.set_defaults(func=lambda a: _stamp(a.lock))

    t = sub.add_parser("check-train")
    t.add_argument("--dir", required=True)
    t.set_defaults(func=cmd_check_train)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
