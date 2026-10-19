"""A100-specific checks: --gpu-capability (validation, dry-run, env and preflight passthrough)
and vsft.hpc.gpu_report against a monkeypatched torch.cuda (A100 / V100 / build w/o sm_80)."""

import hashlib
import re
import shlex
from types import SimpleNamespace

import pytest

# ------------------------------------------------------------------ submit.sh flag


@pytest.mark.parametrize("bad", ["8", "8.", "sm_80", "8.0.1", "a.b", "80"])
def test_gpu_capability_validation(h, bad):
    r = h.submit("gpucheck", "--dry-run", "--gpu-capability", bad)
    assert r.returncode == 2 and "--gpu-capability" in r.stderr, r.stderr
    assert h.sbatch_calls() == []


@pytest.mark.parametrize("val", ["8.0", "9.0", "10.0", ""])
def test_gpu_capability_dry_run_ok(h, val):
    r = h.submit("gpucheck", "--dry-run", "--gpu-capability", val)
    assert r.returncode == 0, r.stderr
    assert "[dry-run] gpucheck:" in r.stdout


@pytest.mark.parametrize("args,expected", [([], "8.0"), (["--gpu-capability", "9.0"], "9.0"),
                                           (["--gpu-capability", ""], "")])
def test_gpu_capability_exported(h, args, expected):
    r = h.submit("gpucheck", "--no-monitor", *args)
    assert r.returncode == 3, r.stderr
    env = h.sbatch_calls()[0]["env"]
    assert env["VSFT_GPU_CAPABILITY"] == expected
    assert env["VSFT_GPU_TYPE"] == "A100" and env["VSFT_MIN_GPU_MEM_GB"] == "39"


def _argv_logging_venv(h):
    """A venv whose python logs its argv and fails like a failed preflight would."""
    venv = h.root / "logvenv"
    (venv / "bin").mkdir(parents=True)
    log = h.root / "py.argv"
    py = venv / "bin" / "python"
    py.write_text(f'#!/bin/bash\nprintf "%q " "$@" >> {log}; echo >> {log}\nexit 2\n')
    py.chmod(0o755)
    sha = hashlib.sha256((h.project / "uv.lock").read_bytes()).hexdigest()
    (venv / ".vsft-uv-lock.sha256").write_text(sha + "\n")
    return venv, log


@pytest.mark.parametrize("args,expected", [([], ["--gpu-capability", "8.0"]),
                                           (["--gpu-capability", "9.0"],
                                            ["--gpu-capability", "9.0"]),
                                           (["--gpu-capability", ""], None)])
def test_gpu_capability_reaches_preflight(h, args, expected):
    venv, log = _argv_logging_venv(h)
    r = h.submit("gpucheck", "--venv", venv, *args, env={"FAKE_SBATCH_RUN": "1"})
    assert r.returncode == 1, r.stdout + r.stderr  # preflight "failed" (exit 2)
    calls = [shlex.split(x) for x in log.read_text().splitlines() if "preflight" in x]
    assert len(calls) == 1, log.read_text()
    argv = calls[0]
    for flag in ("--require-gpu", "--require-bf16", "--require-lock"):
        assert flag in argv
    assert argv[argv.index("--gpu-type") + 1] == "A100"
    assert argv[argv.index("--min-gpu-mem-gb") + 1] == "39"
    if expected is None:
        assert "--gpu-capability" not in argv
    else:
        i = argv.index("--gpu-capability")
        assert argv[i:i + 2] == expected


# ------------------------------------------------------------------ gpu_report unit tests
GiB = 2**30


@pytest.fixture
def fake_cuda(monkeypatch):
    import torch

    from vsft import hpc

    state = {"devices": [], "arch": [], "cuda": "12.4", "matmuls": []}
    real_randn = torch.randn

    def randn(*shape, device=None, dtype=None, **kw):
        state["matmuls"].append((str(device), dtype))
        return real_randn(*shape, dtype=dtype if dtype != torch.bfloat16 else torch.float32)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: bool(state["devices"]))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(state["devices"]))
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: state["devices"][i]["props"])
    def mem_get_info(i):
        total = state["devices"][i]["total"]
        return total - GiB, total

    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: list(state["arch"]))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch, "randn", randn)
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(hpc, "_nvidia_driver_version", lambda: "535.183.01")

    def configure(devices, arch, cuda="12.4"):
        state["devices"] = [
            {"props": SimpleNamespace(name=n, major=ma, minor=mi), "total": int(gb * GiB)}
            for n, ma, mi, gb in devices]
        state["arch"] = arch
        monkeypatch.setattr(torch.version, "cuda", cuda)
        return state

    return configure


A100 = ("NVIDIA A100-SXM4-40GB", 8, 0, 40.0)
V100 = ("Tesla V100-SXM2-32GB", 7, 0, 32.0)
ARCH = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]


def report(**kw):
    from vsft.hpc import gpu_report

    args = dict(require_gpu=True, gpu_type="A100", min_mem_gb=39, need_bf16=True,
                capability="8.0")
    args.update(kw)
    return gpu_report(**args)


def test_gpu_report_a100_passes(fake_cuda):
    state = fake_cuda([A100, A100], ARCH)
    rep, problems = report()
    assert problems == []
    assert rep["cuda_arch_list"] == ARCH and rep["driver"] == "535.183.01"
    assert [d["capability"] for d in rep["devices"]] == ["8.0", "8.0"]
    assert rep["devices"][0]["total_gb"] == 40.0 and rep["devices"][0]["bf16"] is True
    # bf16 and fp32 matmuls on every device
    devs = {d for d, _ in state["matmuls"]}
    assert devs == {"cuda:0", "cuda:1"}
    import torch
    assert {dt for _, dt in state["matmuls"]} == {torch.bfloat16, torch.float32}


def test_gpu_report_v100_fails(fake_cuda):
    state = fake_cuda([V100], ARCH)
    _rep, problems = report()
    text = "\n".join(problems)
    assert "compute capability 7.0, expected 8.0" in text
    assert "expected A100" in text
    assert "32.0 GB < 39" in text
    assert "lacks bf16" in text
    assert state["matmuls"] == []  # no kernels launched on a rejected device


def test_gpu_report_build_without_sm80_fails(fake_cuda):
    state = fake_cuda([A100], ["sm_70", "sm_75"])
    _rep, problems = report()
    assert len(problems) == 1 and re.search(r"lacks sm_80 kernels", problems[0]), problems
    assert state["matmuls"] == []


def test_gpu_report_cpu_only_torch(fake_cuda):
    fake_cuda([], [], cuda=None)
    _rep, problems = report()
    text = "\n".join(problems)
    assert "CPU-only build" in text and "no CUDA device" in text


def test_gpu_report_mixed_devices_fail(fake_cuda):
    fake_cuda([A100, V100], ARCH)
    _rep, problems = report()
    assert any("GPU 1" in p and "7.0" in p for p in problems)
    assert not any("GPU 0" in p for p in problems)


def test_gpu_report_any_capability(fake_cuda):
    fake_cuda([("NVIDIA H100 80GB HBM3", 9, 0, 80.0)], ARCH)
    _rep, problems = report(capability=None, gpu_type=None)
    assert problems == []
    _rep, problems = report(capability="", gpu_type="")
    assert problems == []
    _rep, problems = report()  # A100 defaults reject an H100
    assert any("9.0, expected 8.0" in p for p in problems)


def test_gpu_report_not_required(fake_cuda):
    fake_cuda([], [], cuda=None)
    rep, problems = report(require_gpu=False)
    assert problems == [] and rep["cuda_available"] is False


def test_preflight_cli_has_capability_flag():
    import subprocess

    from conftest import REAL_VENV, _clean_env
    p = subprocess.run([str(REAL_VENV / "bin" / "python"), "-m", "vsft.hpc", "preflight",
                        "--help"], capture_output=True, text=True, env=_clean_env())
    assert p.returncode == 0 and "--gpu-capability" in p.stdout
