"""Real-model `vsft train` / `vsft eval` with Qwen3.5-0.8B (LoRA, CPU/MPS).

Run with: VSFT_RUN_SLOW=1 HF_HUB_OFFLINE=1 HF_HOME=<cache dir> uv run pytest -m slow
"""

import json
import math
from pathlib import Path

import pytest

from vsft.cli import main

pytestmark = pytest.mark.slow

MODEL = "Qwen/Qwen3.5-0.8B"
FIXTURE = Path(__file__).parent / "fixtures" / "tiny_math.jsonl"
SETTINGS = [
    f"model.name_or_path={MODEL}", "train.lora_r=8", "train.num_epochs=1",
    "train.per_device_batch_size=1", "train.grad_accum_steps=1", "train.max_seq_len=256",
    "train.source=expert", "train.log_every_steps=1", "train.learning_rate=1e-4",
]


def _sets(items):
    return [x for s in items for x in ("--set", s)]


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("qwen_train")
    assert main(["train", "--data", str(FIXTURE), "--output-dir", str(out), *_sets(SETTINGS)]) == 0
    return out


def test_train_outputs(run_dir):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(run_dir / "final")
    assert not any("lora" in k for k in m.state_dict())
    hist = [json.loads(x) for x in (run_dir / "train_history.jsonl").read_text().splitlines()]
    assert len(hist) == 3 and all(math.isfinite(h["loss"]) for h in hist)
    man = json.loads((run_dir / "train_manifest.json").read_text())
    prec = man["precision"]
    assert set(prec) >= {"configured_param_dtype", "effective_param_dtype",
                         "configured_mixed_precision", "effective_mixed_precision", "save_dtype"}
    assert prec["effective_param_dtype"].startswith("torch.")
    from vsft.hpc import main as hpc_main

    assert hpc_main(["check-train", "--dir", str(run_dir)]) == 0


def test_resume_with_more_epochs_refused(run_dir):
    """Resume is strict by design: extending a 1-epoch run to 2 epochs is refused."""
    with pytest.raises(ValueError, match="num_epochs"):
        main(["train", "--data", str(FIXTURE), "--output-dir", str(run_dir),
              "--resume-from", str(run_dir / "state-epoch1"),
              *_sets([*SETTINGS, "train.num_epochs=2"])])
    hist = [json.loads(x) for x in (run_dir / "train_history.jsonl").read_text().splitlines()]
    assert [h["epoch"] for h in hist] == [0, 0, 0]


def test_eval_trained_model(run_dir, tmp_path):
    out = tmp_path / "eval"
    assert main(["eval", "--model", str(run_dir / "final"), "--data", str(FIXTURE),
                 "--output-dir", str(out), "--limit", "2",
                 *_sets(["model.device=cpu", "model.dtype=float32", "eval.max_new_tokens=48",
                         "eval.batch_size=2"])]) == 0
    m = json.loads((out / "tiny_math.metrics.json").read_text())
    assert m["num_examples"] == 2 and 0.0 <= m["accuracy"] <= 1.0
    summ = json.loads((out / "eval_summary.json").read_text())
    assert summ["model"] == str(run_dir / "final") and "tiny_math" in summ["accuracy"]
    preds = [json.loads(x) for x in (out / "tiny_math.predictions.jsonl").read_text().splitlines()]
    assert len(preds) == 2 and all(isinstance(p["response"], str) for p in preds)
