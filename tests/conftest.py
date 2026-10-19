import json
import os
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_math.jsonl"


_JOB_ENV = ("VSFT_STOP_FILE", "VSFT_RUN_DIR", "VSFT_STAGE", "VSFT_RECEIPT_PY",
            "VSFT_STOPPED_RECORD", "SLURM_JOB_ID", "RANK", "LOCAL_RANK")


@pytest.fixture(autouse=True)
def _isolate_job_env(monkeypatch):
    """Tests never inherit Slurm/stop-file settings from the calling shell, never leak the
    in-process stop flag, and never leave cli.main's SIGTERM handler installed."""
    import signal

    import vsft.hpc

    for k in _JOB_ENV:
        monkeypatch.delenv(k, raising=False)
    # never touch the network (e.g. `vsft eval` probing AutoConfig for an uncached model)
    import huggingface_hub.constants as hf_constants

    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    vsft.hpc._STOP_FLAG = False
    old = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, old)
    vsft.hpc._STOP_FLAG = False


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: real-model integration test (run with VSFT_RUN_SLOW=1)"
    )


def pytest_collection_modifyitems(config, items):
    if os.environ.get("VSFT_RUN_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="set VSFT_RUN_SLOW=1 to run slow tests")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    """A 2-layer random Qwen2 model + tiny byte-level BPE tokenizer saved locally (offline).

    Byte-level so that AutoTokenizer (which maps model_type qwen2 to Qwen2Tokenizer and
    installs Qwen's byte-level pre-tokenizer) encodes every string without <unk>.
    """
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import AutoTokenizer, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    from vsft.prompts import STUDENT_SYSTEM

    texts = [STUDENT_SYSTEM, "System: User: Assistant:"]
    for line in FIXTURE.read_text().splitlines():
        r = json.loads(line)
        texts += [r["question"], r["expert"]]
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=400, special_tokens=["<pad>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<eos>", pad_token="<pad>")
    d = tmp_path_factory.mktemp("tiny")
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=tk.get_vocab_size() + 16, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=1024, eos_token_id=1, pad_token_id=0,
                      tie_word_embeddings=True)
    Qwen2ForCausalLM(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    loaded = AutoTokenizer.from_pretrained(d)
    assert len(loaded) <= cfg.vocab_size and loaded.pad_token_id == 0
    assert loaded.decode(loaded("What is 12 + 30?")["input_ids"]) == "What is 12 + 30?"
    return d
