"""Real-model integration test (HF backend, Qwen3.5-0.8B on CPU).

Run with: VSFT_RUN_SLOW=1 HF_HUB_OFFLINE=1 HF_HOME=<cache dir> uv run pytest -m slow
"""

import pytest

pytestmark = pytest.mark.slow

MODEL = "Qwen/Qwen3.5-0.8B"


@pytest.fixture(scope="module")
def backend():
    from vsft.backends.hf import HFBackend
    from vsft.config import BackendConfig, ModelConfig

    return HFBackend(
        ModelConfig(name_or_path=MODEL, dtype="float32", device="cpu", max_model_len=512),
        BackendConfig(gen_batch_size=2, score_batch_size=2, seed=0),
    )


def test_token_logprobs_matches_manual_forward(backend):
    import torch

    from vsft.prompts import student_messages

    prompts = [
        backend.render(student_messages("What is 12 + 30?")),
        backend.render(student_messages("Simplify 6/8 to lowest terms, please.")),
    ]
    responses = ["12 + 30 = 42, so \\boxed{42}.", "Divide by 2: \\boxed{3/4}"]
    got = backend.token_logprobs(prompts, responses)  # batched, right-padded
    tok, model = backend.tokenizer, backend.model
    for p, r, lp in zip(prompts, responses, got, strict=True):
        p_ids = tok(p, add_special_tokens=False)["input_ids"]
        r_ids = tok(r, add_special_tokens=False)["input_ids"]
        assert len(lp) == len(r_ids) == backend.count_tokens(r)
        ids = torch.tensor([p_ids + r_ids])
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0].float()
        logp = torch.log_softmax(logits, -1)
        manual = [logp[len(p_ids) - 1 + t, r_ids[t]].item() for t in range(len(r_ids))]
        assert lp == pytest.approx(manual, abs=1e-3)
        assert all(x <= 0 for x in lp)


def test_generate_returns_n_samples(backend):
    from vsft.prompts import student_messages

    prompts = [backend.render(student_messages(q)) for q in ["Say hi.", "What is 2+2? Be brief."]]
    out = backend.generate(prompts, n=3, temperature=1.0, top_p=0.95, max_tokens=8)
    assert len(out) == 2 and all(len(g) == 3 for g in out)
    for gens in out:
        for g in gens:
            assert isinstance(g.text, str) and isinstance(g.finished, bool)
            if not g.finished:
                assert backend.count_tokens(g.text) <= 8 + 1
    greedy = backend.generate(prompts, n=1, temperature=0.0, top_p=1.0, max_tokens=6)
    again = backend.generate(prompts, n=1, temperature=0.0, top_p=1.0, max_tokens=6)
    assert [g[0].text for g in greedy] == [g[0].text for g in again]


def test_generate_uses_fresh_generation_config(backend, monkeypatch):
    seen = []
    orig = backend.model.generate

    def spy(*a, **kw):
        seen.append(kw["generation_config"])
        return orig(*a, **kw)

    # the checkpoint generation_config samples with top_k=20, repetition_penalty=1.1
    ckpt = backend.model.generation_config
    assert ckpt.top_k == 20 and ckpt.repetition_penalty > 1.0
    monkeypatch.setattr(backend.model, "generate", spy)
    from vsft.prompts import student_messages

    backend.generate([backend.render(student_messages("Hi"))], n=2, temperature=0.8,
                     top_p=0.95, max_tokens=4)
    [gc] = seen
    assert gc.do_sample and gc.top_k == 0 and gc.repetition_penalty == 1.0
    assert gc.temperature == 0.8 and gc.top_p == 0.95
    assert set(gc.eos_token_id) == backend.eos


def test_completion_uses_im_end(backend):
    from vsft.train import IGNORE, tokenize_pair

    tok = backend.tokenizer
    item = tokenize_pair(tok, "What is 12 + 30?", "It is \\boxed{42}.", 256)
    labels = [x for x in item["labels"] if x != IGNORE]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    assert im_end in labels and labels.index(im_end) >= len(labels) - 2
    assert tok.convert_tokens_to_ids("<|endoftext|>") not in labels
    assert tok.decode(labels).startswith("It is \\boxed{42}.")
