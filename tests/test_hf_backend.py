"""HFBackend on a tiny random local model (fast, offline)."""

import pytest
import torch

from vsft.backends.base import Generation


@pytest.fixture(scope="module")
def backend(tiny_model):
    from vsft.backends.hf import HFBackend
    from vsft.config import BackendConfig, ModelConfig

    return HFBackend(
        ModelConfig(name_or_path=str(tiny_model), dtype="float32", device="cpu",
                    max_model_len=64),
        BackendConfig(gen_batch_size=2, score_batch_size=3, seed=0),
    )


def manual_logprobs(backend, prompt, response):
    tok, model = backend.tokenizer, backend.model
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    r = tok(response, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        logp = torch.log_softmax(model(input_ids=torch.tensor([p + r])).logits[0].float(), -1)
    return [logp[len(p) - 1 + t, r[t]].item() for t in range(len(r))]


def test_token_logprobs_left_padding_matches_naive(backend):
    prompts = ["User: What is 12 + 30?", "User: Simplify", "User: A rectangle has width 3"]
    responses = ["We add the two numbers: 12 + 30 = 42.", "3/4", "The area is 15 ."]
    got = backend.token_logprobs(prompts, responses)
    for p, r, lp in zip(prompts, responses, got, strict=True):
        assert lp == pytest.approx(manual_logprobs(backend, p, r), abs=1e-4)
    # batch composition does not change results
    single = [backend.token_logprobs([p], [r])[0] for p, r in zip(prompts, responses, strict=True)]
    for a, b in zip(got, single, strict=True):
        assert a == pytest.approx(b, abs=1e-4)


def test_token_logprobs_empty_and_truncated(backend):
    long_resp = " ".join(["numbers"] * 100)
    out = backend.token_logprobs(["User: Q", "User: Q", ""], ["", long_resp, "x"])
    assert out[0] == [] and out[2] == []
    n_prompt = backend.count_tokens("User: Q")
    assert len(out[1]) == 64 - n_prompt  # cut at max_model_len


def test_generate_shapes_and_overlong(backend):
    long_prompt = " ".join(["numbers"] * 80)  # > max_model_len
    out = backend.generate(["User: Q", long_prompt, "User: area"], n=2, temperature=1.0,
                           top_p=1.0, max_tokens=5)
    assert len(out) == 3 and all(len(g) == 2 for g in out)
    assert out[1] == [Generation("", False), Generation("", False)]
    for g in out[0] + out[2]:
        assert isinstance(g.text, str) and isinstance(g.finished, bool)
    assert backend.generate([long_prompt], n=3, temperature=0.0, top_p=1.0,
                            max_tokens=5) == [[Generation("", False)] * 3]


def test_gen_config_ignores_checkpoint_defaults(backend, monkeypatch):
    backend.model.generation_config.top_k = 20
    backend.model.generation_config.repetition_penalty = 1.05
    seen = []
    orig = backend.model.generate

    def spy(*a, **kw):
        seen.append(kw["generation_config"])
        return orig(*a, **kw)

    monkeypatch.setattr(backend.model, "generate", spy)
    backend.generate(["User: Q"], n=2, temperature=0.7, top_p=0.9, max_tokens=4)
    backend.generate(["User: Q"], n=1, temperature=0.0, top_p=1.0, max_tokens=4)
    s, g = seen
    assert s.do_sample and s.top_k == 0 and s.repetition_penalty == 1.0
    assert s.temperature == 0.7 and s.top_p == 0.9 and s.num_return_sequences == 2
    assert s.max_new_tokens == 4 and set(s.eos_token_id) == backend.eos
    assert not g.do_sample and g.repetition_penalty == 1.0


class SpTok:
    """SentencePiece-like decode: a leading-space marker is dropped at sequence start."""

    all_special_tokens = ["<eos>", "<pad>"]
    pieces = {1: "▁Hello", 2: "▁world", 3: "!", 4: "<pad>", 5: "\n"}

    def decode(self, ids, skip_special_tokens=False):
        s = "".join(self.pieces[i] for i in ids)
        if skip_special_tokens:
            s = s.replace("<pad>", "")
        return s.replace("▁", " ").lstrip(" ")


def test_decode_continuation():
    from vsft.backends.hf import HFBackend

    be = HFBackend.__new__(HFBackend)
    be.tokenizer = SpTok()
    # naive tok.decode([2, 3]) would lose the space: "world!"
    assert be._decode_continuation(1, [2, 3]) == " world!"
    assert be._decode_continuation(5, [1]) == " Hello"
    assert be._decode_continuation(1, []) == ""
    assert be._decode_continuation(1, [2, 4, 3]) == " world!"  # special stripped
