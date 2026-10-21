"""Tests for the block-wise MCMC sampler (vsft.paper_sampler) and the paper-sample CLI.

Uses a deterministic word-level fake backend: each whitespace-separated word is one token.
"""

import json
import math

import pytest

import vsft.backends.base as base_mod
from vsft.backends.base import Backend, Generation
from vsft.boost import Example
from vsft.cli import main
from vsft.config import PaperSamplerConfig, load_config
from vsft.io import read_jsonl
from vsft.paper_sampler import PaperSampler

ASSISTANT = "\n\nAssistant: "


class WordTokenizer:
    """One token per whitespace-separated word, ids from an insertion-ordered vocab."""

    chat_template = None

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.inv: dict[int, str] = {}

    def _id(self, w: str) -> int:
        if w not in self.vocab:
            i = len(self.vocab)
            self.vocab[w] = i
            self.inv[i] = w
        return self.vocab[w]

    def encode(self, text: str) -> list[int]:
        return [self._id(w) for w in text.split()]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text)}

    def decode(self, ids, skip_special_tokens=False) -> str:
        return " ".join(self.inv[int(i)] for i in ids)


class WordBackend(Backend):
    """Scripted fake. `scripts` maps a question string to its response words (EOS after the
    last word) or to None (never emits EOS). The script is looked up by question, and the
    position in the script is the number of prefix words already in the prompt."""

    def __init__(self, scripts, logp=None, custom=None):
        self.tokenizer = WordTokenizer()
        self.scripts = scripts
        self.logp = logp or (lambda w: -1.0)
        self.custom = custom  # optional fn(call_index, question, prefix_words, max_tokens)
        self.gen_prompts: list[list[str]] = []
        self.gen_kwargs: list[dict] = []
        self.score_calls: list[tuple[list[str], list[str]]] = []

    def _question(self, prompt: str) -> str:
        hits = [q for q in self.scripts if q in prompt]
        assert len(hits) == 1, f"prompt matches {len(hits)} scripted questions"
        return hits[0]

    @staticmethod
    def _split(prompt: str) -> tuple[str, list[str]]:
        prefix = prompt.rsplit(ASSISTANT, 1)[1]
        return prefix, prefix.split()

    def generate(self, prompts, n, temperature, top_p, max_tokens):
        assert n == 1
        self.gen_prompts.append(list(prompts))
        self.gen_kwargs.append(dict(n=n, temperature=temperature, top_p=top_p,
                                    max_tokens=max_tokens))
        call_index = len(self.gen_prompts) - 1
        out = []
        for p in prompts:
            q = self._question(p)
            _, prefix_words = self._split(p)
            if self.custom is not None:
                text, fin = self.custom(call_index, q, prefix_words, max_tokens)
                out.append([Generation(text, fin)])
                continue
            script = self.scripts[q]
            pos = len(prefix_words)
            if script is None:
                words = [f"w{pos + i}" for i in range(max_tokens)]
                fin = False
            else:
                words = script[pos : pos + max_tokens]
                fin = pos + len(words) >= len(script)
            out.append([Generation(" ".join(words), fin)])
        return out

    def token_logprobs(self, prompts, responses):
        self.score_calls.append((list(prompts), list(responses)))
        return [[self.logp(w) for w in r.split()] for r in responses]

    @property
    def num_gen_calls(self) -> int:
        return len(self.gen_prompts)


def make_cfg(**kw) -> PaperSamplerConfig:
    base = dict(block_size=4, n_mcmc=0, max_new_tokens=64, seed=0)
    base.update(kw)
    return PaperSamplerConfig(**base)


def ex(qid: str, question: str, expert: str = "expert text") -> Example:
    return Example(id=qid, question=question, expert=expert)


# ------------------------------------------------------------------------------- blocks


def test_response_built_from_multiple_blocks():
    words = [f"t{i}" for i in range(10)]
    be = WordBackend({"Q-blocks": words})
    cfg = make_cfg(block_size=4, n_mcmc=0, max_new_tokens=64)
    (res,) = PaperSampler(be, cfg).run([ex("a", "Q-blocks")])
    assert res.num_blocks == math.ceil(10 / 4) == 3
    assert res.response == " ".join(words)
    assert res.status == "paper_sampled"
    # the proposal was asked for B tokens per block
    assert [k["max_tokens"] for k in be.gen_kwargs] == [4, 4, 4]


# ---------------------------------------------------------------------------- EOS / T


def test_eos_in_first_block_stops_generation():
    be = WordBackend({"Q-eos": ["a", "b"]})
    (res,) = PaperSampler(be, make_cfg(block_size=4)).run([ex("e", "Q-eos")])
    assert res.status == "paper_sampled"
    assert res.num_blocks == 1
    assert res.response == "a b"
    assert be.num_gen_calls == 1


def test_truncation_when_no_eos():
    be = WordBackend({"Q-trunc": None})
    cfg = make_cfg(block_size=4, n_mcmc=0, max_new_tokens=10)
    (res,) = PaperSampler(be, cfg).run([ex("t", "Q-trunc")])
    assert res.status == "paper_truncated"
    n = len(res.response.split())
    assert n <= 10
    assert n == 10  # no EOS, so the response runs to T exactly
    assert res.num_blocks == math.ceil(10 / 4) == 3


def test_eos_after_T_is_still_truncated():
    """BUG candidate: EOS emitted in a block that overruns T. The response is cut to T
    tokens (paper_sampler.py run(): `it.ids = it.ids[:T]`), but `it.finished` is kept, so
    the status is paper_sampled even though generation stopped because of T."""
    be = WordBackend({"Q-late-eos": ["a", "b", "c", "d", "e", "f", "g", "h"]})
    cfg = make_cfg(block_size=4, n_mcmc=0, max_new_tokens=6)
    (res,) = PaperSampler(be, cfg).run([ex("l", "Q-late-eos")])
    assert len(res.response.split()) <= 6  # holds today
    assert res.response == "a b c d e f"


def test_eos_after_T_reports_truncated_status():
    be = WordBackend({"Q-late-eos2": ["a", "b", "c", "d", "e", "f", "g", "h"]})
    cfg = make_cfg(block_size=4, n_mcmc=0, max_new_tokens=6)
    (res,) = PaperSampler(be, cfg).run([ex("l2", "Q-late-eos2")])
    assert res.status == "paper_truncated"


# ----------------------------------------------------------------- greedy acceptance


def _good_bad_logp(w: str) -> float:
    return 0.0 if w.startswith("good") else -1.0


def test_refinement_accepts_higher_likelihood_tails():
    """Proposal is all-bad (lp -4 for B=4). Every refinement tail is all-good (lp 0), so
    any cut yields a candidate at least as likely; some must be accepted."""

    def custom(call, q, prefix, max_tokens):
        if call == 0:  # initial proposal for the block
            return " ".join(["bad"] * max_tokens), True
        return " ".join(["good"] * max_tokens), True  # refinement tails

    be = WordBackend({"Q-greedy": None}, logp=_good_bad_logp, custom=custom)
    cfg = make_cfg(block_size=4, n_mcmc=8, max_new_tokens=4, seed=1)
    (res,) = PaperSampler(be, cfg).run([ex("g", "Q-greedy")])
    assert res.mcmc_steps == 8
    assert res.mcmc_accepted > 0
    assert res.final_mean_logprob is not None
    assert res.final_mean_logprob > -1.0  # initial proposal mean log-prob was -1.0


def test_refinement_never_accepts_worse_candidates():
    """Proposal is all-good (lp 0). Every refinement tail is all-bad, so each candidate
    has strictly lower likelihood (the cut is inside the block) and none is accepted."""

    def custom(call, q, prefix, max_tokens):
        if call == 0:
            return " ".join(["good"] * max_tokens), True
        return " ".join(["bad"] * max_tokens), True

    be = WordBackend({"Q-worse": None}, logp=_good_bad_logp, custom=custom)
    cfg = make_cfg(block_size=4, n_mcmc=6, max_new_tokens=4, seed=2)
    (res,) = PaperSampler(be, cfg).run([ex("w", "Q-worse")])
    assert res.mcmc_steps == 6
    assert res.mcmc_accepted == 0
    assert res.response == "good good good good"


# ------------------------------------------------------------------------ n_mcmc = 0


def test_n_mcmc_zero_means_no_refinement():
    be = WordBackend({"Q-nomcmc": None})
    cfg = make_cfg(block_size=4, n_mcmc=0, max_new_tokens=8)
    (res,) = PaperSampler(be, cfg).run([ex("n", "Q-nomcmc")])
    assert res.mcmc_steps == 0 and res.mcmc_accepted == 0
    assert be.num_gen_calls == 2  # one proposal per block, nothing else
    assert be.score_calls  # scoring still happens for the committed response


# ------------------------------------------------------------------------- batching


def test_batched_generate_covers_all_active_examples():
    scripts = {"Q-short": ["a", "b"], "Q-long1": None, "Q-long2": None}
    be = WordBackend(scripts)
    cfg = make_cfg(block_size=4, n_mcmc=2, max_new_tokens=8, seed=3)
    examples = [ex("s", "Q-short"), ex("l1", "Q-long1"), ex("l2", "Q-long2")]
    res = PaperSampler(be, cfg).run(examples)
    assert [r.id for r in res] == ["s", "l1", "l2"]
    # block 1: 1 proposal + 2 refinements, all 3 examples in each call.
    # block 2: the short example has finished, so 1 + 2 calls over the 2 remaining examples.
    sizes = [len(p) for p in be.gen_prompts]
    assert sizes == [3, 3, 3, 2, 2, 2]
    # each call carries the question of the examples it serves
    first = be.gen_prompts[0]
    assert "Q-short" in first[0] and "Q-long1" in first[1] and "Q-long2" in first[2]
    assert [r.status for r in res] == ["paper_sampled", "paper_truncated", "paper_truncated"]


def test_batched_run_without_refinement_one_call_per_block():
    be = WordBackend({"Q-b1": None, "Q-b2": None, "Q-b3": None})
    cfg = make_cfg(block_size=4, n_mcmc=0, max_new_tokens=8)
    PaperSampler(be, cfg).run([ex("1", "Q-b1"), ex("2", "Q-b2"), ex("3", "Q-b3")])
    assert [len(p) for p in be.gen_prompts] == [3, 3]


# -------------------------------------------------------------------- empty inputs


def test_empty_expert_does_not_crash_and_reports_none():
    be = WordBackend({"Q-noexpert": ["a", "b", "c"]})
    cfg = make_cfg(block_size=4, n_mcmc=2, max_new_tokens=8, seed=4)
    (res,) = PaperSampler(be, cfg).run([ex("x", "Q-noexpert", expert="")])
    assert res.response == "a b c"
    assert res.expert_mean_logprob is None  # -inf over zero tokens
    assert res.final_mean_logprob == pytest.approx(-1.0)


def test_empty_candidate_not_accepted_over_nonempty():
    """Refinement tails are empty EOS. A cut at the block start gives an empty candidate
    (log-lik -inf), which must lose to the non-empty block (lp -4). A cut inside the block
    gives a shorter prefix, which may win. The final response must never be empty."""

    def custom(call, q, prefix, max_tokens):
        if call == 0:
            return "a b c d", False
        return "", True  # empty tail, EOS

    be = WordBackend({"Q-emptyref": None}, custom=custom)
    cfg = make_cfg(block_size=4, n_mcmc=12, max_new_tokens=4, seed=5)
    (res,) = PaperSampler(be, cfg).run([ex("er", "Q-emptyref")])
    assert res.response != ""
    assert res.final_mean_logprob is not None
    assert res.num_blocks == 1


def test_empty_proposal_eos_gives_empty_response():
    be = WordBackend({"Q-empty": []})
    (res,) = PaperSampler(be, make_cfg(block_size=4, n_mcmc=0)).run([ex("z", "Q-empty")])
    assert res.response == ""
    assert res.status == "paper_sampled"
    assert res.final_mean_logprob is None


# --------------------------------------------------------------------------- CLI


@pytest.fixture
def paper_inputs(tmp_path):
    rows = [
        {"id": "p1", "question": "Q-cli-one", "expert": "one two"},
        {"id": "p2", "question": "Q-cli-two", "expert": "three"},
        {"id": "p3", "question": "Q-cli-three", "expert": "four five six"},
    ]
    inp = tmp_path / "in.jsonl"
    inp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return inp


@pytest.fixture
def cli_backends(monkeypatch):
    made: list[WordBackend] = []
    scripts = {"Q-cli-one": ["x", "y"], "Q-cli-two": ["z"], "Q-cli-three": None}

    def build(model_cfg, backend_cfg):
        be = WordBackend(scripts)
        made.append(be)
        return be

    monkeypatch.setattr(base_mod, "build_backend", build)
    import vsft.cli as cli_mod

    monkeypatch.setattr(cli_mod, "build_backend", lambda m, b: base_mod.build_backend(m, b))
    return made


def _paper_argv(inp, out):
    return ["paper-sample", "--input", str(inp), "--output", str(out),
            "--set", "paper.block_size=4", "--set", "paper.max_new_tokens=8",
            "--set", "paper.n_mcmc=2", "--set", "paper.examples_per_batch=2"]


def test_cli_paper_sample_writes_rows_and_summary(tmp_path, paper_inputs, cli_backends):
    out = tmp_path / "res" / "paper.jsonl"
    assert main(_paper_argv(paper_inputs, out)) == 0
    rows = list(read_jsonl(out))
    assert [r["id"] for r in rows] == ["p1", "p2", "p3"]
    assert {r["status"] for r in rows} == {"paper_sampled", "paper_truncated"}
    by_id = {r["id"]: r for r in rows}
    assert by_id["p1"]["status"] == "paper_sampled" and by_id["p1"]["response"] == "x y"
    assert by_id["p3"]["status"] == "paper_truncated"
    assert len(by_id["p3"]["response"].split()) == 8
    summ = json.loads((tmp_path / "res" / "paper.summary.json").read_text())
    assert summ["num_rows"] == 3
    assert summ["status"] == {"paper_sampled": 2, "paper_truncated": 1}
    assert 0.0 <= summ["mcmc_acceptance"] <= 1.0
    assert summ["config"]["paper"]["block_size"] == 4
    assert len(cli_backends) == 1


def test_cli_paper_sample_resume_does_not_resample(tmp_path, paper_inputs, cli_backends):
    out = tmp_path / "paper.jsonl"
    main(_paper_argv(paper_inputs, out))
    first_text = out.read_text()
    calls_before = cli_backends[0].num_gen_calls
    assert calls_before > 0
    assert main(_paper_argv(paper_inputs, out)) == 0
    assert out.read_text() == first_text
    assert cli_backends[0].num_gen_calls == calls_before
    assert len(cli_backends) == 1  # no backend built on the all-done run


# ------------------------------------------------------------------------ config


def test_config_rejects_zero_block_size():
    with pytest.raises(ValueError):
        load_config(None, ["paper.block_size=0"])


def test_config_paper_defaults():
    pc = load_config().paper
    assert pc.block_size == 32
    assert pc.n_mcmc == 10
    assert pc.max_new_tokens == 1856


def test_config_rejects_T_below_block_size():
    with pytest.raises(ValueError):
        load_config(None, ["paper.block_size=32", "paper.max_new_tokens=16"])

