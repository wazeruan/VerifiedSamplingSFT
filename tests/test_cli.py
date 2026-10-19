import json
from pathlib import Path

import pytest
from fakes import ACCEPT, FakeBackend, reject_at

import vsft.backends.base as base_mod
from vsft.cli import main
from vsft.io import read_jsonl

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_math.jsonl"


@pytest.fixture
def fake(monkeypatch):
    """Patch build_backend; the fake accepts '\\boxed{<ref>}' answers, rejects 't2' always."""
    refs = {r["question"]: r["answer"] for r in read_jsonl(FIXTURE)}
    made = []

    def propose(q, prefix, r):
        return [(f"Work.\nSo \\boxed{{{refs[q]}}}.", True), ("noise", True)]

    def judge(q, steps):
        return reject_at(1) if "rectangle" in q else ACCEPT

    def build(model_cfg, backend_cfg):
        be = FakeBackend(propose, judge, lambda q, r: -0.5 if r.startswith("Work") else -1.0)
        made.append(be)
        return be

    monkeypatch.setattr(base_mod, "build_backend", build)
    return made


def run(inp, out, *extra):
    return main(
        ["boost", "--input", str(inp), "--output", str(out),
         "--set", "boost.max_rounds=2", "--set", "boost.num_candidates=2", *extra]
    )


def test_boost_end_to_end_and_summary(tmp_path, fake):
    out = tmp_path / "res" / "out.jsonl"
    assert run(FIXTURE, out, "--set", "boost.examples_per_batch=2") == 0
    rows = list(read_jsonl(out))
    assert [r["id"] for r in rows] == ["t1", "t2", "t3"]
    status = {r["id"]: r["status"] for r in rows}
    assert status == {"t1": "accepted", "t2": "fallback_expert", "t3": "accepted"}
    assert rows[1]["num_rounds"] == 2 and rows[1]["response"] == rows[1]["expert"]
    # batched in groups of 2 -> two booster.boost calls, one backend
    assert len(fake) == 1
    summ = json.loads((tmp_path / "res" / "out.summary.json").read_text())
    assert summ["num_rows"] == 3
    assert summ["status"] == {"accepted": 2, "fallback_expert": 1}
    assert summ["accept_rate"] == pytest.approx(2 / 3)
    assert summ["mean_rounds"] == pytest.approx((1 + 2 + 1) / 3)
    assert summ["mean_logprob_gain_vs_expert"] == pytest.approx(0.5)
    assert summ["config"]["boost"]["max_rounds"] == 2


def test_resume_skips_done(tmp_path, fake):
    out = tmp_path / "out.jsonl"
    run(FIXTURE, out, "--limit", "1")
    assert [r["id"] for r in read_jsonl(out)] == ["t1"]
    run(FIXTURE, out)
    assert [r["id"] for r in read_jsonl(out)] == ["t1", "t2", "t3"]
    # second run never asked the model about t1
    asked = " ".join(p for call in fake[1].proposal_calls for p in call)
    assert "12 + 30" not in asked
    # third run: everything done -> no backend built at all
    run(FIXTURE, out)
    assert len(fake) == 2
    assert len(list(read_jsonl(out))) == 3


def test_resume_repairs_torn_line(tmp_path, fake):
    out = tmp_path / "out.jsonl"
    run(FIXTURE, out, "--limit", "1")
    good = out.read_text()
    out.write_text(good + '{"id": "t2", "question": "A rect')  # crash mid-write
    run(FIXTURE, out)
    rows = list(read_jsonl(out))  # must parse cleanly
    assert [r["id"] for r in rows] == ["t1", "t2", "t3"]
    assert out.read_text().startswith(good)


def test_duplicate_ids_rejected(tmp_path, fake):
    inp = tmp_path / "in.jsonl"
    inp.write_text('{"id": "a", "question": "q", "expert": "e"}\n' * 2)
    with pytest.raises(SystemExit, match="duplicate"):
        run(inp, tmp_path / "o.jsonl")


def test_bad_override_errors(tmp_path, fake):
    with pytest.raises(ValueError):
        run(FIXTURE, tmp_path / "o.jsonl", "--set", "boost.nope=1")


def test_parser_requires_args():
    with pytest.raises(SystemExit):
        main(["boost", "--input", "x"])


# ---------------------------------------------------------------- milestone 2 additions


def test_summary_counts_only_current_input(tmp_path, fake):
    out = tmp_path / "out.jsonl"
    # stale rows from another input in the same output file
    out.write_text(
        json.dumps({"id": "old1", "status": "dropped", "num_rounds": 6,
                    "final_mean_logprob": None, "expert_mean_logprob": -1.0}) + "\n"
    )
    run(FIXTURE, out, "--limit", "2")
    summ = json.loads((tmp_path / "out.summary.json").read_text())
    assert summ["num_rows"] == 2
    assert summ["status"] == {"accepted": 1, "fallback_expert": 1}
    assert "dropped" not in summ["status"]


def test_summary_on_empty_todo_creates_files(tmp_path, fake):
    inp = tmp_path / "in.jsonl"
    inp.write_text("")
    out = tmp_path / "o.jsonl"
    assert run(inp, out) == 0
    summ = json.loads((tmp_path / "o.summary.json").read_text())
    assert summ["num_rows"] == 0 and summ["mean_logprob_gain_vs_expert"] is None
    assert fake == []


def test_summary_with_unscorable_expert(tmp_path, monkeypatch):
    def build(model_cfg, backend_cfg):
        return FakeBackend(
            lambda q, p, r: [("ok \\boxed{42}", True)] * 2,
            lambda q, s: ACCEPT,
            lambda q, r: [] if r.startswith("We add") else -0.5,
        )

    monkeypatch.setattr(base_mod, "build_backend", build)
    out = tmp_path / "o.jsonl"
    run(FIXTURE, out, "--limit", "1")
    summ = json.loads((tmp_path / "o.summary.json").read_text())
    assert summ["mean_logprob_gain_vs_expert"] is None


def test_cli_train_dispatch(tmp_path, monkeypatch):
    import vsft.train as train_mod

    seen = {}

    def fake_train(cfg, data, out_dir, resume_from):
        seen.update(cfg=cfg, data=data, out=out_dir, resume=resume_from)
        return Path(out_dir) / "final"

    monkeypatch.setattr(train_mod, "train", fake_train)
    assert main(["train", "--data", str(FIXTURE), "--output-dir", str(tmp_path / "r"),
                 "--set", "train.lora_r=8", "--set", "train.source=expert"]) == 0
    assert seen["cfg"].train.lora_r == 8 and seen["cfg"].train.source == "expert"
    assert seen["resume"] is None
    with pytest.raises(SystemExit, match="trainer_state.json"):
        main(["train", "--data", str(FIXTURE), "--output-dir", str(tmp_path / "r"),
              "--resume-from", str(tmp_path)])
    (tmp_path / "st").mkdir()
    (tmp_path / "st" / "trainer_state.json").write_text("{}")
    main(["train", "--data", str(FIXTURE), "--output-dir", str(tmp_path / "r"),
          "--resume-from", str(tmp_path / "st")])
    assert Path(seen["resume"]) == tmp_path / "st"


def test_cli_eval(tmp_path, monkeypatch):
    refs = {r["question"]: r["answer"] for r in read_jsonl(FIXTURE)}
    built = []

    def build(model_cfg, backend_cfg):
        built.append(model_cfg.name_or_path)
        return FakeBackend(lambda q, p, r: [], lambda q, s: ACCEPT)

    # eval prompts are student prompts: answer correctly except for the rectangle question
    def gen(self, prompts, n, temperature, top_p, max_tokens):
        from fakes import parse_student

        from vsft.backends.base import Generation

        out = []
        for p in prompts:
            q = parse_student(p)
            out.append([Generation(f"\\boxed{{{'0' if 'rectangle' in q else refs[q]}}}", True)])
        return out

    monkeypatch.setattr(base_mod, "build_backend", build)
    monkeypatch.setattr(FakeBackend, "generate", gen)
    od = tmp_path / "ev"
    argv = ["eval", "--model", "my/model", "--data", str(FIXTURE), "--output-dir", str(od)]
    assert main(argv) == 0
    assert built == ["my/model"]
    summ = json.loads((od / "eval_summary.json").read_text())
    assert summ["model"] == "my/model"
    assert summ["accuracy"]["tiny_math"] == pytest.approx(2 / 3)
    assert (od / "tiny_math.metrics.json").exists()
    assert (od / "tiny_math.predictions.jsonl").exists()


def test_cli_prepare_unknown_dataset():
    with pytest.raises(SystemExit, match="unknown dataset"):
        main(["prepare", "nope"])


def test_cli_prepare_dispatch(tmp_path, monkeypatch):
    import vsft.data as data_mod

    calls = []
    monkeypatch.setitem(data_mod.PREPARERS, "gsm8k", lambda d: calls.append(("gsm8k", d)) or 1)
    monkeypatch.setitem(data_mod.PREPARERS, "math500", lambda d: calls.append(("m5", d)) or 2)
    assert main(["prepare", "gsm8k", "math500", "--out-dir", str(tmp_path)]) == 0
    assert calls == [("gsm8k", tmp_path), ("m5", tmp_path)]


# ------------------------------------------------------------ Slurm stop / STARTED hooks


def test_boost_stops_after_first_batch_and_resumes(tmp_path, monkeypatch):
    from vsft.cli import EXIT_STOPPED

    stop = tmp_path / "STOP"
    monkeypatch.setenv("VSFT_STOP_FILE", str(stop))
    refs = {r["question"]: r["answer"] for r in read_jsonl(FIXTURE)}
    asked = []

    def propose(q, prefix, r):
        asked.append(q)
        stop.write_text("")  # the job script asks to stop while batch 1 is running
        return [(f"So \\boxed{{{refs[q]}}}.", True)] * 2

    monkeypatch.setattr(
        base_mod, "build_backend", lambda m, b: FakeBackend(propose, lambda q, s: ACCEPT)
    )
    out = tmp_path / "out.jsonl"
    assert EXIT_STOPPED == 75
    assert run(FIXTURE, out, "--set", "boost.examples_per_batch=1") == 75
    assert [r["id"] for r in read_jsonl(out)] == ["t1"]  # batch 1 finished and was kept
    assert len(asked) == 1
    assert not (tmp_path / "out.summary.json").exists()
    # stop file still present: a rerun stops before doing anything new
    assert run(FIXTURE, out, "--set", "boost.examples_per_batch=1") == 75
    assert [r["id"] for r in read_jsonl(out)] == ["t1"]
    # cleared: the rerun resumes and completes
    stop.unlink()
    monkeypatch.setattr(
        base_mod, "build_backend",
        lambda m, b: FakeBackend(lambda q, p, r: [(f"So \\boxed{{{refs[q]}}}.", True)] * 2,
                                 lambda q, s: ACCEPT),
    )
    assert run(FIXTURE, out, "--set", "boost.examples_per_batch=1") == 0
    assert [r["id"] for r in read_jsonl(out)] == ["t1", "t2", "t3"]
    assert json.loads((tmp_path / "out.summary.json").read_text())["num_rows"] == 3


def test_boost_and_eval_notify_started(tmp_path, monkeypatch, fake):
    monkeypatch.setenv("VSFT_RUN_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("VSFT_STAGE", "boost")
    run(FIXTURE, tmp_path / "o.jsonl")
    m = json.loads((tmp_path / "runs" / "status" / "boost.STARTED").read_text())
    assert m["todo"] == 3
    # nothing left to do -> no backend is built, but STARTED is still recorded (todo=0)
    (tmp_path / "runs" / "status" / "boost.STARTED").unlink()
    n_backends = len(fake)
    run(FIXTURE, tmp_path / "o.jsonl")
    assert json.loads((tmp_path / "runs" / "status" / "boost.STARTED").read_text())["todo"] == 0
    assert len(fake) == n_backends

    monkeypatch.setenv("VSFT_STAGE", "eval")
    monkeypatch.setattr(FakeBackend, "generate",
                        lambda self, p, n, temperature, top_p, max_tokens:
                        [[__import__("vsft.backends.base", fromlist=["G"]).Generation("x", True)]
                         for _ in p])
    main(["eval", "--model", "m/x", "--data", str(FIXTURE), "--output-dir", str(tmp_path / "ev")])
    m = json.loads((tmp_path / "runs" / "status" / "eval.STARTED").read_text())
    assert m["model"] == "m/x"


def test_boost_stops_mid_batch_between_rounds(tmp_path, monkeypatch):
    stop = tmp_path / "STOP"
    record = tmp_path / "stopped.json"
    monkeypatch.setenv("VSFT_STOP_FILE", str(stop))
    monkeypatch.setenv("VSFT_STOPPED_RECORD", str(record))
    monkeypatch.setenv("SLURM_JOB_ID", "555")
    rounds = []

    def propose(q, prefix, r):
        rounds.append((q, r))
        if len(rounds) == 1:
            stop.write_text("")  # arrives while round 0 of the first batch is running
        return [("Work.\nSo \\boxed{0}.", True)] * 2

    # reject in round 0 so the batch needs a round 1: the stop is seen before it
    monkeypatch.setattr(
        base_mod, "build_backend", lambda m, b: FakeBackend(propose, lambda q, s: reject_at(1))
    )
    out = tmp_path / "out.jsonl"
    assert run(FIXTURE, out, "--set", "boost.examples_per_batch=3") == 75
    assert len(rounds) == 3 and {r for _, r in rounds} == {0}  # one round, then stopped
    assert not out.exists() or list(read_jsonl(out)) == []  # nothing emitted for the batch
    rec = json.loads(record.read_text())
    assert rec == {"job_id": "555", "written": 0, "total": 3}
    assert not (tmp_path / "out.summary.json").exists()

    stop.unlink()
    record.unlink()
    rounds.clear()
    rounds.append(("sentinel", -1))  # keep the hook from creating the stop file again
    assert run(FIXTURE, out, "--set", "boost.examples_per_batch=3") == 0
    assert [r["id"] for r in read_jsonl(out)] == ["t1", "t2", "t3"]
    assert not record.exists()


def test_boost_stop_record_counts_written_batches(tmp_path, monkeypatch):
    stop = tmp_path / "STOP"
    record = tmp_path / "stopped.json"
    monkeypatch.setenv("VSFT_STOP_FILE", str(stop))
    monkeypatch.setenv("VSFT_STOPPED_RECORD", str(record))
    calls = []

    def propose(q, prefix, r):
        calls.append(q)
        if len(calls) == 2:  # during batch 2 (round 0)
            stop.write_text("")
        return [("So \\boxed{0}.", True)] * 2

    monkeypatch.setattr(
        base_mod, "build_backend", lambda m, b: FakeBackend(propose, lambda q, s: reject_at(1))
    )
    out = tmp_path / "out.jsonl"
    assert run(FIXTURE, out, "--set", "boost.examples_per_batch=1") == 75
    assert [r["id"] for r in read_jsonl(out)] == ["t1"]
    assert json.loads(record.read_text()) == {"job_id": None, "written": 1, "total": 3}


def test_boost_stopped_raised_by_booster(tmp_path):
    from fakes import boost_cfg, verifier_cfg

    from vsft.boost import Booster, BoostStopped, Example

    be = FakeBackend(lambda q, p, r: [("x", True)], lambda q, s: reject_at(1))
    b = Booster(be, boost_cfg(num_candidates=1, max_rounds=3), verifier_cfg())
    exs = [Example("a", "q", "e", "1")]
    checks = iter([False, True])
    with pytest.raises(BoostStopped):
        b.boost(exs, should_stop=lambda: next(checks))
    assert len(be.proposal_calls) == 1
    # never consulted after every example is done
    be2 = FakeBackend(lambda q, p, r: [("\\boxed{1}", True)], lambda q, s: ACCEPT)
    b2 = Booster(be2, boost_cfg(num_candidates=1, max_rounds=3), verifier_cfg())
    seen = []
    [res] = b2.boost(exs, should_stop=lambda: seen.append(1) or False)
    assert res.status == "accepted" and len(seen) == 1


def test_main_installs_term_handler_only_in_stage(tmp_path, monkeypatch, fake):
    import signal

    import vsft.hpc as hpc

    run(FIXTURE, tmp_path / "a.jsonl")
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    monkeypatch.setenv("VSFT_STAGE", "boost")
    run(FIXTURE, tmp_path / "b.jsonl")
    assert signal.getsignal(signal.SIGTERM) is hpc.request_stop
    # (the autouse fixture restores the original handler in teardown)


def test_term_handler_per_subcommand(tmp_path, monkeypatch):
    """With VSFT_STAGE set, train/boost install the handler; eval/prepare keep SIG_DFL."""
    import signal

    import vsft.data as data_mod
    import vsft.hpc as hpc
    import vsft.train as train_mod
    from vsft.backends.base import Generation

    monkeypatch.setenv("VSFT_STAGE", "x")
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL

    monkeypatch.setattr(base_mod, "build_backend", lambda m, b: FakeBackend(
        lambda *a: [], lambda *a: ACCEPT))
    monkeypatch.setattr(FakeBackend, "generate", lambda self, p, n, temperature, top_p,
                        max_tokens: [[Generation("\\boxed{1}", True)] for _ in p])
    assert main(["eval", "--data", str(FIXTURE), "--output-dir", str(tmp_path / "ev")]) == 0
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL

    monkeypatch.setitem(data_mod.PREPARERS, "gsm8k", lambda d: 0)
    assert main(["prepare", "gsm8k", "--out-dir", str(tmp_path)]) == 0
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL

    handler_during_train = []
    monkeypatch.setattr(train_mod, "train", lambda cfg, d, o, r: handler_during_train.append(
        signal.getsignal(signal.SIGTERM)) or Path(o) / "final")
    assert main(["train", "--data", str(FIXTURE), "--output-dir", str(tmp_path / "t")]) == 0
    assert handler_during_train == [hpc.request_stop]
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    monkeypatch.setattr(base_mod, "build_backend", lambda m, b: FakeBackend(
        lambda q, p, r: [("\\boxed{1}", True)] * 2, lambda *a: ACCEPT))
    assert run(FIXTURE, tmp_path / "b.jsonl") == 0
    assert signal.getsignal(signal.SIGTERM) is hpc.request_stop
