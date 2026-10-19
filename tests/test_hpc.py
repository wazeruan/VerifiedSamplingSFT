import json

import pytest

from vsft.hpc import main


def make_run(d, losses=(1.0, 0.5), result=True, weights=True):
    (d / "final").mkdir(parents=True)
    (d / "final" / "config.json").write_text("{}")
    if weights:
        (d / "final" / "model.safetensors").write_bytes(b"x")
    if result:
        (d / "train_result.json").write_text("{}")
    (d / "train_history.jsonl").write_text(
        "".join(json.dumps({"step": i, "loss": x}) + "\n" for i, x in enumerate(losses))
    )
    return d


def test_check_train_ok(tmp_path, capsys):
    make_run(tmp_path)
    assert main(["check-train", "--dir", str(tmp_path)]) == 0
    assert "training output OK" in capsys.readouterr().out


@pytest.mark.parametrize(
    "kw,msg",
    [
        (dict(losses=(1.0, float("nan"))), "non-finite or missing loss"),
        (dict(losses=(float("inf"),)), "non-finite or missing loss"),
        (dict(result=False), "train_result.json missing"),
        (dict(weights=False), "no *.safetensors"),
    ],
)
def test_check_train_failures(tmp_path, capsys, kw, msg):
    make_run(tmp_path, **kw)
    assert main(["check-train", "--dir", str(tmp_path)]) == 3
    assert msg in capsys.readouterr().err


def test_check_train_empty_dir(tmp_path, capsys):
    assert main(["check-train", "--dir", str(tmp_path)]) == 3
    assert "config.json missing" in capsys.readouterr().err


def test_config_get(tmp_path, capsys):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  name_or_path: some/model\ntrain:\n  learning_rate: 2e-5\n")
    assert main(["config-get", "--config", str(p), "model.name_or_path"]) == 0
    assert capsys.readouterr().out.strip() == "some/model"
    main(["config-get", "--config", str(p), "--set", "train.lora_r=8", "train.lora_r"])
    assert capsys.readouterr().out.strip() == "8"
    main(["config-get", "--config", str(p), "train.learning_rate"])
    assert float(capsys.readouterr().out) == pytest.approx(2e-5)
    with pytest.raises(AttributeError):
        main(["config-get", "--config", str(p), "train.nope"])


def test_preflight_cpu_only(tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    need = tmp_path / "in.jsonl"
    need.write_text("{}\n")
    assert main(["preflight", "--run-dir", str(run), "--stage", "boost",
                 "--require-file", str(need)]) == 0
    started = json.loads((run / "status" / "boost.STARTED").read_text())
    assert started["stage"] == "boost" and started["state"] == "STARTED"
    assert json.loads((run / "status" / "boost.state").read_text())["state"] == "STARTED"
    # missing input -> nonzero and no marker
    run2 = tmp_path / "run2"
    run2.mkdir()
    assert main(["preflight", "--run-dir", str(run2), "--stage", "b",
                 "--require-file", str(tmp_path / "missing")]) == 2
    assert not (run2 / "status").exists()


def test_check_train_missing_loss_key(tmp_path, capsys):
    make_run(tmp_path)
    with open(tmp_path / "train_history.jsonl", "a") as f:
        f.write('{"step": 9}\n')
    assert main(["check-train", "--dir", str(tmp_path)]) == 3
    assert "train_history.jsonl contains a non-finite or missing loss" in capsys.readouterr().err
