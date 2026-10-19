import json

import pytest

from vsft.io import (
    append_jsonl,
    completed_ids,
    read_jsonl,
    repair_tail,
    write_json_atomic,
    write_jsonl_atomic,
)


def test_read_jsonl_skips_blank_and_reports_line(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"id": 1}\n\n  \n{"id": 2}\n')
    assert [r["id"] for r in read_jsonl(p)] == [1, 2]
    p.write_text('{"id": 1}\n{bad\n')
    with pytest.raises(ValueError, match=":2:"):
        list(read_jsonl(p))


def test_completed_ids(tmp_path):
    p = tmp_path / "o.jsonl"
    assert completed_ids(p) == set()
    p.write_text('{"id": 1}\n{"x": 2}\n{"id": "b"}\n{"id": "tor')
    assert completed_ids(p) == {"1", "b"}
    p.write_text('[1, 2]\n"id"\n42\nnull\n\n{"id": 0}\n')
    assert completed_ids(p) == {"0"}


def test_repair_tail(tmp_path):
    p = tmp_path / "o.jsonl"
    repair_tail(p)  # missing file: no-op
    p.write_text("")
    repair_tail(p)
    assert p.read_text() == ""
    p.write_text('{"id": 1}\n{"id": 2}\n')
    repair_tail(p)
    assert p.read_text() == '{"id": 1}\n{"id": 2}\n'
    p.write_text('{"id": 1}\n{"id": 2, "res')
    repair_tail(p)
    assert p.read_text() == '{"id": 1}\n'
    p.write_text('{"id": 1, "torn')
    repair_tail(p)
    assert p.read_text() == ""
    # a complete record that only lacks its newline is kept (and terminated)
    p.write_text('{"id": 1}\n{"id": 2}')
    repair_tail(p)
    assert p.read_text() == '{"id": 1}\n{"id": 2}\n'
    append_jsonl(p, [{"id": 3}])
    assert [r["id"] for r in read_jsonl(p)] == [1, 2, 3]
    # torn multi-byte UTF-8 character
    p.write_bytes('{"id": 1}\n{"id": "é'.encode()[:-1])
    repair_tail(p)
    assert p.read_text() == '{"id": 1}\n'



def test_append_and_atomic(tmp_path):
    p = tmp_path / "sub" / "o.jsonl"
    write_jsonl_atomic(p, [{"id": 1, "t": "é"}])
    append_jsonl(p, [{"id": 2}, {"id": 3}])
    assert [r["id"] for r in read_jsonl(p)] == [1, 2, 3]
    assert "é" in p.read_text()
    j = tmp_path / "deep" / "s.json"
    write_json_atomic(j, {"a": 1})
    write_json_atomic(j, {"a": 2})
    assert json.loads(j.read_text()) == {"a": 2}
    assert sorted(x.name for x in j.parent.iterdir()) == ["s.json"]  # no temp leftovers
