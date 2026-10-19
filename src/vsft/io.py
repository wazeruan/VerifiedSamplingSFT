"""JSONL helpers with resumable appends and atomic whole-file writes."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({e})") from e


def completed_ids(path: str | Path) -> set[str]:
    """IDs already present in an output JSONL. A torn final line (crash mid-write) is ignored."""
    p = Path(path)
    if not p.exists():
        return set()
    ids = set()
    with open(p) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "id" in row:
                ids.add(str(row["id"]))
    return ids


def repair_tail(path: str | Path) -> None:
    """Truncate a partially written last line so appends start on a clean line."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return
    with open(p, "rb+") as f:
        data = f.read()
        if data.endswith(b"\n"):
            return
        cut = data.rfind(b"\n") + 1
        try:  # a complete record that only lacks its newline is kept
            json.loads(data[cut:])
            f.write(b"\n")
        except ValueError:  # JSONDecodeError or a torn multi-byte UTF-8 character
            f.seek(cut)
            f.truncate()


def append_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json_atomic(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def write_jsonl_atomic(path: str | Path, rows: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
