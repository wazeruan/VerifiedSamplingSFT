#!/usr/bin/env python3
"""Stdlib-only reader for the YAML subset used by configs/*.yaml.

Works on login nodes without the project venv (Python >= 3.6). Supports two-level
`section:` / `  key: value` files, quoted or bare scalars, flow lists `[a, b]` and block
lists (`  key:` followed by `    - item`).

    cfg.py FILE section.key [DEFAULT ...]

Prints a scalar as is, and a list one item per line. If the key is missing, the DEFAULT
arguments are printed instead (one per line), so callers can fall back to old behaviour
for configs written before the key existed.
"""

import re
import sys


def _strip_comment(line):
    """Drop a trailing `# comment`: a '#' starts one only outside quotes and after whitespace
    (or at the start), so `run#1`, `"a #b"` and URLs with fragments survive."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i]
    return line


def _split_flow(inner):
    """Split `a, "b, c", d` on commas outside quotes."""
    items, cur, quote = [], [], None
    for ch in inner:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            cur.append(ch)
        elif ch == ",":
            items.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    items.append("".join(cur))
    return [x for x in items if x.strip()]


def _scalar(v):
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        return [_scalar(x) for x in _split_flow(v[1:-1])]
    if len(v) >= 2 and v[0] in "'\"" and v[-1] == v[0]:
        return v[1:-1]
    return v


def parse(text):
    root, section, open_list = {}, None, None
    for raw in text.splitlines():
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        m = re.match(r"^(\w+):\s*$", line)
        if m:
            section, open_list = m.group(1), None
            root[section] = {}
            continue
        m = re.match(r"^\s+-\s+(.*)$", line)
        if m and open_list is not None:
            root[open_list[0]][open_list[1]].append(_scalar(m.group(1)))
            continue
        m = re.match(r"^\s+(\w+):\s*(.*)$", line)
        if m and section is not None:
            key, val = m.group(1), m.group(2).strip()
            if val == "":
                root[section][key] = []
                open_list = (section, key)
            else:
                root[section][key] = _scalar(val)
                open_list = None
    return root


def get(path, dotted):
    section, _, key = dotted.partition(".")
    with open(path) as f:
        return parse(f.read()).get(section, {}).get(key)


def main(argv):
    if len(argv) < 3:
        raise SystemExit(__doc__)
    value = get(argv[1], argv[2])
    if value is None:
        value = argv[3:]
    if isinstance(value, list):
        print("\n".join(str(x) for x in value))
    else:
        print(value)


if __name__ == "__main__":
    main(sys.argv)
