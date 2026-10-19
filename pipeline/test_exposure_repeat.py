"""exposure.py on a part listed several times (stdlib only): python3 test_exposure_repeat.py
Red-on-base: distinct / distinct% counted every listing copy of a repeated row as its own distinct row."""
import json
import os
import tempfile

import exposure as E

tmp = tempfile.mkdtemp()
sha = "abc123def456"
# a part listed 3x (rows A, B -> 6 index lines) and a part listed once (row C)
idx = [{"i": i, "h": h, "d": d, "part": part, "tag": "t", "lang": "de", "kinds": ["pct"]}
       for i, (h, d, part) in enumerate([("hA", 200, "p"), ("hB", 300, "p")] * 3 + [("hC", 400, "q")])]
with open(f"{tmp}/row_index.{sha}.jsonl", "w") as f:
    f.write("".join(json.dumps(r) + "\n" for r in idx))
steps = [{"start": 0, "manifest": sha}]
steps += [{"step": k, "h": ["hA", "hA", "hA", "hC"], "d": [200, 200, 200, 400]} for k in range(1, 3)]  # A drawn 3x per step, never B
with open(f"{tmp}/seen_rows.jsonl", "w") as f:
    f.write("".join(json.dumps(r) + "\n" for r in steps))
e = E.exposure(tmp)
p, q = e["parts"]["p"], e["parts"]["q"]
assert p["listed_rows"] == 6 and p["listed_unique"] == 2 and abs(p["draws"] - 6) < 1e-9, p
assert abs(p["distinct"] - 1.0) < 1e-9, p  # row A is one distinct row (drawn 6 times), row B none: 1 of 2 unique rows, not 6 of 6 copies
assert q["listed_rows"] == 1 and q["listed_unique"] == 1 and q["distinct"] == 1.0 and abs(q["draws"] - 2) < 1e-9, q
c = e["cells"]["de/pct"]
assert c["listed_unique"] == 3 and abs(c["distinct"] - 2.0) < 1e-9, c
assert "50.0" in E.report(tmp)  # part p: 50% of its unique rows
print("test_exposure_repeat: ok")
