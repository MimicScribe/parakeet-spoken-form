"""mix_build._cap_top_years: no year value above the share; only rows whose every number is that year leave."""
import random

import mix_build as B

rows = [{"id": f"a{i}", "spans": [{"text": "2020", "kind": "int"}]} for i in range(40)]
rows += [{"id": f"m{i}", "spans": [{"text": "2020", "kind": "int"}, {"text": "5", "kind": "int"}]} for i in range(5)]
rows += [{"id": f"y{i}", "spans": [{"text": str(1940 + i % 60), "kind": "int"}]} for i in range(600)]
out, st = B._cap_top_years(rows, 0.03, random.Random(1))
ids = {x["id"] for x in out}
assert all(f"m{i}" in ids for i in range(5))                       # rows with another number stay
n2020 = sum(1 for x in out for sp in x["spans"] if sp["text"] == "2020")
tot = sum(1 for x in out for sp in x["spans"] if B._year_of(sp))
assert n2020 <= 0.03 * tot + 1, (n2020, tot)
assert st[2020]["before"] == 45 and st[2020]["rows_out"] > 0, st
assert len([x for x in out if x["id"].startswith("y")]) == 600      # other years untouched
print("year cap ok")
