"""Owner listening test = ground truth for spoken year readings (owner 2026-10-13: "my listening test is definitive").
Merges the listening page's items + answers into gold/owner_year_readings.json (append/update by clip id; keeps
earlier answers). Audio stays local; this file holds only ids, source paths, the blanked text and the reading.

    python gold/save_owner_years.py <listen_dir>
"""

import json
import os
import sys
import time

D = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "owner_year_readings.json")
items = {i["id"]: i for i in json.load(open(f"{D}/items.json"))}
ans = json.load(open(f"{D}/answers.json"))
gold = json.load(open(OUT)) if os.path.exists(OUT) else {}
for k, a in ans.items():
    if k.endswith("_v") or k not in items:
        continue
    it = items[k]
    gold[it["src"] + "|" + str(it["value"])] = {
        "clip": k, "src": it["src"], "audio": it.get("vol", "earnings21/audio/4366893.wav"),
        "value": it["value"], "context": it["text"], "heard": a,
        "heard_value": ans.get(k + "_v"),
        "label_reading": it.get("label_reading") or it.get("ref_reading"), "v10_reading": it.get("v10_reading"),
        "saved": time.strftime("%Y-%m-%d %H:%M")}
json.dump(gold, open(OUT, "w"), indent=1, ensure_ascii=False)
print(len(gold), "gold readings ->", OUT)
