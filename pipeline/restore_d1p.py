"""R3.17 F1 for D1p: restore caption-dropped fillers/repeats from v10pt's decode of the same rows.

    python restore_d1p.py <plain3.jsonl> <plain_all_v10.json> <out plain3_ok.jsonl>
"""
import collections
import json
import sys

from restore_fillers import merge

rows = [json.loads(line) for line in open(sys.argv[1])]
dec = json.load(open(sys.argv[2]))
st = collections.Counter()
with open(sys.argv[3], "w") as f:
    for x in rows:
        h = dec.get(x["audio_filepath"])
        if h is None:
            st["no_decode"] += 1
        else:
            t, n = merge(x["text"], h)
            if n:
                st["rows_changed"] += 1
                st["words"] += n
                x = {**x, "text": t, "text_caption": x["text"]}
        st["rows"] += 1
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
print(dict(st))
