"""mix_build._one_per_id: one row per id across labelling passes; a copy whose audio no longer matches is dropped and a
regional-reading pass wins where both copies are sound."""
import json
import os
import tempfile

import numpy as np
import soundfile as sf

import mix_build as B

d = tempfile.mkdtemp()
def wav(name, sec):
    p = os.path.join(d, name)
    sf.write(p, np.zeros(int(sec * 16000), dtype=np.int16), 16000, subtype="PCM_16")
    return p
good, short = wav("a.wav", 5.0), wav("b.wav", 2.0)
def put(name, rows):
    p = os.path.join(d, name)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p
p1 = put("rows_rec7x.jsonl", [{"id": "x", "text": "plain", "duration": 5.0, "audio_filepath": good},
                              {"id": "y", "text": "y1", "duration": 5.0, "audio_filepath": good},
                              {"id": "z", "text": "only", "duration": 5.0, "audio_filepath": good}])
p2 = put("rows_yvar1_regional.jsonl", [{"id": "x", "text": "regional", "duration": 5.0, "audio_filepath": good},
                                       {"id": "y", "text": "y2", "duration": 5.0, "audio_filepath": short}])
rows, st = B._one_per_id([p1, p2])
by = {r["id"]: r["text"] for r in rows}
assert sorted(by) == ["x", "y", "z"], by
assert by["x"] == "regional", by          # regional pass preferred when both are sound
assert by["y"] == "y1", by                # the copy whose audio no longer matches is dropped
assert st == {"dup_copies": 2, "dup_audio_mismatch": 1}, st
print("one_per_id ok")
