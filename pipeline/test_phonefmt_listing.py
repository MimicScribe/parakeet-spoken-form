"""phonefmt switch in the v4 listing (same extra packages as test_run2_mix.py)."""
import os
import tempfile

import mix_build as MB
import modal_app as MA

R2 = MA.RUN2_LANGS
off = [p for p, _, _ in MA.retrain3_mix_v4(R2, anchor18=2, win_repeat=3)]
assert not any("d2_phonefmt" in p for p in off)  # off = the R5-b listing
with tempfile.TemporaryDirectory() as d:  # on, with a part written for one language only
    MA.VOL = d
    os.makedirs(f"{d}/mix4")
    open(f"{d}/mix4/d2_phonefmt_da.jsonl", "w").close()
    on = [p for p, _, _ in MA.retrain3_mix_v4(R2, anchor18=2, win_repeat=3, phonefmt=True)]
    assert on.count("mix4/d2_phonefmt_da.jsonl") == 2 and "mix4/d2anchor_da.jsonl" not in on
    assert on.count("mix4/d2anchor_et.jsonl") == 2
    open(f"{d}/mix4/d2phone_et.jsonl", "w").close()
    pa = [p for p, _, _ in MA.retrain3_mix_v4(R2, anchor18=2, win_repeat=3, phone_anchor=2)]
    assert pa.count("mix4/d2phone_et.jsonl") == 2 and "mix4/d2phone_da.jsonl" not in pa
    assert "mix4/d2phone_et.jsonl" not in on
assert MB.number_share_of("mix4/d2_phonefmt_da.jsonl") == 1.0 and MB.number_share_of("mix4/d2phone_et.jsonl") == 1.0
print("test_phonefmt_listing: ok")
