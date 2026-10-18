"""15 s windows in the v4 listing: listed only when the build wrote them, tagged nonen_span (English: span), counted as
non-English / English parts of the share, renamed into a candidate dir; English held-out windows stay out."""
import os
import tempfile

import mix_build as MB
import modal_app as MA

R2 = MA.RUN2_LANGS
with tempfile.TemporaryDirectory() as d:
    MA.VOL = d
    os.makedirs(f"{d}/mix4final9")
    base = MA.retrain3_mix_v4(R2, anchor18=2, win_repeat=2, mix_dir="mix4final9")
    assert not any("win15" in p for p, _, _ in base)  # no part written: the listing is unchanged
    for name in ("win15_de.jsonl", "win15_it.jsonl", "en_win15.jsonl"):
        open(f"{d}/mix4final9/{name}", "w").close()
    on = MA.retrain3_mix_v4(R2, anchor18=2, win_repeat=2, mix_dir="mix4final9")
    assert on[:len(base)] == base  # appended, nothing else moves
    new = on[len(base):]
    assert new == [("mix4final9/win15_de.jsonl", 0, "nonen_span"), ("mix4final9/win15_it.jsonl", 0, "nonen_span"),
                   ("mix4final9/en_win15.jsonl", 0, "span")], new
    two = MA.retrain3_mix_v4(R2, anchor18=2, win_repeat=2, mix_dir="mix4final9", win15_repeat=2)[len(base):]
    assert two == new * 2
    assert MA.retrain3_mix_v4(R2, anchor18=2, win_repeat=2, mix_dir="mix4final9", win15_repeat=0) == base
    os.makedirs(f"{d}/mix4")  # the mix4 dir keeps its own names
    open(f"{d}/mix4/win15_fr.jsonl", "w").close()
    assert ("mix4/win15_fr.jsonl", 0, "nonen_span") in MA.retrain3_mix_v4(R2, anchor18=2, win_repeat=2)
# share bookkeeping: per-language windows are non-English, the English windows are English; both number-bearing
assert "mix4/win15_de.jsonl".startswith(MA._NON_EN_PARTS) and not "mix4/en_win15.jsonl".startswith(MA._NON_EN_PARTS)
assert MB.number_share_of("mix4final9/win15_de.jsonl") == 1.0 and MB.number_share_of("mix4final9/en_win15.jsonl") == 1.0
# English held-out rule (current lists): a held-out video by stream or parent id, a held-out plenary day by parent id
vids, days = {"V0_g_uKXKoM"}, {"20170327"}
assert MA.en_win15_held({"stream": "yt:V0_g_uKXKoM", "parents": []}, vids, days)
assert MA.en_win15_held({"stream": "x", "parents": ["en125_00000202_V0_g_uKXKoM_573_48_6_50_0"]}, vids, days)
assert MA.en_win15_held({"stream": "vox:a", "parents": ["20170327-1100-PLENARY_en_5"]}, vids, days)
assert not MA.en_win15_held({"stream": "yt:abc", "parents": ["20170328-1100-PLENARY_en_5", "en1_abc_1"]}, vids, days)
print("test_win15_listing: ok")
