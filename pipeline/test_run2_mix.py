"""Run-2 mix plumbing (2026-10-17; no GPU, no volume). Needs the modal + numpy packages:
    uv run --no-project --with modal==1.6.1 --with numpy python test_run2_mix.py
langs threaded through the v4 listing, anchors = untrained languages only, windows + seam pairs x win_repeat, the English
balance copies, every listed part has a number share, the R7 smoke arms."""
from collections import Counter

import mix_build as MB
import modal_app as MA

R2 = MA.RUN2_LANGS
assert set(R2) == set(MA.WAVE1) | {"ru", "uk", "pl", "ro", "sv", "cs", "hu", "fi", "sk", "bg", "hr"}

# anchors: OTHER18 minus the trained languages (a trained language's digit rows would teach digits)
mix = MA.retrain3_mix_v4(R2, repeat=1, anchor18=2, win_repeat=3, en_bal=True, en_plain_h=80.0)
paths = Counter(p for p, _, _ in mix)
anch = {p[len("mix4/d2anchor_"):-6] for p in paths if p.startswith("mix4/d2anchor_")}
assert anch == set(MA.untrained(R2)) == {"da", "et", "el", "lv", "lt", "mt", "sl"}, anch
assert all(paths[f"mix4/d2anchor_{l}.jsonl"] == 2 for l in anch)
# spoken-form rows / synthetic rows once, windows 3x, for EVERY trained language (not just wave 1)
for l in R2:
    assert paths[f"mix4/p3_{l}.jsonl"] == 1 and paths[f"mix4/tts_{l}.jsonl"] == 1, l
    assert paths[f"mix4/win_{l}.jsonl"] == 3, l
# English: balance-capped copies listed instead of the year-capped sources, LibriSpeech at 80 h, windows once
for p in MB.EN_REAL_PARTS:
    assert paths[MB.en_bcap_path(p)] == 1 and p not in paths, p
assert dict((p, h) for p, h, _ in mix)["replay/lspc_train.jsonl"] == 80.0
assert paths["d1/en4_ok/win_en.jsonl"] == 1 and paths["d1/en4_ok/seam_en.jsonl"] == 1
# every part of the listing that holds numbers has a number share (mix_share would count a missing one as 0%)
free = {"replay/lspc_train.jsonl", "replay/icsi_train.jsonl", "mix3b/d1p_v.jsonl", "mix3b/d10p_v.jsonl",
        "mix4/d2p.jsonl", "mix4/fleurs_train_multi.jsonl"}
missing = [p for p in paths if p not in free and not p.startswith("mix4/plain_") and MB.number_share_of(p) == 0.0]
assert not missing, missing
# the R4-R6 listings are unchanged (same parts, same counts) when win_repeat is not given
old = Counter(p for p, _, _ in MA.retrain3_mix_v4(repeat=3, anchor18=2))
want = Counter()
for p, _, _ in MA.retrain3_mix_v3():
    want[{"mix3/d2.jsonl": "mix4/d2.jsonl", "mix3/d2p.jsonl": "mix4/d2p.jsonl",
          "mix3/fleurs_train_multi.jsonl": "mix4/fleurs_train_multi.jsonl", "mix3b/d1.jsonl": "mix3b/d1_ycap.jsonl",
          "mix3b/d10n.jsonl": "mix3b/d10n_ycap.jsonl"}.get(p, p)] += 1
for p, _, _ in MA.EN4_PARTS + MA.EN_WIN_PARTS:
    want[p] += 1
for l in MA.WAVE1:
    for k in ("p3", "win", "tts"):
        want[f"mix4/{k}_{l}.jsonl"] += 3
for l in MA.OTHER18:
    want[f"mix4/d2anchor_{l}.jsonl"] += 2
assert old == want, (old - want, want - old)
# the smoke arms: R5-b recipe on the run-2 mix (wave1_repeat 1, windows x3, English balance), R7-b = + the drop fix
a, b = MA.R3_ARMS["R7-a"], MA.R3_ARMS["R7-b"]
assert a["recipe"]["langs"] == ",".join(R2) and a["wave1_repeat"] == 1 and a["win_repeat"] == 2 and a["en_bal"]
assert a["en_free_repeat"] == 2 and a["en_plain_h"] >= 90.9
m2 = Counter(p for p, _, _ in MA.retrain3_mix_v4(R2, win_repeat=2, en_free_repeat=2))
assert all(m2[p] == 2 for p in MA.EN_FREE_PARTS) and m2["replay/lspc_train.jsonl"] == 1
assert a["lr"] == MA.R3_ARMS["R5-b"]["lr"] and a["wf_weight"] == MA.R3_ARMS["R5-b"]["wf_weight"]
assert {k: v for k, v in b.items() if k not in a} == {"suffix_anchor": True, "anchor_after": 1, "gate": 0.9,
                                                      "gate_human": 0.5}
assert MA._langs("") == MA.WAVE1 and MA._langs("de,pl") == ("de", "pl")
print("test_run2_mix: ok")
