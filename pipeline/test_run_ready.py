"""Run-readiness fixes (2026-10-17; no GPU, no Modal): python3 test_run_ready.py
windows / seam edge rule for compound number words, second-recognizer exclusion in windows2, build4 balance caps,
parent filters, seam top-up files, the generalised number share and per-language batch share."""
import inspect
import json
import os
import random
import re
import tempfile

import ml_balance as B
import ml_p3_core as C
import mix_build as MB


def row(text, times, num_chars, kind="int", rid="r", dur=None):
    return {"text": text, "word_times": times, "num_chars": num_chars, "spans": [{"kind": kind}] * len(num_chars),
            "duration": dur or times[-1][1] + 0.3, "id": rid}


# ---- 1b. compound number words (de / nl): a number word an edge cuts is labelled only when >= 0.9 of it is inside
assert C.num_keep_frac("de") == 0.9 and C.num_keep_frac("nl") == 0.9 and C.num_keep_frac("es") == 0.9  # all languages
# "Es waren dreihundertfünfundzwanzig Leute da." -- one long compound token
t = "Es waren dreihundertfünfundzwanzig Leute da heute."
times = [[0.2, 0.4], [0.5, 0.8], [1.0, 2.6], [2.8, 3.1], [3.2, 3.4], [3.5, 3.9]]
r = row(t, times, [[9, 34]])
_, words, total = C.stream_words([r])
num = [w for w in words if w[3]][0]
lo, hi = num[1] - C.LEAD, num[2] + C.TAIL
cut80 = lo + 0.8 * (hi - lo)  # 80% of the compound inside a window ending here: 2/3 rule labels it, 0.9 rule refuses
assert abs(C.inside_frac(num, 0.0, cut80) - 0.8) < 1e-9
lab_old = B.crop_label(words, 0.0, cut80)
assert lab_old is not None and "dreihundertfünfundzwanzig" in [w[0] for w in lab_old["inside"]]  # red-on-base
assert B.crop_label(words, 0.0, cut80, num_keep=0.9) is None
cut20 = lo + 0.2 * (hi - lo)  # 20% inside: old rule drops it (a "zwanzig"-sized piece is heard), new rule refuses
assert B.crop_label(words, cut20, total, num_keep=None) is not None
assert B.crop_label(words, cut20, total, num_keep=0.9) is None
cut95 = lo + 0.95 * (hi - lo)  # >= 0.9 inside: labelled whole; the mirror side (5%) leaves it out
assert [w[0] for w in B.crop_label(words, 0.0, cut95, num_keep=0.9)["inside"]][-1] == "dreihundertfünfundzwanzig"
assert "dreihundertfünfundzwanzig" not in [w[0] for w in B.crop_label(words, cut95, total, num_keep=0.9)["inside"]]
# a NON-number word keeps the 2/3 rule under num_keep
w_leute = words[3]
c = (w_leute[1] - C.LEAD) + 0.75 * ((w_leute[2] + C.TAIL) - (w_leute[1] - C.LEAD))
assert B.crop_label(words, 0.0, c, num_keep=0.9) is not None
assert C.compound_splits(words, 0.0, cut80) == 1 and C.compound_splits(words, 0.0, cut95) == 0

# build_windows: with num_keep no window labels a number word that sits 0.1-0.9 inside; without it some do
r2 = row("Danach waren es vierundzwanzig oder fünfunddreißig Leute heute.",
         [[0.1, 0.5], [0.6, 0.9], [1.0, 1.2], [1.4, 2.4], [2.5, 2.8], [3.0, 4.1], [4.2, 4.6], [4.7, 5.0]],
         [[16, 30], [36, 51]], rid="b")
stream = [r, r2, dict(r, id="c"), dict(r2, id="d")]
_, sw, _ = C.stream_words(stream)
n_old = n_new = 0
for seed in range(60):
    for w in C.build_windows(stream, random.Random(seed), win=3.0, stride=1.1, jitter=0.5, min_words=1):
        n_old += C.compound_splits(sw, w["a"], w["b"]) > 0
    for w in C.build_windows(stream, random.Random(seed), win=3.0, stride=1.1, jitter=0.5, min_words=1, num_keep=0.9):
        n_new += C.compound_splits(sw, w["a"], w["b"]) > 0
        lab = B.crop_label(sw, w["a"], w["b"], num_keep=0.9)  # crop_label and build_windows agree under the rule too
        assert lab is not None and " ".join(x[0] for x in lab["inside"]) == w["text"]
assert n_old > 0 and n_new == 0, (n_old, n_new)

# seam crops: no in-word cut positions for compound languages; the halves still partition the reading
ps_old = B.seam_crops([r, r2])
ps_new = B.seam_crops([r, r2], num_keep=0.9)
assert any(p["pos"].startswith("in") for p in ps_old)
assert ps_new and not any(p["pos"].startswith("in") for p in ps_new)
for p in ps_new:
    for side in ("left", "right"):
        h = p[side]
        assert not any(C.num_edge_ambiguous((x[0], 0, 0, True), x[1], 0.9) for x in h["edge_fracs"]
                       if x[0] in ("dreihundertfünfundzwanzig", "vierundzwanzig", "fünfunddreißig")), (p["pos"], h)

# ---- 1a. windows2 drops every second-recognizer span kind build4 drops (ext and ext_pick)
_p3 = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_p3.py")).read()  # ml_p3 needs modal: read it
win_src = _p3[_p3.index("def windows2_lang("):_p3.index("def windows2(")]
seam_src = _p3[_p3.index("def _seam_pass("):_p3.index("def seam_crops(")]
assert 'sp.get("by") == "ext"' not in win_src and "in EXT_BY" in win_src, "windows2 must exclude mix_build.EXT_BY spans"
assert B.EXT_BY == MB.EXT_BY == ("ext", "ext_pick")
keep_call = re.compile(r"num_keep=(?:C\.num_keep_frac|_num_keep)\(lang\)")  # _num_keep: + the WAVE2 languages
assert keep_call.search(win_src) and ("_num_keep(lang)" in seam_src and "num_keep=nk" in seam_src
                                      or keep_call.search(seam_src)), "windows2 / seam pass must use the compound rule"

# ---- 2c. balance caps
rows = []
for i in range(20):  # one video, 20 number rows: 13 stay (English YouTube: 20 stay, cap 30)
    rows.append({"id": f"de100_00000001_VVVVVVVVVVV_{i}_00_5_00", "caption": f"Es kostet {i} Euro, Teil {'abcdefghijklmnopqrst'[i]}.", "text": "x",
                 "duration": 1.0, "spans": [{"kind": "cur" if i < 3 else "int"}]})
for i in range(5):  # one caption template, five sessions: 3 stay
    rows.append({"id": f"2015010{i}-0900-PLENARY-1-de_x_{i}", "caption": f"Im Jahr {2000 + i} war es so.", "text": "y",
                 "duration": 1.0, "spans": [{"kind": "int"}]})
rows.append({"id": "tts_1", "caption": "Im Jahr 1.", "text": "z", "duration": 1.0, "spans": [{"kind": "int"}], "origin": "tts"})
kept, st = MB.balance_caps(rows, "de", random.Random(0))
assert st["group_out"] == 7 and st["template_out"] == 2 and st["rare_over_cap_kept"] == 0, st
assert sum(1 for x in kept if "VVVVVVVVVVV" in x["id"]) == 13
assert all(any(x["id"].endswith(f"_{i}_00_5_00") for x in kept) for i in range(3)), "rare-class rows are kept first"
assert any(x["id"] == "tts_1" for x in kept), "synthetic rows are not capped"
kept_en, st_en = MB.balance_caps(rows, "en", random.Random(0))
rows_r = [dict(x, spans=[{"kind": "phone"}]) if k < 15 else x for k, x in enumerate(rows[:20])]  # 15 rare in one video
kept_r, st_r = MB.balance_caps(rows_r, "de", random.Random(0))
assert all(x in kept_r for x in rows_r[:15]) and st_r["rare_over_cap_kept"] == 2 and st_r["group_out"] == 5, st_r
assert MB.template_key({"caption": "75."}) == "" and MB.template_key({"caption": "Es kostet 5 Euro."}) == "es kostet # euro."
assert st_en["group_out"] == 0
assert [x["id"] for x in kept] == [x["id"] for x in rows if x in kept]  # input order kept
# mirror of the data gate's group key
assert MB.group_id({"id": "pt100_00000486_dJjRBXP8rfo_100_26_2_14"}) == "yt:dJjRBXP8rfo"
assert MB.group_id({"id": "20160913-0900-PLENARY-21-es_20160913-23:07:5"}) == "ep:20160913-0900-PLENARY-21-es"


# ---- build4: parents decided by a second recognizer / held out now leave windows and seam; seam top-up files;
# a language without labelled rows keeps its D2 digit rows; balance caps
def fixture():
    vol = tempfile.mkdtemp()

    def wj(p, rs):
        os.makedirs(os.path.dirname(f"{vol}/{p}"), exist_ok=True)
        with open(f"{vol}/{p}", "w") as f:
            f.write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rs))

    wj("mix3/d2.jsonl", [{"id": "d2_de", "lang": "de", "text": "1990", "duration": 1.0},
                         {"id": "d2_pl", "lang": "pl", "text": "1990", "duration": 1.0}])
    wj("mix3/d2p.jsonl", [])
    wj("mix3/fleurs_train_multi.jsonl", [])
    base = "ml/p3v2/de"
    rs = [{"id": f"de100_0000000{i}_V{i}VVVVVVVVV_10_00_5_00", "text": f"Es waren zweitausend{z} Leute da, sagte er.",
           "caption": f"Es waren 20{10 + i} Leute da, sagte er {i}.", "duration": 5.0, "lang": "de", "src": "yodas",
           "num_chars": [[9, 9 + len("zweitausend" + z)]], "spans": [{"kind": "int", "by": "ext_pick" if i == 1 else "ctc"}]}
          for i, z in enumerate(["zehn", "elf", "zwölf"])]
    wj(f"{base}/rows.jsonl", rs)
    wj(f"{base}/judged_rows.json", [])
    open(f"{vol}/ml/p3v2/heldout_de.txt", "w").write(rs[2]["id"] + "\n")  # carved AFTER the windows were built
    open(f"{vol}/{base}/reject_ids.txt", "w").write("")
    win = lambda k, parents: {"id": f"de_w{k}", "text": "zweitausendzehn Leute da", "num_tokens": [True, False, False],  # noqa: E731
                              "duration": 3.0, "parents": parents, "audio_filepath": "w.wav"}
    wj(f"{base}/windows.jsonl", [win(0, [rs[0]["id"]]), win(1, [rs[0]["id"], rs[1]["id"]]), win(2, [rs[2]["id"]])])
    half = lambda pid, side, text, par: {"id": f"de_{pid}_{side}", "pair": pid, "side": side, "pos_class": "gap",  # noqa: E731
                                         "kind": "int", "text": text, "num_tokens": [True] * len(text.split()),
                                         "duration": 4.0, "parents": par, "lang": "de"}
    wj(f"{base}/seam.jsonl", [half("s0_0", "left", "Es waren", [rs[0]["id"]]), half("s0_0", "right", "zweitausendzehn", [rs[0]["id"]]),
                              half("s0_1", "left", "Es", [rs[1]["id"]]), half("s0_1", "right", "waren", [rs[1]["id"]])])
    # top-up file: its own s0_0 (a different cut) plus a copy of the main file's s0_0 (dropped as a duplicate)
    wj(f"{base}/seam_b.jsonl", [half("s0_0", "left", "Es waren zweitausendzehn", [rs[0]["id"]]),
                                half("s0_0", "right", "Leute da", [rs[0]["id"]]),
                                half("s0_1", "left", "Es waren", [rs[0]["id"]]), half("s0_1", "right", "zweitausendzehn", [rs[0]["id"]])])
    return vol


vol = fixture()
st = MB.build4(vol, ("de", "pl"), rows_h=99, win_h=1, allow_no_judge=True, seam_h=1.0, balance=True)
win = [json.loads(l) for l in open(f"{vol}/mix4/win_de.jsonl")]
assert [w["id"] for w in win] == ["de_w0"], win  # w1: an ext_pick parent; w2: a parent held out since
assert st["win_unsafe_parent_dropped"]["de"] == 2
seam = [json.loads(l) for l in open(f"{vol}/mix4/seam_de.jsonl")]
assert sorted({x["pair"] for x in seam}) == ["seam:s0_0", "seam_b:s0_0"], sorted({x["pair"] for x in seam})
d2 = [json.loads(l) for l in open(f"{vol}/mix4/d2.jsonl")]
assert [x["id"] for x in d2] == ["d2_pl"] and st["langs_without_rows"] == ["pl"]  # pl: no rows -> digits stay
assert "pl" not in st["langs"] and st["balance_caps"]["de"]["rows_in"] >= 1
p3 = [json.loads(l) for l in open(f"{vol}/mix4/p3_de.jsonl")]
assert all(x["id"] != "de100_00000002_V2VVVVVVVVV_10_00_5_00" for x in p3)  # held out now: eval only

# ---- number share: per-language parts of ANY language count; seam parts are measured; batch share per language
assert MB.number_share_of("mix4/p3_pl.jsonl") == 1.0 and MB.number_share_of("mix4/win_hr.jsonl") == 1.0
assert MB.number_share_of("mix4/d2anchor_da.jsonl") == 1.0 and MB.number_share_of("mix4/plain_pl.jsonl") == 0.0
assert MB.number_share_of("mix4/seam_ro.jsonl", [{"duration": 3.0, "num_tokens": [True]}, {"duration": 1.0, "num_tokens": [False]}]) == 0.75
sh = MB.mix_share(vol, [("mix4/p3_de.jsonl", 0, "nonen_span"), ("mix4/seam_de.jsonl", 0, "nonen_span"),
                        ("mix4/seam_de.jsonl", 0, "nonen_span")])
assert sh["lang_h"]["de"] > 0 and abs(sum(sh["lang_share"].values()) - 1.0) < 0.01
assert sh["parts"]["mix4/seam_de.jsonl"] == round(2 * 4 * 4.0 / 3600, 2)  # 4 halves x 4 s, listed twice
print("test_run_ready: ok")

# ---- seam pairs at the stream's ends (2026-10-17 crash: "deux mille vingt et un." in1_lo, a pair whose far edge moved
# past a reading word): sound spans are clipped to the stream, and every kept pair partitions its reading
fz = random.Random(5)
for trial in range(1500):
    toks, tms, t0 = [], [], fz.uniform(0, 0.3)
    for i in range(fz.randint(4, 24)):
        d = fz.uniform(0.06, 0.9)
        toks.append(fz.choice(["uno", "dos", "mil", "millones", "casa", "de", "la", "veinte"]))
        tms.append([round(t0, 3), round(t0 + d, 3)])
        t0 += d + fz.choice([0.0, 0.01, 0.03, 0.1, 0.4])
    a_ = fz.randint(0, len(toks) - 2)
    b_ = fz.randint(a_ + 1, min(len(toks), a_ + 6))
    st_ = len(" ".join(toks[:a_])) + (1 if a_ else 0)
    rr = {"text": " ".join(toks), "word_times": tms, "num_chars": [[st_, len(" ".join(toks[:b_]))]],
          "spans": [{"kind": "int"}], "duration": t0 + fz.choice([0.0, 0.05, 0.3]), "id": "f"}
    for nk in (None, 0.9):
        B.seam_crops([rr] if trial % 2 else [rr, dict(rr, id="g")], random.Random(trial), 0, num_keep=nk, strict=True)
# the first word of a stream (no audio before 0 to lose) counts whole under the 0.9 rule
w0 = ("uno", 0.004, 0.29, True, 0, 0)
assert C.inside_frac(w0, 0.0, 2.0) < 0.9 and C.inside_frac(w0, 0.0, 2.0, 0.0, 10.0) == 1.0
print("test_run_ready: stream-edge seam ok")

# ---- build4 dry dir + long-number seam pairs (seam_long.jsonl): same parent filters, own part, mix4 untouched
vol = fixture()
base_ = f"{vol}/ml/p3v2/de"
rows0 = [json.loads(l) for l in open(f"{base_}/rows.jsonl")]
long_ = lambda pid, par: [{"id": f"de_l{pid}_{s}", "pair": pid, "side": s, "pos_class": "gap", "kind": "phone", "pos": "in0_lo",  # noqa: E731
                           "text": "null sechs" if s == "left" else "eins zwei", "num_tokens": [True, True],
                           "duration": 3.0, "parents": [par], "lang": "de"} for s in ("left", "right")]
with open(f"{base_}/seam_long.jsonl", "w") as f:
    dupe = [{k: v for k, v in x.items() if k != "pos"} for x in long_("l2", rows0[0]["id"])]  # = a main pair's key
    for x in long_("l0", rows0[0]["id"]) + long_("l1", rows0[1]["id"]) + dupe:  # rows0[1]: an ext_pick parent -> out
        f.write(json.dumps(x) + "\n")
st = MB.build4(vol, ("de",), rows_h=99, win_h=1, allow_no_judge=True, seam_h=1.0, seam_long_h=1.0, mix_dir="mix4dry")
sl = [json.loads(l) for l in open(f"{vol}/mix4dry/seam_long_de.jsonl")]
assert [x["pair"] for x in sl] == ["l0", "l0"] and st["langs"]["de"]["seam_long_rows"] == 2, [x["pair"] for x in sl]
assert st["seam_long_dup_pairs"]["de"] == 1
assert not os.path.exists(f"{vol}/mix4/p3_de.jsonl") and os.path.exists(f"{vol}/mix4dry/p3_de.jsonl")
assert MB.number_share_of("mix4dry/seam_long_de.jsonl", sl) == 1.0
print("test_run_ready: dry dir + seam_long ok")

# ---- per-class ext cap (data_gate checks ext share per class, not per language)
rs = [{"id": f"x{i}", "spans": [{"kind": "dec", "text": "1,5", "by": "ext_pick" if i < 12 else "ctc"}]} for i in range(30)]
rs += [{"id": f"y{i}", "spans": [{"kind": "int", "text": "4500", "by": "ext" if i < 3 else "ctc"}]} for i in range(30)]
kept, xst = MB.cap_ext_per_class(rs, 0.25, random.Random(0))
dec = [sp for x in kept for sp in x["spans"] if sp["kind"] == "dec"]
assert sum(sp["by"] != "ctc" for sp in dec) <= 0.25 * len(dec), xst  # 12/30 = 0.40 before
assert xst["classes"]["dec"][0] == 0.4 and xst["classes"]["dec"][1] <= 0.25
assert len([x for x in kept if x["id"].startswith("y")]) == 30  # int_big at 0.10 is untouched
assert MB.ext_class({"kind": "int", "text": "2015"}) == "year" and MB.ext_class({"kind": "int", "text": "4.500"}) == "int_big"
print("test_run_ready: ext per class ok")

# ---- en_balance on a dry build's listing: mix4dry/ parts are non-English like mix4/ (they were counted as English)
v2 = tempfile.mkdtemp()
os.makedirs(f"{v2}/mix4dry", exist_ok=True); os.makedirs(f"{v2}/d1/en4_ok", exist_ok=True)
open(f"{v2}/mix4dry/p3_de.jsonl", "w").write(json.dumps({"duration": 3600.0, "text": "x", "lang": "de"}) + "\n")
open(f"{v2}/d1/en4_ok/yodas_en4_cap30_ycap.jsonl", "w").write(json.dumps({"duration": 3600.0, "text": "seventy five", "video": "v"}) + "\n")
st2 = MB.en_balance(v2, [("mix4dry/p3_de.jsonl", 0, "nonen_span"), ("d1/en4_ok/yodas_en4_cap30_ycap.jsonl", 0, "span")],
                    non_en=("mix4/p3_",))
assert st2["share_before"]["total_h"] == 1.0, st2["share_before"]
print("test_run_ready: dry en_balance ok")

# ---- free fill: rows a cap removed come back for a value still under its floor, never past the 3% top-year share
mk = lambda i, y: {"id": f"f{i}", "caption": f"Im Jahr {y} war es {i}.", "spans": [{"kind": "int", "text": str(y)}], "duration": 1.0}  # noqa: E731
base_rows = [mk(i, 2000 + i % 40) for i in range(40 * 45)]  # 45 each of 2000-2039
pool = [mk(10000 + i, 1995) for i in range(50)] + [mk(20000 + i, 2005) for i in range(50)]
out, fst = MB.fill_thin(base_rows, pool, random.Random(0))
assert sum(x["spans"][0]["text"] == "1995" for x in out) == MB.FILL_FLOOR["year_main"], fst  # 0 -> 42, then stops
assert sum(x["spans"][0]["text"] == "2005" for x in out) == 45  # not short: nothing added
out2, _ = MB.fill_thin([mk(i, 2010) for i in range(10)], [mk(100 + i, 1999) for i in range(5)], random.Random(0), top_share=0.03)
assert len(out2) == 10  # 1999 would hold > 3% of the year spans: not added
print("test_run_ready: fill ok")
