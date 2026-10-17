"""Local tests for ml_balance + the build4 balance hooks (no GPU, no Modal): python3 test_ml_balance.py"""
import json
import os
import random
import tempfile

import ml_balance as B
import ml_p3_core as C


def row(text, times, num_chars, kind="year", rid="r", dur=None, src=None):
    return {"text": text, "word_times": times, "num_chars": num_chars, "spans": [{"kind": kind}] * len(num_chars),
            "duration": dur or times[-1][1] + 0.3, "id": rid, "src": src}


# ---- 1. seam crops ----------------------------------------------------------------------------------------------
# "Im Jahr zwei tausend zwanzig kam es." -- a three-word reading with clear gaps
t = "Im Jahr zwei tausend zwanzig kam es."
times = [[0.2, 0.4], [0.6, 0.9], [1.2, 1.5], [1.7, 2.2], [2.4, 2.9], [3.2, 3.5], [3.7, 3.9]]
r1 = row(t, times, [[8, 28]], rid="a")
r2 = row("Danach war alles anders und neu.", [[0.1, 0.5], [0.6, 0.8], [0.9, 1.3], [1.4, 1.9], [2.0, 2.2], [2.3, 2.6]],
         [], rid="b")
stream = [r1, r2]
_, words, total = C.stream_words(stream)

# crop_label == build_windows on every window build_windows emits (same rule, same labels)
for seed in range(30):
    rng = random.Random(seed)
    for w in C.build_windows(stream, rng, win=2.5, stride=1.3, jitter=0.4, min_words=1):
        lab = B.crop_label(words, w["a"], w["b"])
        assert lab is not None and " ".join(x[0] for x in lab["inside"]) == w["text"], (w, lab)
# a cut through the middle of a word: no label (build_windows skips such windows)
mid = (times[3][0] - C.LEAD + times[3][1] + C.TAIL) / 2
assert B.crop_label(words, 0.0, mid) is None

rd = B.readings(words, stream)
assert len(rd) == 1 and [words[j][0] for j in rd[0]["idx"]] == ["zwei", "tausend", "zwanzig"], rd
pos = dict(B.cut_positions(words, rd[0]))
assert set(pos) == {"before", "after", "gap0", "gap1", "in0_lo", "in0_hi", "in1_lo", "in1_hi", "in2_lo", "in2_hi"}, pos
pairs = B.seam_crops(stream)
by = {p["pos"]: p for p in pairs}
assert set(by) == set(pos), sorted(by)
for p in pairs:
    # mirrored halves partition the reading: left part + right part = the full reading, each word once
    joined = " ".join(x for x in (p["left"]["reading_part"], p["right"]["reading_part"]) if x)
    assert joined == "zwei tausend zwanzig", (p["pos"], p["left"]["reading_part"], p["right"]["reading_part"])
    assert p["left"]["b"] == p["right"]["a"]
assert by["before"]["left"]["reading_part"] == "" and by["before"]["right"]["reading_part"] == "zwei tausend zwanzig"
assert by["after"]["left"]["reading_part"] == "zwei tausend zwanzig" and by["after"]["right"]["reading_part"] == ""
assert by["gap0"]["left"]["reading_part"] == "zwei" and by["gap0"]["right"]["reading_part"] == "tausend zwanzig"
# the edge rule inside a word: 1/4 in -> the word belongs to the right half, 3/4 in -> to the left half
assert by["in1_lo"]["left"]["reading_part"] == "zwei" and by["in1_lo"]["right"]["reading_part"] == "tausend zwanzig"
assert by["in1_hi"]["left"]["reading_part"] == "zwei tausend" and by["in1_hi"]["right"]["reading_part"] == "zwanzig"
assert by["in1_hi"]["right"]["edge_fracs"][0][2] == "dropped" and by["in1_hi"]["left"]["edge_fracs"][-1][2] == "kept"
# full window geometry: each half is at most one 14.96 s window; a compound single-token reading still partitions
r3 = row("Es war zweitausendzwanzig sodann.", [[1.1, 1.3], [1.4, 1.6], [1.8, 3.0], [3.2, 3.4]], [[7, 25]], rid="c")
ps = B.seam_crops([r3, r2])
assert {p["pos"] for p in ps} == {"before", "in0_lo", "in0_hi", "after"}
for p in ps:
    assert p["left"]["b"] - p["left"]["a"] <= C.WIN + 1e-6 and p["right"]["b"] - p["right"]["a"] <= C.WIN + 1e-6
    assert " ".join(x for x in (p["left"]["reading_part"], p["right"]["reading_part"]) if x) == "zweitausendzwanzig"
assert B.pos_class("in0_lo", 1) == "in_word" and B.pos_class("gap0", 2) == "gap"
# no cut position next to a two-letter word ("cinquenta e quatro": the gap before / after "e", and "e" itself)
r4 = row("Foram cinquenta e quatro casos.", [[0.9, 1.2], [1.4, 2.0], [2.1, 2.15], [2.3, 2.7], [2.9, 3.3]], [[6, 24]],
         kind="int", rid="d")
_, w4, _ = C.stream_words([r4, r2])
p4 = dict(B.cut_positions(w4, B.readings(w4, [r4, r2])[0]))
assert not any(k.startswith(("gap", "in1")) for k in p4) and "in0_lo" in p4 and "in2_hi" in p4, sorted(p4)

# seam re-decode check: a head piece of the dropped word at a left half's cut edge is fine, the whole word is not
h = {"side": "left", "text": "Es war", "cut_edge_dropped": "zweitausendzwanzig", "edge_fracs": [["zweitausendzwanzig", 0.2, "dropped"]]}
assert B.seam_edges_ok(h, "es war zw") and B.seam_edges_ok(h, "es war z") and B.seam_edges_ok(h, "es war")
assert not B.seam_edges_ok(h, "es war zweitausendzwanzig") and not B.seam_edges_ok(h, "war")
assert not C.edges_match(h["text"], "es war z")  # the old rule rejects a one-letter piece
# a kept cut word may come out shortened at the cut edge (left half keeps "zweitausendzwanzig" at 3/4)
h = {"side": "left", "text": "Es war zweitausendzwanzig", "cut_edge_dropped": None, "cut_edge_kept": True,
     "edge_fracs": [["zweitausendzwanzig", 0.75, "kept"]]}
assert B.seam_edges_ok(h, "es war zweitausendzwan") and not B.seam_edges_ok(h, "es war zwei")
# right half: the tail piece of the dropped word at the start is fine
h = {"side": "right", "text": "so danach", "cut_edge_dropped": "zweitausendzwanzig", "cut_right": None, "edge_fracs": []}
assert B.seam_edges_ok(h, "zig so danach") and not B.seam_edges_ok(h, "zweitausendzwanzig so danach")
for p in B.seam_crops([r3, r2]):  # the generated halves carry the fields the check reads
    for side in ("left", "right"):
        assert "cut_edge_dropped" in p[side] and "cut_edge_kept" in p[side]

# take_seam: whole pairs only, every (position, kind) cell before a second pair of any cell
sr = []
for i, (pc, k) in enumerate([("before", "year"), ("before", "year"), ("gap", "int"), ("after", "year")]):
    for side in ("left", "right"):
        sr.append({"pair": f"p{i}", "side": side, "pos_class": pc, "kind": k, "duration": 15.0})
sr.append({"pair": "lonely", "side": "left", "pos_class": "gap", "kind": "int", "duration": 15.0})
got = B.take_seam(sr, 90 / 3600, random.Random(1))
assert len(got) == 6 and len({x["pair"] for x in got}) == 3 and "lonely" not in {x["pair"] for x in got}
assert {x["pos_class"] for x in got} == {"before", "gap", "after"}

# ---- 2. number-free + look-alikes ------------------------------------------------------------------------------
cases = [("Wir dürfen das nicht außer Acht lassen.", "de", True, "homophone"),
         ("Es gibt acht Länder.", "de", False, None),
         ("Wir sollten darauf achten, dass acht Leute kommen.", "de", False, None),
         ("Das ist eine gute Idee.", "de", True, "article"),
         ("Es waren drei Leute.", "de", False, None),
         ("Es waren 21 Leute.", "de", False, None),
         ("Tu sei sicuro?", "it", True, "homophone"),
         ("Ci sono sei gruppi politici.", "it", False, None),
         ("È uno dei problemi.", "it", True, "pronoun"),
         ("Il est sans doute là.", "fr", True, "homophone"),
         ("Il y a vingt ans.", "fr", False, None),
         ("Il y a un problème et une solution.", "fr", True, "article"),
         ("Cada uno de ellos lo sabe.", "es", True, "pronoun"),
         ("Treinta y uno.", "es", False, None),
         ("Me siento bien.", "es", True, "homophone"),
         ("Foi sem dúvida um erro.", "pt", True, "homophone"),
         ("Ik acht het nodig.", "nl", True, "homophone"),
         ("Het is een van de landen.", "nl", True, "pronoun"),
         ("Er zijn twee landen.", "nl", False, None),
         # compounds and hyphenated readings are number words too; everyday look-alike words are not
         ("Das hat zweieinhalb Jahre gedauert.", "de", False, None),
         ("Er kam zum zehnten Mal.", "de", False, None),
         ("Wir müssen die Regeln achten.", "de", True, "homophone"),
         ("Das war am achten Mai.", "de", False, None),
         ("Achtung, das ist noch einmal eine Einheit.", "de", True, "article"),
         ("Er zijn tienduizenden mensen.", "nl", False, None),
         ("We vieren de successen van Europa.", "nl", True, "homophone"),
         ("Il y a dix-sept ans.", "fr", False, None),
         ("Nel duemilaventi a settembre.", "it", False, None)]
for text, lang, free, tag in cases:
    assert B.number_free(text, lang) == free, (text, lang)
    if tag:
        assert tag in B.lookalikes(text, lang), (text, B.lookalikes(text, lang))
assert abs(B.plain_target_h(10.0, 0.6) - 15.0) < 1e-9 and B.plain_target_h(10.0, 0.6, 20.0) == 0.0

pool = ([{"text": "Il est sans doute là.", "duration": 3600.0, "src": "vox", "id": "h"}]
        + [{"text": f"Il y a un problème numéro {i}.", "duration": 3600.0, "src": "vox", "id": f"d{i}"} for i in range(2)]
        + [{"text": "Il y a un problème.", "duration": 3600.0, "src": "vox", "id": "a"},
           {"text": "Nous parlons ici.", "duration": 3600.0, "src": "yodas", "id": "p"},
           {"text": "Il y a vingt ans.", "duration": 3600.0, "src": "yodas", "id": "n"}])
got, st = B.pick_plain(pool, "fr", 3.0, random.Random(0))
assert [x["id"] for x in got] == ["h", "p"], got  # homophone first; the article row would put vox over the 60% cap
assert st["not_number_free"] == 3 and all(x["id"] != "n" for x in got)  # digits / "vingt" are number rows

# ---- 3. dedupe ------------------------------------------------------------------------------------------------
cap = "Wir haben im Jahr 2020 sehr viel erreicht, sagte sie."
train = [
    {"id": "20150609-0900-PLENARY-6-de_20150609-12:03:29_11", "text": "a", "caption": "Erste Fassung des Satzes hier.", "part": "rows"},
    {"id": "20150609-0900-PLENARY-7-de_20150609-12:03:29_11", "text": "b", "caption": "Erste Fassung des Satzes, hier!", "part": "rows_rec"},
    {"id": "de102_00000032_qyM1XkMLEco_419_22_10_00", "text": "c", "caption": "Ein ganz anderer Satz im Video eins.", "part": "rows"},
    {"id": "de102_00000032_qyM1XkMLEco_419_22_10_00_p0", "piece": [0.0, 6.0], "text": "d", "caption": "Ein ganz anderer.", "part": "rows_rec6"},
    {"id": "de100_00000001_AAAAAAAAAAA_10_00_5_00", "text": "e", "caption": "Abonniert den Kanal und lasst ein Like da!", "part": "rows"},
    {"id": "de100_00000002_BBBBBBBBBBB_10_00_5_00", "text": "f", "caption": "Abonniert den Kanal und lasst ein Like da", "part": "rows_yodas2"},
    {"id": "de100_00000003_CCCCCCCCCCC_10_00_5_00", "text": "g", "caption": "Abonniert den Kanal und lasst ein Like da, danke!", "part": "rows_yodas3"},
    {"id": "de100_00000004_DDDDDDDDDDD_10_00_5_00", "text": "h", "caption": cap, "part": "rows"},
    {"id": "20180101-0900-PLENARY-3_de_104", "text": "i", "caption": "Die Abstimmung findet morgen um zwölf Uhr statt und danach die Erklärungen.", "part": "granvox"},
    {"id": "de100_00000005_EEEEEEEEEEE_10_00_5_00", "text": "Danke.", "caption": "Danke.", "part": "rows"},
    {"id": "de100_00000006_FFFFFFFFFFF_10_00_5_00", "text": "Danke.", "caption": "Danke.", "part": "rows"},
]
prot = [{"id": "de100_00000009_ZZZZZZZZZZZ_99_00_5_00", "text": "x", "caption": cap},
        {"id": "20180101-0900-PLENARY-3-de_20180101-10:00:00_4", "text": "y",
         "caption": "Die Abstimmung findet morgen um zwölf Uhr statt."}]
kept, rep = B.dedupe(train, prot)
ids = [x["id"] for x in kept]
rm = rep["removed"]
assert len([i for i in ids if "PLENARY-6" in i or "PLENARY-7" in i]) == 1, ids  # one utterance, two agenda ids
assert rm.get("de102_00000032_qyM1XkMLEco_419_22_10_00_p0") == "dup_audio_yt" or rm.get("de102_00000032_qyM1XkMLEco_419_22_10_00") == "dup_audio_yt"
assert sum(i.startswith(("de100_00000001", "de100_00000002", "de100_00000003")) for i in ids) == 1, ids  # jingle
assert rm["de100_00000004_DDDDDDDDDDD_10_00_5_00"] == "heldout_text"  # a held-out caption leaves TRAINING
assert rm["20180101-0900-PLENARY-3_de_104"] == "heldout_session_text"  # Granary segment = held-out utterance
assert "de100_00000005_EEEEEEEEEEE_10_00_5_00" in ids and "de100_00000006_FFFFFFFFFFF_10_00_5_00" in ids  # short text
assert ids == [x["id"] for x in train if x["id"] in set(ids)]  # input order kept
# FLEURS: several speakers read one prompt -> the same text is not the same recording; a short phrase is no identity
fl = [{"id": "fleurs_1257_1", "text": "x", "caption": "Innerhalb eines Jahres kann eine Person zehn Kontakte haben."},
      {"id": "fleurs_1257_2", "text": "x", "caption": "Innerhalb eines Jahres kann eine Person zehn Kontakte haben."},
      {"id": "pt101_00000351_BNxhOf-1gFc_1029_04_1_86", "text": "cento e cinquenta reais", "caption": "150 reais"}]
k2, _ = B.dedupe(fl, [{"id": "pt100_00000130_zwW8Zwo1io8_3892_54_1_32", "text": "cento e cinquenta reais", "caption": "150 reais."}])
assert len(k2) == 3, k2
w, wr = B.dedupe([{"id": "w1", "text": cap}, {"id": "w2", "text": cap}], prot, within=False)
assert w == [] and wr["counts"]["heldout_text"] == 2
w, _ = B.dedupe([{"id": "w1", "text": "Ein Fenster mit langem Text hier."}] * 2, [], within=False)
# near-duplicate captions that differ only in a year are two utterances: the rare forward year must survive
long = ("Die Kommission hat im Ausschuss erklärt dass das Programm für die Mitgliedstaaten bis zum Jahr {} vollständig "
        "umgesetzt werden soll und dass die Finanzierung aus dem Haushalt der Union kommt wie es im Bericht steht")
yrs, _ = B.dedupe([{"id": "y1", "text": "x", "caption": long.format(2030)}, {"id": "y2", "text": "x", "caption": long.format(2050)}], [])
assert len(yrs) == 2, yrs
same, srep = B.dedupe([{"id": "y1", "text": "x", "caption": long.format(2030)},
                       {"id": "y2", "text": "x", "caption": long.format(2030).replace("Bericht", "Berichte")}], [])
assert len(same) == 1 and srep["counts"].get("dup_near_text") == 1, srep
assert len(w) == 2  # within=False: windows overlap by design
assert B.held_unit("20150609-0900-PLENARY-6-de_x") == "day:20150609" and B.held_unit("de102_00000032_qyM1XkMLEco_4_2_1_5") == "vid:qyM1XkMLEco"
assert B.session_key("20161208-0900-COMMITTEE-DROI_de_104") == "20161208-0900-COMMITTEE-DROI"


# ---- build4 hooks (a tiny fake volume) ----------------------------------------------------------------------
def build4_fixture():
    import mix_build as MB

    vol = tempfile.mkdtemp()

    def wj(p, rows):
        os.makedirs(os.path.dirname(f"{vol}/{p}"), exist_ok=True)
        with open(f"{vol}/{p}", "w") as f:
            f.write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))

    wj("mix3/d2.jsonl", [])
    wj("mix3/d2p.jsonl", [{"id": "20150609-0900-PLENARY-6-de_20150609-12:00:00_1", "text": "Sitzung am Held-Tag.", "duration": 2.0, "lang": "de"},
                          {"id": "20160101-0900-PLENARY-6-de_20160101-12:00:00_1", "text": "Gute Sitzung heute.", "duration": 2.0, "lang": "de"}])
    wj("mix3/fleurs_train_multi.jsonl", [])
    base = "ml/p3v2/de"
    rows = [{"id": f"de100_0000000{i}_V{i}VVVVVVVVV_10_00_5_00", "text": f"Im Jahr zweitausend{z} war es so weit, sagte er.",
             "caption": f"Im Jahr 20{10 + i} war es so weit, sagte er.", "duration": 5.0, "lang": "de", "src": "yodas",
             "num_chars": [[7, 7 + len("zweitausend" + z)]], "spans": [{"kind": "year"}]}
            for i, z in enumerate(["zehn", "elf", "zwölf"])]
    dup = dict(rows[0], id="20160505-0900-PLENARY-1-de_20160505-12:00:00_9")  # same caption, other pass
    wj(f"{base}/rows.jsonl", rows)
    wj(f"{base}/rows_rec.jsonl", [dup])
    for n in ("rows", "rows_rec"):
        wj(f"{base}/judged_{n}.json", [])
    held = dict(rows[2], id="de100_00000099_HHHHHHHHHHH_10_00_5_00")
    wj(f"{base}/val_vox.jsonl", [held])
    wj(f"{base}/judged_val_vox.json", [])
    open(f"{vol}/ml/p3v2/heldout_de.txt", "w").write("20150609-0900-PLENARY-6-de_20150609-11:00:00_1\n")
    open(f"{vol}/{base}/reject_ids.txt", "w").write("")
    wj(f"{base}/seam.jsonl", [{"id": f"de_seam_s0_{k}_{s}", "pair": f"s0_{k}", "side": s, "pos_class": pc, "kind": "year",
                               "text": "war es so weit" if s == "right" else "Im Jahr zweitausendzehn",
                               "num_tokens": [False, True] if s == "left" else [False] * 4, "duration": 4.0,
                               "parents": [rows[0]["id"]]} for k, pc in enumerate(["before", "gap"]) for s in ("left", "right")])
    wj("ml/plain/de/plain_rec.jsonl", [{"id": f"20170101-0900-PLENARY-2-de_20170101-12:00:0{i}_1", "text": t, "duration": 3600.0,
                                        "lang": "de", "src": "vox"}
                                       for i, t in enumerate(["Das dürfen wir nicht außer Acht lassen.", "Es gibt acht Länder.",
                                                              "Das ist eine gute Idee.", "Wir reden heute darüber."])])
    return MB, vol


MB, vol = build4_fixture()
s0 = MB.build4(vol, ("de",), rows_h=99, win_h=1, allow_no_judge=True)  # defaults: the old behaviour
p3_0 = [json.loads(l) for l in open(f"{vol}/mix4/p3_de.jsonl")]
assert len(p3_0) == 4 and not os.path.exists(f"{vol}/mix4/plain_de.jsonl") and not os.path.exists(f"{vol}/mix4/seam_de.jsonl")
assert "dedupe" not in s0
d2p0 = [json.loads(l) for l in open(f"{vol}/mix4/d2p.jsonl")]
MB, vol = build4_fixture()
s1 = MB.build4(vol, ("de",), rows_h=99, win_h=1, allow_no_judge=True, dedupe=True, seam_h=1.0, plain_share=0.6)
p3 = [json.loads(l) for l in open(f"{vol}/mix4/p3_de.jsonl")]
# the cross-pass duplicate goes, and so does the row whose caption is a validation (held-out) row's caption
assert len(p3) == 2, [x["id"] for x in p3]
assert s1["dedupe"]["de"]["rows"]["heldout_text"] == 1 and s1["dedupe"]["de"]["rows"]["dup_text"] == 1, s1["dedupe"]
ho = [json.loads(l) for l in open(f"{vol}/mix4/heldout_de.jsonl")]
assert [x["id"] for x in ho] == ["de100_00000099_HHHHHHHHHHH_10_00_5_00"]  # eval untouched
seam = [json.loads(l) for l in open(f"{vol}/mix4/seam_de.jsonl")]
assert len(seam) == 4 and s1["langs"]["de"]["seam_rows"] == 4
plain = [json.loads(l) for l in open(f"{vol}/mix4/plain_de.jsonl")]
assert plain and plain[0]["text"].startswith("Das dürfen") and all("acht Länder" not in x["text"] for x in plain)
assert s1["langs"]["de"]["number_free_share"] > 0.5, s1["langs"]["de"]
d2p = [json.loads(l) for l in open(f"{vol}/mix4/d2p.jsonl")]
assert len(d2p0) == 2 and [x["text"] for x in d2p] == ["Gute Sitzung heute."]  # held-out day leaves the anchor part
print("test_ml_balance OK")
