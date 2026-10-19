"""Adversarial pack (advB): frozen readout splits, sentence planner, voice assignment, pilot sampling, and the matched TEST planner
(held-out values / constructions / voices plus a normal-speech control slice). Nothing here calls a network or a model.
Reuses the tier-A tooling (edge_a: assemble, voices, pilot sampler) with its own inventory.

    python adv_b.py freeze --out adv_b_split                                   # advBsplit_<lang>.json + manifest (BEFORE any synthesis)
    python adv_b.py plan --split adv_b_split --out plan --counts '{"en": {...}}'
    python adv_b.py voices --planned plan/planned_es.jsonl --refs refs.json --out voiced_es.jsonl [--chirp chirp.json]
    python adv_b.py testplan --split adv_b_split --out testplan --per-cell 5

Splits: per construction the value keys are cut in two (V_a is voiced in the pack; V_b never appears in a row or a window); constructions that
share a `split_group` (all styles of one phone-number kind) share ONE cut, so a number held out is held out of every style; about a fifth of the
constructions of each language x family are held out whole (C_b). Rows are written for the A halves only. Every row carries `family`
(phone / stumble / phrasing), `edge` [family], `ambig` (open label-decision keys), `construction`, `sub`, so a build can include / exclude each
family and an A/B test can run with and without the pack."""
import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys

import adv_b_content as K
import adv_b_frames as F
import adv_b_words as W
import edge_a as E

VERSION = "advB-1"
SEED = 1019
LANGS = ("en", "es", "fr", "de", "pt", "it", "nl")
FAMILIES = ("phone", "stumble", "phrasing")
TAG = "ttsADV1"

# constructions held out whole (C_b), chosen BEFORE any row was planned; a sibling reading stays in C_a
C_B = {
    "en": {"en.phone.tollfree_ohoh", "en.phone.uk_london_pairs", "en.phone.us_double", "en.stum.restart_scale", "en.stum.trail_resume",
           "en.phr.clock_mil", "en.phr.date_iso", "en.phr.power", "en.phr.ratio"},
    "es": {"es.phone.doble", "es.phone.tollfree_mx", "es.stum.trail_resume", "es.stum.restart_scale", "es.phr.clock_lat", "es.phr.fever", "es.phr.range_fast"},
    "fr": {"fr.phone.be_septante", "fr.phone.tollfree", "fr.stum.trail_resume", "fr.stum.digit_fix", "fr.phr.clock_24", "fr.phr.fever", "fr.phr.range_fast"},
    "de": {"de.phone.country"}, "pt": {"pt.phone.country"}, "it": {"it.phone.country"}, "nl": {"nl.phone.country"},
}
P_TWO = 0.2
P_LEAD, P_TAIL = 0.5, 0.4
WCAP = 5.0
NOCOMMA = E.NOCOMMA


def inventory(lang: str) -> list:
    return K.BUILD[lang]()


# ------------------------------------------------------------------------------------------------ splits
def split_values(c, seed: int = SEED) -> tuple:
    """(V_a, V_b): the keys are paired in a seeded shuffle and one of each pair goes to each half (both halves cover the range). A group of
    constructions with one value list shares its cut (keyed by the group id)."""
    vals = list(c.values)
    rng = random.Random(f"{seed}|advsplit|{c.split_group or c.id}")
    order = sorted(set(vals), key=lambda v: hashlib.md5(f"{seed}|{c.split_group or c.id}|{v}".encode()).hexdigest())
    va, vb = [], []
    for i in range(0, len(order), 2):
        pair = order[i:i + 2]
        if len(pair) == 2:
            rng.shuffle(pair)
            va.append(pair[0])
            vb.append(pair[1])
        else:
            (va if rng.random() < 0.5 else vb).append(pair[0])
    # a construction sees only the keys of its own list: restrict
    keep = set(vals)
    return [v for v in va if v in keep], [v for v in vb if v in keep]


def freeze(lang: str, seed: int = SEED) -> dict:
    cons = inventory(lang)
    ids = {c.id for c in cons}
    assert C_B[lang] <= ids, C_B[lang] - ids
    out = {"version": VERSION, "lang": lang, "seed": seed, "constructions": {}}
    for c in cons:
        va, vb = split_values(c, seed)
        out["constructions"][c.id] = {"family": c.family, "sub": c.sub, "side": "C_b" if c.id in C_B[lang] else "C_a", "n_values": len(c.values),
                                      "split_group": c.split_group, "ambig": list(c.ambig), "has_ctrl": c.ctrl is not None, "V_a": va, "V_b": vb}
    out["n_constructions"] = {f: sum(1 for c in cons if c.family == f) for f in FAMILIES}
    out["n_held_out"] = {f: sum(1 for c in cons if c.family == f and c.id in C_B[lang]) for f in FAMILIES}
    return out


canonical = E.canonical


def check_split(split: dict, man: dict, lang: str):
    assert hashlib.sha256(canonical(split)).hexdigest() == man["sha256"][lang], "split changed"


# ------------------------------------------------------------------------------------------------ sentences
def _caption_of(p) -> str:
    """The piece as it would be written: every number reading of the label replaced by its written text (digits)."""
    cap, cur, out = p.label, 0, ""
    for kind, written, reading in p.nums:
        j = p.label.find(reading, cur)
        out += p.label[cur:j] + written
        cur = j + len(reading)
    return out + p.label[cur:]


def _prep(p):
    from edge_a_content import Piece
    q = Piece(p.label, _caption_of(p), list(p.nums), tuple(p.ambig))
    q.synth = getattr(p, "synth", p.label)
    return q


def _synth_piece(p):
    from edge_a_content import Piece
    return Piece(p.synth, p.synth, [], ())


def build_row(lang, c, cores, keys, pieces, core, lead, tail, rng, tag, extra=None):
    ps = {s: _prep(p) for s, p in pieces.items()}
    r = E.assemble(lang, c.family, core, ps, rng, lead, tail)
    rs = E.assemble(lang, c.family, core, {s: _synth_piece(p) for s, p in ps.items()}, rng, lead, tail)
    template = f"{lead or ''}|{core}|{tail or ''}"
    ambig = set(c.ambig)
    for p in ps.values():
        ambig |= set(p.ambig)
    rid = f"{tag}_{lang}_{hashlib.md5((r['text'] + c.id).encode()).hexdigest()[:10]}"
    row = {"id": rid, "lang": lang, "cls": "edge", "register": "edge", "caption": r["caption"], "text": r["text"], "synth_text": rs["text"],
           "spans": r["spans"], "num_chars": r["num_chars"], "edge_spans": r["edge_spans"], "kind": sorted({s["kind"] for s in r["spans"]}),
           "split": "train", "template": template, "family": c.family, "edge": [c.family], "construction": c.id, "sub": c.sub,
           "constructions": [c.id] * len(keys), "values": keys, "ambig": sorted(ambig), "slots": [s["slot"] for s in r["spans"]], "advB": True}
    if extra:
        row.update(extra)
    return row


def _cores(lang, c, short):
    cores = list(F.FRAMES[lang][c.cat])
    return cores


def plan(lang: str, split: dict, counts: dict, seed: int = 19, tag: str = TAG, per_template: int = 3, min_each: int = 20) -> tuple:
    """Planned rows for the A halves: counts {family: sentences}. Every C_a construction gets min_each rows first (the pilot gate needs n per
    construction), the rest by weight x sqrt(values). Returns (rows, report)."""
    cons = {c.id: c for c in inventory(lang)}
    rng = random.Random(f"{tag}|{seed}|{lang}")
    used = collections.Counter()
    tpl_n = collections.Counter()
    seen, rows, drops = set(), [], collections.Counter()
    by_cons = collections.Counter()

    def one(c, fam):
        va = split["constructions"][c.id]["V_a"]
        low = min(used[(c.id, k)] for k in va)
        key = rng.choice([k for k in va if used[(c.id, k)] == low])
        pieces = {"X": c.make(key, random.Random(f"{key}|{used[(c.id, key)]}|{seed}|{c.id}"))}
        keys = [key]
        short = len(pieces["X"].label.split()) <= 5
        fr2 = F.FRAMES2.get(lang, {}).get(c.cat) or []
        if fr2 and short and c.family == "phrasing" and rng.random() < P_TWO and len(va) > 1:
            others = [k for k in va if k != key]
            k2 = rng.choice(others)
            pieces["Y"] = c.make(k2, random.Random(f"{k2}|{used[(c.id, k2)]}|{seed}|{c.id}"))
            keys.append(k2)
            core = rng.choice(fr2)
        else:
            core = rng.choice(F.FRAMES[lang][c.cat])
        lead = rng.choice(F.LEADS[lang]) if rng.random() < P_LEAD else None
        tail = rng.choice(F.TAILS[lang]) if rng.random() < P_TAIL else None
        if lead and tail and lead.lower() == tail.lower():
            tail = None
        row = build_row(lang, c, None, keys, pieces, core, lead, tail, rng, tag, {"half": "A"})
        return row, keys

    for fam in FAMILIES:
        n = counts.get(fam, 0)
        pool = [c for c in cons.values() if c.family == fam and split["constructions"][c.id]["side"] == "C_a"]
        if not pool or n <= 0:
            continue
        w = [c.weight * min(len(split["constructions"][c.id]["V_a"]) ** 0.5, WCAP) for c in pool]
        quota = collections.Counter()
        base = min(min_each, n // len(pool))
        for c in pool:
            quota[c.id] = base
        rest = n - base * len(pool)
        for _ in range(max(rest, 0)):
            quota[rng.choices(pool, w)[0].id] += 1
        for c in pool:
            got, stall = 0, 0
            while got < quota[c.id] and stall < 60 * max(quota[c.id], 1):
                row, keys = one(c, fam)
                if row["text"] in seen or re.search(r"[0-9{}]", row["text"]):
                    drops["duplicate_or_symbol"] += 1
                    stall += 1
                    continue
                if tpl_n[(c.id, row["template"])] >= per_template:
                    drops["template_cap"] += 1
                    stall += 1
                    continue
                seen.add(row["text"])
                tpl_n[(c.id, row["template"])] += 1
                for k in keys:
                    used[(c.id, k)] += 1
                by_cons[c.id] += 1
                stall = 0
                got += 1
                rows.append(row)
    rep = {"rows": len(rows), "drops": dict(drops), "by_construction": dict(by_cons), "values_used": len(used),
           "max_value_draws": max(used.values()) if used else 0}
    rng.shuffle(rows)
    return rows, rep


# ------------------------------------------------------------------------------------------------ TEST plan
def test_plan(lang: str, split: dict, per_cell: int = 5, seed: int = 23, tag: str = "advBt") -> tuple:
    """The held-out ADVERSARIAL TEST rows + the matched normal-speech CONTROL rows (same value, same carrier / lead / tail, the construction's
    standard reading `ctrl`; rendered with the same held-out voice by the caller).
    Quadrants per construction: C_a x V_b (value generalisation), C_b x V_a (construction generalisation), C_b x V_b (both). A control row is
    planned for every test row whose construction has a `ctrl`."""
    cons = {c.id: c for c in inventory(lang)}
    rng = random.Random(f"{tag}|{seed}|{lang}")
    rows, ctrl_rows, seen, used = [], [], set(), collections.Counter()
    for cid, c in sorted(cons.items()):
        sc = split["constructions"][cid]
        cells = [("C_a_V_b", sc["V_b"])] if sc["side"] == "C_a" else [("C_b_V_a", sc["V_a"]), ("C_b_V_b", sc["V_b"])]
        for quad, keys in cells:
            if not keys:
                continue
            k_shuf = list(keys)
            rng.shuffle(k_shuf)
            for key in k_shuf[:per_cell]:
                fr = F.FRAMES[lang][c.cat]
                piece = c.make(key, random.Random(f"t|{key}|{c.id}|{seed}"))
                core = rng.choice(fr)
                lead = rng.choice(F.LEADS[lang]) if rng.random() < P_LEAD else None
                tail = rng.choice(F.TAILS[lang]) if rng.random() < P_TAIL else None
                if lead and tail and lead.lower() == tail.lower():
                    tail = None
                row = build_row(lang, c, None, [key], {"X": piece}, core, lead, tail, rng, tag, {"half": "B", "bed": "adversarial", "quad": quad})
                if row["text"] in seen or re.search(r"[0-9{}]", row["text"]):
                    continue
                seen.add(row["text"])
                rows.append(row)
                if c.ctrl is not None:
                    cp = c.ctrl(key, random.Random(f"c|{key}|{c.id}|{seed}"))
                    crow = build_row(lang, c, None, [key], {"X": cp}, core, lead, tail, rng, tag + "c",
                                     {"half": "B", "bed": "control", "quad": quad, "control_of": row["id"], "family": c.family, "ambig": []})
                    crow["id"] = row["id"] + "_ctl"
                    crow["construction"] = c.id + "#ctrl"
                    crow["control_construction"] = c.id
                    ctrl_rows.append(crow)
    return rows, ctrl_rows


# ------------------------------------------------------------------------------------------------ gate classes + voices
add_gate_classes = E.add_gate_classes
voice_group = E.voice_group


def hold_out_refs(refs: list, per_group: int = 2) -> tuple:
    """(pack refs, test refs): per accent group the last `per_group` references (by a fixed hash) are test voices."""
    by = collections.defaultdict(list)
    for v in refs:
        by[v.get("accent")].append(v)
    pack, test = [], []
    for acc, vs in by.items():
        vs = sorted(vs, key=lambda v: hashlib.md5(("advB|" + v["spk"]).encode()).hexdigest())
        if len(vs) > per_group + 3:
            test += vs[-per_group:]
            pack += vs[:-per_group]
        else:
            pack += vs
    return pack, test


def assign(rows: list, lang: str, refs: list, cloud: list | None, chirp_share: float, seed: int = 19, vtag: str = "advB", engine: str = "qwen3",
           test: bool = False, min_voices: int = 4) -> dict:
    """Round robin over the clone references; chirp_share of the rows go to cloud voices. Pack: held-out persona excluded (assign_voices
    does it). Test (test=True): ONLY the held-out Chirp persona of the language is used for the cloud share. Normal rates (0.9-1.25)."""
    import ml_tts_symbol as S

    rng = random.Random(f"advB|voices|{seed}|{lang}|{'t' if test else 'p'}")
    order = list(range(len(rows)))
    rng.shuffle(order)
    nc = int(round(len(rows) * chirp_share)) if cloud else 0
    ch = [rows[i] for i in order[:nc]]
    cl = [rows[i] for i in order[nc:]]
    if ch:
        if test:
            from mix_build import HELDOUT_VOICES, heldout_persona
            persona = heldout_persona(HELDOUT_VOICES[lang]) if lang in HELDOUT_VOICES else "Charon"
            pool = [v for v in cloud if heldout_persona(v[0]) == persona]
            assert pool, (lang, persona)
            for k, r in enumerate(ch):
                v = pool[k % len(pool)]
                r.update(engine="presynth", engine_src="chirp3hd", voice=v[0], locale=v[1], voice_license="Google Cloud Text-to-Speech output",
                         voice_path=None, speaker=None, speed=rng.choice(S.RATES), pitch=1.0, presynth=f"/vol/ml/{vtag}/{lang}/presynth/{r['id']}.wav")
        else:
            S.assign_voices(ch, lang, vtag, None, cloud, "chirp", 4, S.RATES)
    if lang == "en":
        S.assign_voices_en2(cl, refs, vtag)
        for r in cl:
            r["speed"] = rng.choice(S.RATES)
    else:
        S.assign_voices(cl, lang, vtag, refs, None, engine, min_voices, S.RATES, 0.0)
        by = {v["spk"]: v for v in refs}
        for r in cl:
            r["accent"] = by[r["voice"].split(":")[-1]].get("accent")
    for r in rows:
        r["group"] = voice_group(r)
        r["voice_set"] = "test" if test else "pack"
    return dict(collections.Counter(r["group"] for r in rows))


pilot_pick = E.pilot_pick


# ------------------------------------------------------------------------------------------------ CLI
_write, _rows = E._write, E._rows


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--out", required=True)
    f.add_argument("--langs", default=",".join(LANGS))
    p = sub.add_parser("plan")
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--counts", required=True, help='json {lang: {family: n}}')
    p.add_argument("--seed", type=int, default=19)
    p.add_argument("--min-each", type=int, default=20)
    t = sub.add_parser("testplan")
    t.add_argument("--split", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--per-cell", type=int, default=5)
    t.add_argument("--langs", default=",".join(LANGS))
    a = ap.parse_args()
    if a.cmd == "freeze":
        os.makedirs(a.out, exist_ok=True)
        man = {"version": VERSION, "seed": SEED, "sha256": {}}
        for lang in a.langs.split(","):
            s = freeze(lang)
            path = os.path.join(a.out, f"advBsplit_{lang}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(s, fh, ensure_ascii=False, indent=1, sort_keys=True)
            man["sha256"][lang] = hashlib.sha256(canonical(s)).hexdigest()
            print(lang, s["n_constructions"], "held out", s["n_held_out"], man["sha256"][lang][:12])
        json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    elif a.cmd == "plan":
        counts = json.loads(a.counts)
        man = json.load(open(os.path.join(a.split, "manifest.json")))
        os.makedirs(a.out, exist_ok=True)
        for lang, cnt in counts.items():
            split = json.load(open(os.path.join(a.split, f"advBsplit_{lang}.json"), encoding="utf-8"))
            check_split(split, man, lang)
            rows, rep = plan(lang, split, cnt, a.seed, min_each=a.min_each)
            add_gate_classes(rows, lang)
            _write(os.path.join(a.out, f"planned_{lang}.jsonl"), rows)
            json.dump(rep, open(os.path.join(a.out, f"plan_report_{lang}.json"), "w"), indent=1)
            print(lang, rep["rows"], rep["drops"], "chars", sum(len(r["text"]) for r in rows))
    elif a.cmd == "testplan":
        man = json.load(open(os.path.join(a.split, "manifest.json")))
        os.makedirs(a.out, exist_ok=True)
        for lang in a.langs.split(","):
            split = json.load(open(os.path.join(a.split, f"advBsplit_{lang}.json"), encoding="utf-8"))
            check_split(split, man, lang)
            rows, ctl = test_plan(lang, split, a.per_cell)
            add_gate_classes(rows, lang)
            add_gate_classes(ctl, lang)
            _write(os.path.join(a.out, f"test_{lang}.jsonl"), rows)
            _write(os.path.join(a.out, f"control_{lang}.jsonl"), ctl)
            print(lang, "test", len(rows), "control", len(ctl), dict(collections.Counter(r["quad"] for r in rows)))


if __name__ == "__main__":
    main()
