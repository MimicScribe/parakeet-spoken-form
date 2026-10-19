"""Edge-case pack, tier A (financial, colloquial, disfluency; en / es / fr): frozen readout splits, sentence planner,
voice assignment, pilot sampling. Nothing here calls a network or a model.

    python edge_a.py freeze --out edge_a_split                       # ec1split_<lang>.json + manifest (BEFORE any synthesis)
    python edge_a.py plan --split edge_a_split --out plan --counts '{"fin": 700, "col": 420, "dis": 120}'
    python edge_a.py voices --planned plan/planned_es.jsonl --refs refs.json --out voiced_es.jsonl [--chirp chirp.json]
    python edge_a.py pilot --voiced voiced_es.jsonl --per-group 30 --out pilot_es.jsonl

Splits (the design's section 6): per construction the value keys are cut in two (V_a is voiced, V_b never appears in a
row or a window); per language and family about a fifth of the constructions are held out whole (C_b). Rows are written
for the A halves only (a C_a construction with a V_a value). `ambig` lists the open label-decision keys a row depends on (so it can be relabelled later).
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys

import edge_a_content as K
import edge_a_frames as F

VERSION = "ec1-a1"
SEED = 1019
LANGS = ("en", "es", "fr")

# constructions held out whole (C_b), chosen BEFORE any row was planned: a sibling form stays in C_a so the readout asks
# "did the model learn the idea, not only the form" (bips vs bps, a billion two vs one point two bil, ...)
C_B = {
    "en": {"en.fin.bips", "en.fin.bil_mil", "en.fin.range_pct", "en.col.grand", "en.col.something_odd", "en.dis.number_pause"},
    "es": {"es.fin.range_pct", "es.fin.index_name", "es.fin.q_code", "es.col.cuantificador", "es.dis.relleno"},
    "fr": {"fr.fin.fourchette_pct", "fr.fin.indice_nom", "fr.fin.t_code", "fr.col.et_quelques", "fr.dis.remplissage"},
}
P_LEAD, P_TAIL, P_TWO = 0.55, 0.45, 0.25
NOCOMMA = {"I think", "Creo que", "Je crois que"}      # leads that run into the core without a comma
ELIDE_DE = re.compile(r"\bde (?=[aeiouàâéèêëîïôùûAEIOUÀÉ])")   # fr: de + vowel -> d'
WCAP = 5.0


def inventory(lang: str) -> list:
    return K.BUILD[lang]()


# ------------------------------------------------------------------------------------------------ splits
def split_values(c, seed: int = SEED) -> tuple:
    """(V_a, V_b): consecutive value keys are paired and one of each pair goes to each half (both halves cover the range);
    constructions whose keys are surface forms (split_values False) keep everything in V_a."""
    vals = list(c.values)
    if not c.split_values:
        return vals, []
    rng = random.Random(f"{seed}|split|{c.id}")
    va, vb = [], []
    for i in range(0, len(vals), 2):
        pair = vals[i:i + 2]
        if len(pair) == 2:
            rng.shuffle(pair)
            va.append(pair[0])
            vb.append(pair[1])
        else:
            (va if rng.random() < 0.5 else vb).append(pair[0])
    return va, vb


def freeze(lang: str, seed: int = SEED) -> dict:
    cons = inventory(lang)
    ids = {c.id for c in cons}
    assert C_B[lang] <= ids, C_B[lang] - ids
    out = {"version": VERSION, "lang": lang, "seed": seed, "constructions": {}}
    for c in cons:
        va, vb = split_values(c, seed)
        out["constructions"][c.id] = {"family": c.family, "side": "C_b" if c.id in C_B[lang] else "C_a", "n_values": len(c.values),
                                      "split_values": c.split_values, "ambig": list(c.ambig), "V_a": va, "V_b": vb}
    out["n_constructions"] = {f: sum(1 for c in cons if c.family == f) for f in ("fin", "col", "dis")}
    out["n_held_out"] = {f: sum(1 for c in cons if c.family == f and c.id in C_B[lang]) for f in ("fin", "col", "dis")}
    return out


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def side_of(split: dict, cons_id: str, key: str) -> str:
    """'A' (a row may carry it), 'B' (V_b: never in a row or window), 'Cb' (held-out construction), '?' (unknown construction)."""
    c = split["constructions"].get(cons_id)
    if c is None:
        return "?"
    if c["side"] == "C_b":
        return "Cb"
    if key in c["V_a"]:
        return "A"
    return "B" if key in c["V_b"] else "?"


def hash_side(lang: str, cons_id: str, key: str) -> str:
    """Side of a value the universe does not list (a value found in real speech): a fixed hash of (language, construction, value)."""
    return "A" if int(hashlib.sha256(f"{VERSION}|{lang}|{cons_id}|{key}".encode()).hexdigest(), 16) % 2 == 0 else "B"


# ------------------------------------------------------------------------------------------------ sentences
SLOT = re.compile(r"\{([XY])\}")


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _elide_piece(p):
    """fr: a phrase ending in ' de' before a vowel-initial word keeps the elision ('une trentaine d'' + 'euros')."""
    ends = lambda x: x.endswith(" de")  # noqa: E731
    if not ends(p.label):
        return p
    sw = lambda x: x[:-3] + " d'"  # noqa: E731
    return K.Piece(sw(p.label), sw(p.caption) if ends(p.caption) else p.caption,
                   [(k, w, sw(r) if ends(r) else r) for k, w, r in p.nums], p.ambig)


def assemble(lang: str, family: str, core: str, pieces: dict, rng, lead: str | None, tail: str | None) -> dict:
    """Sentence template -> label, caption, spans, char ranges. pieces: {"X": Piece, "Y": Piece}."""
    if lead and lead in NOCOMMA:
        t = lead + " " + core
    else:
        t = (lead + ", " if lead else "") + core
    t += (", " + tail if tail else "")
    out, cap, pos, nums, edge, spans = "", "", 0, [], [], []
    for m in SLOT.finditer(t):
        pre = t[pos:m.start()]
        p = pieces[m.group(1)]
        after = t[m.end():m.end() + 2]
        if lang == "fr":
            if re.match(r"(un|une)\b", p.label) and pre.endswith(" de "):   # de + un -> d'un
                pre = pre[:-4] + " d'"
            if p.label.endswith(" de") and re.match(r" [aeiouàâéèêëîïôùû]", after):
                p = _elide_piece(p)
                pos_skip = 1
            else:
                pos_skip = 0
        else:
            pos_skip = 0
        out += pre
        cap += pre
        lab, cp = p.label, p.caption
        if not out:
            lab, cp = _cap(lab), _cap(cp)
        a = len(out)
        out += lab
        cap += cp
        edge.append([a, len(out)])
        cur = 0
        for kind, written, reading in p.nums:
            r0 = lab.find(reading, cur)
            if r0 < 0:
                r0 = lab.lower().find(reading.lower(), cur)
            assert r0 >= 0, (lab, reading)
            nums.append([a + r0, a + r0 + len(reading)])
            spans.append({"text": written, "kind": kind, "reading": lab[r0:r0 + len(reading)], "canonical": lab[r0:r0 + len(reading)],
                          "idx": 0, "n_cands": 1, "slot": m.group(1)})
            cur = r0 + len(reading)
        pos = m.end() + pos_skip
    out += t[pos:]
    cap += t[pos:]
    if t[:1].islower() and not out[:1].isupper():
        out, cap = _cap(out), _cap(cap)
    return {"text": out + ".", "caption": cap + ".", "num_chars": nums, "edge_spans": edge, "spans": spans}


def _ambig_for(c, keys) -> list:
    tags = set(c.ambig)
    if c.key_ambig:
        for k in keys:
            tags |= set(c.key_ambig(k))
    return sorted(tags)


def plan(lang: str, split: dict, counts: dict, seed: int = 19, tag: str = "ttsEC1", per_template: int = 3) -> tuple:
    """Planned rows for the A halves: counts {family: sentences}. Returns (rows, report)."""
    cons = {c.id: c for c in inventory(lang)}
    rng = random.Random(f"{tag}|{seed}|{lang}")
    used = collections.Counter()
    tpl_n = collections.Counter()
    seen, rows, drops = set(), [], collections.Counter()
    by_cons = collections.Counter()
    for fam, n in counts.items():
        pool = [c for c in cons.values() if c.family == fam and split["constructions"][c.id]["side"] == "C_a"]
        w = [c.weight * min(len(split["constructions"][c.id]["V_a"]) ** 0.5, WCAP) for c in pool]
        got, stall = 0, 0
        while got < n and stall < 40 * n:
            c = rng.choices(pool, w)[0]
            va = split["constructions"][c.id]["V_a"]
            low = min(used[(c.id, k)] for k in va)
            key = rng.choice([k for k in va if used[(c.id, k)] == low])
            pieces = {"X": c.make(key, random.Random(f"{key}|{used[(c.id, key)]}|{seed}|{c.id}"))}
            keys = [key]
            two = (not c.value_cores) and lang in F.FRAMES2 and c.cat in F.FRAMES2[lang] and rng.random() < P_TWO
            if c.value_cores:
                cores = c.value_cores[key]
            elif two:
                cores = F.FRAMES2[lang][c.cat]
                others = [k for k in va if k != key]
                if not others:
                    two = False
                    cores = F.FRAMES[lang][c.cat]
                else:
                    low2 = min(used[(c.id, k)] for k in others)
                    k2 = rng.choice([k for k in others if used[(c.id, k)] == low2])
                    pieces["Y"] = c.make(k2, random.Random(f"{k2}|{used[(c.id, k2)]}|{seed}|{c.id}"))
                    keys.append(k2)
            else:
                cores = F.FRAMES[lang][c.cat]
            core = rng.choice(cores)
            lead = rng.choice(F.LEADS[lang][fam]) if rng.random() < P_LEAD else None
            tail = rng.choice(F.TAILS[lang][fam]) if rng.random() < P_TAIL else None
            if lead and tail and lead.lower() == tail.lower():
                tail = None
            r = assemble(lang, fam, core, pieces, rng, lead, tail)
            if r["text"] in seen or re.search(r"[0-9{}]", r["text"]):
                drops["duplicate_or_symbol"] += 1
                stall += 1
                continue
            template = f"{lead or ''}|{core}|{tail or ''}"
            if tpl_n[(c.id, template)] >= per_template:
                drops["template_cap"] += 1
                stall += 1
                continue
            seen.add(r["text"])
            tpl_n[(c.id, template)] += 1
            for k in keys:
                used[(c.id, k)] += 1
            by_cons[c.id] += 1
            stall = 0
            got += 1
            ambig = _ambig_for(c, keys)
            rid = f"{tag}_{lang}_{hashlib.md5(r['text'].encode()).hexdigest()[:10]}"
            row = {"id": rid, "lang": lang, "cls": "edge", "register": "edge", "caption": r["caption"], "text": r["text"],
                   "spans": r["spans"], "num_chars": r["num_chars"], "edge_spans": r["edge_spans"], "kind": sorted({s["kind"] for s in r["spans"]}),
                   "split": "train", "template": template, "family": fam, "edge": [fam], "construction": c.id,
                   "constructions": [c.id] * len(keys), "values": keys, "ambig": ambig, "half": "A", "slots": [s["slot"] for s in r["spans"]]}
            rows.append(row)
    rep = {"rows": len(rows), "drops": dict(drops), "by_construction": dict(by_cons),
           "values_used": len(used), "max_value_draws": max(used.values()) if used else 0}
    rng.shuffle(rows)
    return rows, rep


# ------------------------------------------------------------------------------------------------ gate classes
def add_gate_classes(rows: list, lang: str) -> None:
    """gclasses / ckey of each row as the data gate counts its spans (census_common.gate_class_value)."""
    import census_common as CC

    for r in rows:
        gcl = []
        for s in r["spans"]:
            try:
                gcl.append(CC.gate_class_value({"text": r["text"]}, s, lang)[0])
            except Exception:  # noqa: BLE001
                gcl.append(s["kind"])
        r["gclasses"] = gcl
        r["ckey"] = "+".join(sorted(set(gcl)))


# ------------------------------------------------------------------------------------------------ voices + pilot
def voice_group(r: dict) -> str:
    """The pilot / readout group of a row's voice: a clone's accent group, or a Chirp locale."""
    if r.get("engine_src") == "chirp3hd":
        return "chirp:" + (r.get("locale") or "?")
    acc = r.get("accent")
    return "clone:" + (acc if acc and acc != "None" else "untagged")


def assign(rows: list, lang: str, refs: list, cloud: list | None, chirp_share: float, seed: int = 19) -> dict:
    """Round robin over the clone references (accent-labelled ones carry `accent`); chirp_share of the rows go to cloud
    voices (locale-weighted). English clones alternate Qwen3 / Chatterbox per reference (ml_tts_symbol.assign_voices_en2);
    es / fr use Qwen3. Rates 0.9-1.25."""
    import ml_tts_symbol as S

    rng = random.Random(f"edgeA|voices|{seed}|{lang}")
    order = list(range(len(rows)))
    rng.shuffle(order)
    nc = int(round(len(rows) * chirp_share)) if cloud else 0
    ch = [rows[i] for i in order[:nc]]
    cl = [rows[i] for i in order[nc:]]
    vt = "edgeA"        # clips land under ml/edgeA/<lang>/presynth/ (a new volume path)
    if ch:
        S.assign_voices(ch, lang, vt, None, cloud, "chirp", 4, S.RATES_WIDE)
    if lang == "en":
        S.assign_voices_en2(cl, refs, vt)
    else:
        S.assign_voices(cl, lang, vt, refs, None, "qwen3", 10, S.RATES_WIDE, 0.0)
        by = {v["spk"]: v for v in refs}
        for r in cl:
            r["accent"] = by[r["voice"].split(":")[-1]].get("accent")
    for r in rows:
        r["group"] = voice_group(r)
    return dict(collections.Counter(r["group"] for r in rows))


def pilot_pick(rows: list, per_group: int = 30, per_cons: int = 10, seed: int = 19) -> list:
    """Pilot sample from voiced bulk rows: >= per_group clips for every voice group, then up to per_cons for every construction;
    the rest of the rows is bulk. Picks round robin over constructions inside a group."""
    rng = random.Random(f"pilot|{seed}")
    pick: dict = {}
    by_g = collections.defaultdict(list)
    for r in rows:
        by_g[r["group"]].append(r)
    for g, rs in sorted(by_g.items()):
        rng.shuffle(rs)
        bycons = collections.defaultdict(list)
        for r in rs:
            bycons[r["construction"]].append(r)
        keys = sorted(bycons)
        rng.shuffle(keys)
        n, rnd = 0, 0
        while n < per_group and rnd < 50:
            for k in keys:
                if n >= per_group:
                    break
                if rnd < len(bycons[k]) and bycons[k][rnd]["id"] not in pick:
                    pick[bycons[k][rnd]["id"]] = bycons[k][rnd]
                    n += 1
            rnd += 1
    have = collections.Counter(r["construction"] for r in pick.values())
    rest = [r for r in rows if r["id"] not in pick]
    rng.shuffle(rest)
    for r in rest:
        if have[r["construction"]] < per_cons:
            pick[r["id"]] = r
            have[r["construction"]] += 1
    return list(pick.values())


# ------------------------------------------------------------------------------------------------ CLI
def _write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--out", required=True)
    f.add_argument("--langs", default=",".join(LANGS))
    p = sub.add_parser("plan")
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--counts", default='{"fin": 700, "col": 420, "dis": 120}')
    p.add_argument("--langs", default=",".join(LANGS))
    p.add_argument("--seed", type=int, default=19)
    v = sub.add_parser("voices")
    v.add_argument("--planned", required=True)
    v.add_argument("--refs", required=True, help="json with a refs list (ml_tts_open meta.json shape), several files comma-separated")
    v.add_argument("--chirp", default="", help="json [[voice, locale], ...]")
    v.add_argument("--chirp-share", type=float, default=0.15)
    v.add_argument("--out", required=True)
    k = sub.add_parser("pilot")
    k.add_argument("--voiced", required=True)
    k.add_argument("--per-group", type=int, default=30)
    k.add_argument("--per-cons", type=int, default=10)
    k.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "freeze":
        os.makedirs(a.out, exist_ok=True)
        man = {"version": VERSION, "seed": SEED, "sha256": {}}
        for lang in a.langs.split(","):
            s = freeze(lang)
            path = os.path.join(a.out, f"ec1split_{lang}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(s, fh, ensure_ascii=False, indent=1, sort_keys=True)
            man["sha256"][lang] = hashlib.sha256(canonical(s)).hexdigest()
            print(lang, s["n_constructions"], "held out", s["n_held_out"], man["sha256"][lang][:12])
        json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    elif a.cmd == "plan":
        counts = json.loads(a.counts)
        os.makedirs(a.out, exist_ok=True)
        for lang in a.langs.split(","):
            split = json.load(open(os.path.join(a.split, f"ec1split_{lang}.json"), encoding="utf-8"))
            assert hashlib.sha256(canonical(split)).hexdigest() == json.load(open(os.path.join(a.split, "manifest.json")))["sha256"][lang], "split changed"
            rows, rep = plan(lang, split, counts, a.seed)
            add_gate_classes(rows, lang)
            _write(os.path.join(a.out, f"planned_{lang}.jsonl"), rows)
            json.dump(rep, open(os.path.join(a.out, f"plan_report_{lang}.json"), "w"), indent=1)
            print(lang, rep["rows"], rep["drops"], "chars", sum(len(r["text"]) for r in rows))
    elif a.cmd == "voices":
        rows = _rows(a.planned)
        lang = rows[0]["lang"]
        refs = [x for pth in a.refs.split(",") for x in json.load(open(pth))["refs"]]
        cloud = [tuple(x) for x in json.load(open(a.chirp))] if a.chirp else None
        print(assign(rows, lang, refs, cloud, a.chirp_share))
        _write(a.out, rows)
    elif a.cmd == "pilot":
        rows = _rows(a.voiced)
        pk = pilot_pick(rows, a.per_group, a.per_cons)
        _write(a.out, pk)
        print(len(pk), "pilot rows;", dict(collections.Counter(r["group"] for r in pk)))


if __name__ == "__main__":
    main()
