"""Value-spread and rare-form packs (2026-10-18, ttsVS1 / ttsRF1): whole-number VALUES with few training spans, in
natural sentences with 2-4 numbers each, open-licence cloned voices (ml_tts_open.py), per-clip CTC (ml_tts.py).

  coverage   spans per (gate class, value) of a built mix (real + synthetic) and the synthetic room per class
  split      the priority rare values per language and class (greedy on the word-form error gradient by spans per
             value), split at random into V_a (filled) and V_b (left unfilled: the control half), stratified on class,
             reading length and current coverage, and balanced on the held-out spans
  plan       forced-value sentences for V_a (carriers_vs1.json), realised with the value kept: every reading must
             give the written value back through the scorer's reader; each span lands in the intended gate class
  voices     engine / reference / rate / pitch per row (ml_tts_symbol.assign_voices)

    python ml_tts_spread.py coverage --mix <dir with p3_/tts_/heldout_ jsonl> --out cov.json
    python ml_tts_spread.py split --cov cov.json --mix <dir> --out <dir> --budget 2200 [--langs de,es]
    python ml_tts_spread.py plan --split <dir>/vsplit_<lang>.json --cov cov.json --out <dir> --tag ttsVS1
"""
import argparse
import bisect
import collections
import hashlib
import heapq
import json
import math
import os
import random
import re
import sys

import census_common as CC
import ml_cands as M
import ml_p3_core as C
import ml_tts as T
import ml_tts_symbol as S

HERE = os.path.dirname(os.path.abspath(__file__))
LANGS = ["de", "es", "fr", "it", "nl", "pt"]
CLASSES = ("int_small", "year", "year_early", "int_big")
# word-form error (%) by training spans per value in the plain classes : 0 -> 27, 1-4 -> 17, 5-19 -> 11, 20-99 -> 6.7
ERR_PTS = [(0.0, 27.0), (2.5, 17.0), (12.0, 11.0), (60.0, 6.7), (100.0, 6.0)]
VALUE_CAP = 50  # spans one value may take from a pack
SHARE_CAP = 0.03  # and at most this share of its class after the pack
SAFETY = 0.9  # of the synthetic room a pack plans


def err(c: float) -> float:
    x = math.log1p(c)
    pts = [(math.log1p(a), b) for a, b in ERR_PTS]
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]


def weight(cls: str, v: int) -> float:
    """How often a value is said, relative (a prior: rounder and smaller numbers are more common; forward years)."""
    if cls == "int_small":
        base = 5.0 if v % 100 == 0 else 3.0 if v % 50 == 0 else 1.8 if v % 10 == 0 else 1.3 if v % 5 == 0 else 1.0
        return base * (200.0 / (v + 100)) ** 0.7
    if cls == "year":
        if 2031 <= v <= 2060:
            return 3.0 if v in CC.FORWARD_KEY else 2.0
        if 1800 <= v <= 1899:
            return 0.9 if v % 10 == 0 else 0.7
        if 2061 <= v <= 2099:
            return 0.3
        return 0.5
    if cls == "year_early":
        return 0.3
    return 0.8  # int_big


def universe(lang: str) -> dict:
    """{class: [values]} a pack may fill."""
    u = {"int_small": list(range(100, 1000)), "year": list(range(1800, 1900)) + list(range(2031, 2061)),
         "year_early": list(range(1100, 1800)) if lang in ("es", "fr") else [],
         "int_big": [k * 1000 for k in range(2, 100) if not 1100 <= k * 1000 <= 2099] +
                    [k * 500 for k in range(5, 60) if k % 2]}
    return u


# ------------------------------------------------------------------------------------------------ coverage

def _rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if os.path.exists(path) else []


def _ival(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def coverage(mix: str, lang: str, extra: tuple = ()) -> dict:
    """{"real": {cls: {value: spans}}, "syn": ..., "held": ..., "room": {cls: synthetic spans still admitted},
    "totals": {cls: (real, syn)}} for the gate classes a pack fills."""
    import mix_build as MB
    real, syn = _rows(f"{mix}/p3_{lang}.jsonl"), _rows(f"{mix}/tts_{lang}.jsonl")
    for p in extra:
        syn += _rows(p)
    held = _rows(f"{mix}/heldout_{lang}.jsonl")
    rn, rv = MB.class_values(real, lang)
    sn, sv = MB.class_values(syn, lang)
    _, hv = MB.class_values(held, lang)
    out = {"real": {}, "syn": {}, "held": {}, "room": {}, "totals": {}}
    for c in CLASSES:
        for k, src in (("real", rv), ("syn", sv), ("held", hv)):
            out[k][c] = {str(v): n for v, n in (src.get(c) or {}).items() if _ival(v) is not None}
        al = CC.syn_allowed(lang, c, rn[c], rv.get(c), sv.get(c))
        out["room"][c] = max(0, al - sn[c])
        out["totals"][c] = [rn[c], sn[c]]
    return out


def counts(cov: dict, cls: str) -> collections.Counter:
    t = collections.Counter()
    for k in ("real", "syn"):
        for v, n in cov[k].get(cls, {}).items():
            t[int(v)] += n
    return t


# ------------------------------------------------------------------------------------------------ priority list + split

SHARES = {"int_small": 0.60, "year": 0.22, "year_early": 0.06, "int_big": 0.12}  # of a pack's spans, where the room allows


def allocate(lang: str, cov: dict, budget: int, values: dict | None = None, held_boost: float = 20.0) -> dict:
    """{(class, value): new spans}: per class a greedy on the marginal drop of the word-form error per span, weight(value) x
    (err(c) - err(c + 1)), under the class room (SAFETY), VALUE_CAP and SHARE_CAP; the classes share `budget` as
    SHARES says, what a class cannot take (room, values) goes to int_small, then year. values: {class: [values]}
    restriction (default: the universe). Values with held-out spans come first (held_boost)."""
    u = values or universe(lang)
    cnt = {c: counts(cov, c) for c in CLASSES}
    held = {c: collections.Counter({int(v): n for v, n in cov["held"].get(c, {}).items()}) for c in CLASSES}
    room = {c: int(cov["room"][c] * SAFETY) for c in CLASSES}
    tot = {c: sum(cov["totals"][c]) for c in CLASSES}
    new = collections.Counter()

    def run(cls, n):
        heap, spent = [], 0

        def gain(v):
            w = weight(cls, v) * (1.0 + held_boost * min(held[cls].get(v, 0), 2))
            c = cnt[cls][v] + new[(cls, v)]
            return w * (err(c) - err(c + 1))

        for v in u.get(cls, []):
            heapq.heappush(heap, (-gain(v), v))
        while heap and spent < n:
            g, v = heapq.heappop(heap)
            c = cnt[cls][v] + new[(cls, v)]
            if sum(1 for (cc, _) in new if cc == cls) and False:
                break
            used_c = sum(x for (cc, _), x in new.items() if cc == cls)
            if used_c >= room[cls]:
                break
            if new[(cls, v)] >= VALUE_CAP or c + 1 > SHARE_CAP * (tot[cls] + used_c + 1):
                continue
            new[(cls, v)] += 1
            spent += 1
            heapq.heappush(heap, (-gain(v), v))
        return spent

    left = budget
    for cls in CLASSES:
        left -= run(cls, int(budget * SHARES[cls]))
    for cls in ("int_small", "year", "int_big"):
        if left > 0:
            left -= run(cls, left)
    return dict(new)


def length_bin(lang: str, v: int) -> int:
    try:
        r = M.spans(str(v), lang)
        n = len(r[0].cands[0].split()) if r and r[0].cands else 0
    except Exception:  # noqa: BLE001
        n = 0
    return 0 if n <= 2 else 1 if n <= 3 else 2 if n <= 4 else 3


def cov_bin(c: int) -> int:
    return 0 if c == 0 else 1 if c < 5 else 2 if c < 20 else 3


def split_values(lang: str, cov: dict, budget: int, seed: int = 18) -> dict:
    """The priority list the budget would fill (values with held-out spans first), split into V_a / V_b: strata of
    (class, reading-length bin, current-coverage bin, held-out or not); inside a stratum values are ordered by held-out
    spans then value and dealt a b b a a b ... so both halves get the same held-out spans, the same lengths and the
    same coverage. V_a is filled with the whole budget (about twice the rows per value), V_b is left alone."""
    rng = random.Random(f"{seed}|split|{lang}")
    pri = allocate(lang, cov, budget)
    cnt = {c: counts(cov, c) for c in CLASSES}
    held = {c: collections.Counter({int(v): n for v, n in cov["held"].get(c, {}).items()}) for c in CLASSES}
    strata = collections.defaultdict(list)
    for (c, v), n in pri.items():
        strata[(c, length_bin(lang, v), cov_bin(cnt[c][v]), held[c][v] > 0)].append(v)
    va, vb = [], []
    hs = {"a": collections.Counter(), "b": collections.Counter()}
    for k in sorted(strata):
        vs = strata[k]
        rng.shuffle(vs)
        vs.sort(key=lambda v: -held[k[0]][v])
        na = nb = 0
        for i, v in enumerate(vs):
            h = held[k[0]][v]
            if abs(na - nb) >= 1 and h == 0:
                to_a = na < nb
            elif h and hs["a"][k[0]] != hs["b"][k[0]]:
                to_a = hs["a"][k[0]] < hs["b"][k[0]] if abs(na - nb) <= 1 else na < nb
            else:
                to_a = na < nb if na != nb else rng.random() < 0.5
            if to_a:
                va.append((k[0], v)); na += 1; hs["a"][k[0]] += h
            else:
                vb.append((k[0], v)); nb += 1; hs["b"][k[0]] += h
    return {"lang": lang, "seed": seed, "budget": budget, "priority": {f"{c}:{v}": n for (c, v), n in sorted(pri.items())},
            "V_a": sorted(f"{c}:{v}" for c, v in va), "V_b": sorted(f"{c}:{v}" for c, v in vb),
            "held_spans": {h: {c: sum(held[c][v] for cc, v in vs if cc == c) for c in CLASSES}
                           for h, vs in (("V_a", va), ("V_b", vb))},
            "n_values": {h: dict(collections.Counter(c for c, _ in vs)) for h, vs in (("V_a", va), ("V_b", vb))}}


# ------------------------------------------------------------------------------------------------ sentences

SLOT_RE = re.compile(r"\{([nlyb])(\d?)\}")
SLOT_CLASS = {"n": "count", "l": "count", "b": "count", "y": "year"}


def load_carriers(lang: str, name: str = "carriers_vs1.json") -> dict:
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)[lang]


def _bad_end(lang, n) -> bool:
    return n % 10 in (1, 2) and n % 100 not in (11, 12)


def _written_big(n: int, lang: str, rng) -> str:
    return f"{n:,}".replace(",", " " if lang == "fr" else ".")


def _value_back(lang, reading, N) -> set:
    """The single value a reading says, else an empty set."""
    got = N.extract_numbers(reading, lang)
    if len(got) != 1:
        return set()
    return {int(x) for x in got[0].vals}


def realize_values(lang, written, ranges, rng, used, N):
    """ranges: [(start, end, slot letter, value)]. One span per slot; per span the readings that (a) the clause allows
    (ml_tts.reading_ok, class count / year), (b) give the written value back, (c) for years the full reading or the
    hundreds-style reading only (cands[0] / cands[1]), for counts no feminine / clipped form. Returns
    ((label, spans, offsets), "") or (None, reason)."""
    try:
        sp = M.spans_regional(written, lang, S.NAMES)
    except Exception as e:  # noqa: BLE001
        return None, f"spans_error {type(e).__name__}"
    if len(sp) != len(ranges):
        return None, "span_count"
    if any(not s.cands for s in sp):
        return None, "missing_grammar" if any(s.note.startswith("missing") for s in sp) else "no_cands"
    if not M.all_digits_spanned(written, lang):
        return None, "digits_unspanned"
    picks = []
    for s, (a, b, slot, val) in zip(sp, ranges):
        if s.kind != "int" or not (a <= s.start and s.end <= b + 1):
            return None, "span_shape"
        cls = SLOT_CLASS[slot]
        ok = []
        for i, c in enumerate(s.cands):
            if not T.reading_ok(lang, T.GROUP[cls], cls, "m" if cls == "count" else None, s, c, 0, written):
                continue
            if val not in _value_back(lang, c.strip(), N):
                continue
            if slot == "y" and i > 1:
                continue
            ok.append(i)
        if not ok:
            return None, "no_reading_fits"
        w = [(3.0 if i == 0 else 1.0) / (1.0 + 0.7 * used[(lang, slot, s.cands[i].strip())]) for i in ok]
        picks.append(rng.choices(ok, w)[0])
    offs: list = []
    label = C.render_cap(written, sp, picks, offs)
    if written.endswith(".") and not label.endswith("."):
        label += "."
    if T.FORBIDDEN.search(label) or C.WF.search(label) or not S.label_ok(label):
        return None, "symbol_left"
    if lang in S.ELISION_BAD and S.ELISION_BAD[lang].search(label):
        return None, "elision"
    spans = [{"text": s.text, "kind": "int", "reading": s.cands[p].strip(), "canonical": s.cands[0].strip(), "idx": p,
              "n_cands": len(s.cands), "slot": slot} for s, p, (_, _, slot, _) in zip(sp, picks, ranges)]
    for s, p, (_, _, slot, _) in zip(sp, picks, ranges):
        used[(lang, slot, s.cands[p].strip())] += 1
    return (label, spans, offs), ""


def plan_lang(lang: str, targets: dict, tag: str, seed: int, rep: dict, per_carrier: int = 45,
              carriers: str = "carriers_vs1.json") -> list:
    """targets: {(class, value): spans}. Sentences fill the targets exactly (forced values); a span whose sentence
    fails is not retried with the same value more than a few times."""
    N = S._reader()
    rng = random.Random(f"{tag}|{seed}|{lang}")
    car = load_carriers(lang, carriers)
    used = collections.Counter()
    drops = collections.Counter()
    got = collections.Counter()
    tpl_n = collections.Counter()
    seen, rows, missing = set(), [], []
    bag = {c: [v for (cc, v), n in sorted(targets.items()) if cc == c for _ in range(n)] for c in CLASSES}
    for c in bag:
        rng.shuffle(bag[c])
    kinds = {"int_small": ("count", "label"), "year": ("past", "fut"), "year_early": ("past",), "int_big": ("big",)}
    # a value ending in 1 or 2 (un / uno / una, one / two with a gender) only in a label context (a page, a room, ...)
    lab_only = lambda v: _bad_end(lang, v)  # noqa: E731
    stall = 0
    while any(bag.values()) and stall < 4000:
        cls = max((c for c in bag if bag[c]), key=lambda c: len(bag[c]) * (1.0 + rng.random() * 0.3))
        fam = rng.choice(kinds[cls])
        if cls == "int_small" and any(lab_only(v) for v in bag[cls][-12:]) and rng.random() < 0.6:
            fam = "label"
        tpl = rng.choice(car[fam])
        if tpl_n[tpl] >= per_carrier:
            stall += 1
            continue
        slots = list(SLOT_RE.finditer(tpl))
        picked, pool = [], bag[cls]
        ok = True
        for m in slots:
            idx = None
            for j in range(len(pool) - 1, max(-1, len(pool) - 40), -1):
                v = pool[j]
                if v in [x for x in picked] or (m.group(1) == "n" and lab_only(v)) :
                    continue
                idx = j
                break
            if idx is None:
                ok = False
                break
            picked.append(pool.pop(idx))
        if not ok:
            pool.extend(picked)
            stall += 1
            continue
        written, ranges, pos = "", [], 0
        for m, v in zip(slots, picked):
            written += tpl[pos:m.start()]
            w = _written_big(v, lang, rng) if m.group(1) == "b" else str(v)
            ranges.append((len(written), len(written) + len(w) - 1, m.group(1), v))
            written += w
            pos = m.end()
        written += tpl[pos:]
        res, why = realize_values(lang, written, ranges, rng, used, N)
        if res is None:
            drops[why] += 1
            stall += 1
            if why == "missing_grammar":
                missing.append(written)
            continue
        label, spans, offs = res
        if label in seen:
            drops["duplicate"] += 1
            stall += 1
            continue
        row = {"text": label}
        gcl, bad = [], False
        for s, (_, _, _, v) in zip(spans, ranges):
            gc, gv = CC.gate_class_value(row, s, lang)
            if gc != cls or _ival(gv) != v:
                bad = True
            s["gclass"] = gc
            gcl.append(gc)
        if bad:
            drops["wrong_gate_class"] += 1
            stall += 1
            continue
        seen.add(label)
        tpl_n[tpl] += 1
        stall = 0
        rid = f"{tag}_{lang}_{hashlib.md5(f'{seed}|{label}'.encode()).hexdigest()[:10]}"
        rows.append({"id": rid, "lang": lang, "cls": "sym", "ckey": "+".join(sorted(set(gcl))), "register": "sym",
                     "caption": written, "text": label,
                     "spans": [{k: v for k, v in s.items() if k != "gclass"} for s in spans], "num_chars": offs,
                     "kind": ["int"], "split": "train", "slots": [s["slot"] for s in spans], "gclasses": gcl,
                     "template": tpl, "values": [v for *_, v in ranges]})
        for g, (_, _, _, v) in zip(gcl, ranges):
            got[(g, v)] += 1
    rep.update(rows=len(rows), spans=sum(got.values()), drops=dict(drops), left={c: len(b) for c, b in bag.items()},
               carriers_used=len(tpl_n), missing_texts=missing)
    rng.shuffle(rows)
    return rows


# ------------------------------------------------------------------------------------------------ rare forms (ttsRF1)

EU_PT = {"dezasseis", "dezassete", "dezanove", "quatorze"}  # European Portuguese teens
BR_PT = {"dezesseis", "dezessete", "dezenove", "catorze"}
FORM_SLOT = re.compile(r"\{([fmt])(\d?)\}")


def fem_form(lang: str, base: str) -> str:
    """The reading with EVERY gender-variable word feminine (es doscientas veintiuna; pt duzentas e vinte e duas)."""
    ws = base.split()
    for i, w in enumerate(ws):
        last = i == len(ws) - 1
        if lang == "es":
            if w.endswith("ientos"):
                ws[i] = w[:-2] + "as"
            elif last and w in ("uno", "un"):
                ws[i] = "una"
            elif last and w in ("veintiuno", "veintiún"):
                ws[i] = "veintiuna"
        else:
            if w == "dois":
                ws[i] = "duas"
            elif w == "um":
                ws[i] = "uma"
            elif w.endswith("entos"):
                ws[i] = w[:-2] + "as"
    return " ".join(ws)


def realize_forms(lang, written, ranges, rng, used, N):
    """ranges: [(start, end, slot letter, value)]; slot f: the fully feminine reading, m: es apocope (un / veintiún) /
    pt plain, t: a European Portuguese teen reading. The reading must give the value back through the scorer's reader."""
    try:
        sp = M.spans_regional(written, lang, S.NAMES)
    except Exception as e:  # noqa: BLE001
        return None, f"spans_error {type(e).__name__}"
    if len(sp) != len(ranges):
        return None, "span_count"
    if any(not s.cands for s in sp):
        return None, "missing_grammar" if any(s.note.startswith("missing") for s in sp) else "no_cands"
    picks = []
    for s, (a, b, slot, val) in zip(sp, ranges):
        if s.kind != "int":
            return None, "span_shape"
        base = s.cands[0].strip()
        if slot == "f":
            tgt = fem_form(lang, base)
            if tgt == base or val not in _value_back(lang, tgt, N):
                return None, "no_fem_form"
            if tgt not in [c.strip() for c in s.cands]:
                s.cands.append(" " + tgt if s.cands[0].startswith(" ") else tgt)
            picks.append([c.strip() for c in s.cands].index(tgt))
        elif slot == "m" and lang == "es":
            ok = [i for i, c in enumerate(s.cands) if c.split()[-1] in ("un", "veintiún") and val in _value_back(lang, c.strip(), N)]
            if not ok:
                return None, "no_apocope"
            picks.append(ok[0])
        elif slot == "t":
            ok = [i for i, c in enumerate(s.cands) if set(c.split()) & EU_PT and not set(c.split()) & BR_PT
                  and val in _value_back(lang, c.strip(), N)]
            if not ok:
                return None, "no_eu_teen"
            picks.append(ok[0])
        else:
            picks.append(0)
    offs: list = []
    label = C.render_cap(written, sp, picks, offs)
    if written.endswith(".") and not label.endswith("."):
        label += "."
    if T.FORBIDDEN.search(label) or C.WF.search(label) or not S.label_ok(label):
        return None, "symbol_left"
    spans = [{"text": s.text, "kind": "int", "reading": s.cands[p].strip(), "canonical": s.cands[0].strip(), "idx": p,
              "n_cands": len(s.cands), "slot": slot} for s, p, (_, _, slot, _) in zip(sp, picks, ranges)]
    return (label, spans, offs), ""


def form_pools(lang: str, cov: dict, excluded: set) -> dict:
    cnt = counts(cov, "int_small")
    # pt: the teens 14 / 16 / 17 / 19 (also inside 114 ... 919) only in the European-teen slot, never in the others
    ok = lambda v: v not in excluded and not (lang == "pt" and v % 100 in (14, 16, 17, 19))  # noqa: E731
    pools = {"f": [v for v in range(200, 1000) if ok(v)] + [v for v in list(range(21, 100)) + list(range(101, 200)) if v % 10 == 1 and v % 100 != 11 and ok(v)]
             + ([v for v in range(22, 100) if v % 10 == 2 and ok(v)] + [v for v in range(102, 200) if v % 10 == 2 and ok(v)] if lang == "pt" else []),
             "m": [v for v in range(21, 1000) if v % 10 == 1 and v % 100 != 11 and ok(v)]}
    if lang == "pt":
        pools["t"] = [d + 100 * h for h in range(0, 10) for d in (14, 16, 17, 19) if d + 100 * h not in excluded]
    return {k: (v, {x: 1.0 / (1.0 + cnt[x] / 5.0) * (2.5 if x in (21, 31, 41, 51, 61, 71, 81, 91, 101) or x % 100 == 0 else 1.0) for x in v})
            for k, v in pools.items()}


def plan_forms(lang: str, cov: dict, excluded: set, share: dict, n_spans: int, tag: str, seed: int, rep: dict,
               per_value: int = 24, per_carrier: int = 90) -> list:
    N = S._reader()
    rng = random.Random(f"{tag}|{seed}|{lang}")
    car = load_carriers(lang, "carriers_rf1.json")
    pools = form_pools(lang, cov, excluded)
    drawn = collections.Counter()
    drops, tpl_n, seen, rows, missing = collections.Counter(), collections.Counter(), set(), [], []
    want = {k: int(n_spans * share.get(k, 0)) for k in car}
    got = collections.Counter()
    used = collections.Counter()
    tries, fail, dead = 0, collections.Counter(), set()
    while any(got[k] < want[k] for k in want if k not in dead) and tries < 40 * n_spans:
        tries += 1
        fam = max((k for k in want if got[k] < want[k] and k not in dead), key=lambda k: (want[k] - got[k]) * (1 + rng.random() * 0.4))
        fail[fam] += 1
        if fail[fam] > 300:
            dead.add(fam)
            continue
        tpl = rng.choice(car[fam])
        if tpl_n[tpl] >= per_carrier:
            continue
        slots = list(FORM_SLOT.finditer(tpl))
        vals, vs = [], pools[slots[0].group(1)]
        for m in slots:
            vals_pool, w = pools[m.group(1)]
            cand = [v for v in rng.choices(vals_pool, [w[x] / (1 + drawn[(m.group(1), x)]) for x in vals_pool], k=8)
                    if v not in vals and drawn[(m.group(1), v)] < per_value]
            if not cand:
                break
            vals.append(cand[0])
        if len(vals) != len(slots):
            continue
        written, ranges, pos = "", [], 0
        for m, v in zip(slots, vals):
            written += tpl[pos:m.start()]
            ranges.append((len(written), len(written) + len(str(v)) - 1, m.group(1), v))
            written += str(v)
            pos = m.end()
        written += tpl[pos:]
        res, why = realize_forms(lang, written, ranges, rng, used, N)
        if res is None:
            drops[why] += 1
            if why == "missing_grammar":
                missing.append(written)
            continue
        label, spans, offs = res
        if label in seen:
            drops["duplicate"] += 1
            continue
        row = {"text": label}
        gcl = []
        for sp_, (_, _, _, v) in zip(spans, ranges):
            gc, gv = CC.gate_class_value(row, sp_, lang)
            gcl.append(gc)
            if gc != "int_small" or _ival(gv) != v:
                drops["wrong_gate_class"] += 1
                break
        else:
            seen.add(label)
            tpl_n[tpl] += 1
            for (_, _, k, v) in ranges:
                drawn[(k, v)] += 1
            got[fam] += len(slots)
            fail[fam] = 0
            rid = f"{tag}_{lang}_{hashlib.md5(f'{seed}|{label}'.encode()).hexdigest()[:10]}"
            rows.append({"id": rid, "lang": lang, "cls": "sym", "ckey": "int_small", "register": "sym", "caption": written,
                         "text": label, "spans": spans, "num_chars": offs, "kind": ["int"], "split": "train",
                         "slots": [r[2] for r in ranges], "gclasses": gcl, "template": tpl, "family": fam,
                         "values": [r[3] for r in ranges]})
    rep.update(rows=len(rows), spans=dict(got), want=want, drops=dict(drops), carriers_used=len(tpl_n), missing_texts=missing)
    rng.shuffle(rows)
    return rows


# ------------------------------------------------------------------------------------------------ grammar overlay

def ensure_table(lang: str, texts: list, dumper: str, workdir: str, scratch_table: str) -> int:
    """Grammar-table keys the sentences need that the shared table lacks, asked from the dumper and merged into a COPY
    of the table (scratch_table); ml_cands reads the copy from then on. The shared table is not touched."""
    import shutil
    if not os.path.exists(scratch_table):
        shutil.copy(M.TABLE_PATH, scratch_table)
    M.TABLE_PATH = scratch_table
    M._TABLE = None
    return M.build_table({lang: texts}, dumper, workdir)


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coverage")
    c.add_argument("--mix", required=True)
    c.add_argument("--extra", default="", help="comma list of extra synthetic rows files, {lang} in the name")
    c.add_argument("--out", required=True)
    c.add_argument("--langs", default=",".join(LANGS))
    s = sub.add_parser("split")
    s.add_argument("--cov", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--budget", type=int, default=2200)
    s.add_argument("--langs", default=",".join(LANGS))
    s.add_argument("--seed", type=int, default=18)
    p = sub.add_parser("plan")
    p.add_argument("--splitdir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tag", default="ttsVS1")
    p.add_argument("--langs", default=",".join(LANGS))
    p.add_argument("--seed", type=int, default=18)
    p.add_argument("--per-carrier", type=int, default=45)
    p.add_argument("--budget", type=int, default=0, help="spans for V_a (0: the split's budget)")
    p.add_argument("--cov", required=True)
    p.add_argument("--table", default="", help="scratch copy of the grammar table with the extra keys")
    p.add_argument("--dumper", default="")
    r = sub.add_parser("plan-forms")
    r.add_argument("--cov", required=True)
    r.add_argument("--splitdir", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--tag", default="ttsRF1")
    r.add_argument("--langs", default="es,pt")
    r.add_argument("--spans", type=int, default=2200)
    r.add_argument("--seed", type=int, default=18)
    r.add_argument("--table", default="")
    r.add_argument("--dumper", default="")
    a = ap.parse_args()
    if a.cmd == "plan-forms":
        cov = json.load(open(a.cov))
        os.makedirs(a.out, exist_ok=True)
        shares = {"es": {"f": 0.75, "m": 0.25}, "pt": {"f": 0.55, "m": 0.10, "t": 0.35}}
        for lang in a.langs.split(","):
            sp = json.load(open(os.path.join(a.splitdir, f"vsplit_{lang}.json")))
            excl = {int(k.split(":")[1]) for k in sp["V_b"] if k.startswith("int_small:")}
            rep: dict = {}
            rows = plan_forms(lang, cov[lang], excl, shares[lang], a.spans, a.tag, a.seed, rep)
            if a.table and rep["missing_texts"]:
                n = ensure_table(lang, rep["missing_texts"], a.dumper, a.out, a.table)
                print(lang, "grammar keys added", n)
                rep = {}
                rows = plan_forms(lang, cov[lang], excl, shares[lang], a.spans, a.tag, a.seed, rep)
            rep.pop("missing_texts", None)
            rep["chars"] = sum(len(r["text"]) for r in rows)
            _write(os.path.join(a.out, f"planned_{lang}.jsonl"), rows)
            json.dump(rep, open(os.path.join(a.out, f"plan_report_{lang}.json"), "w"), indent=1)
            print(lang, json.dumps(rep, ensure_ascii=False))
        return
    if a.cmd == "coverage":
        out = {}
        for lang in a.langs.split(","):
            extra = tuple(x.format(lang=lang) for x in a.extra.split(",") if x)
            out[lang] = coverage(a.mix, lang, extra)
            print(lang, out[lang]["room"], out[lang]["totals"])
        json.dump(out, open(a.out, "w"))
    elif a.cmd == "split":
        cov = json.load(open(a.cov))
        os.makedirs(a.out, exist_ok=True)
        for lang in a.langs.split(","):
            sp = split_values(lang, cov[lang], a.budget, a.seed)
            json.dump(sp, open(os.path.join(a.out, f"vsplit_{lang}.json"), "w"), indent=1)
            print(lang, "V_a", sp["n_values"]["V_a"], "V_b", sp["n_values"]["V_b"], "held", sp["held_spans"])
    elif a.cmd == "plan":
        cov = json.load(open(a.cov))
        os.makedirs(a.out, exist_ok=True)
        for lang in a.langs.split(","):
            sp = json.load(open(os.path.join(a.splitdir, f"vsplit_{lang}.json")))
            vals = collections.defaultdict(list)
            for k in sp["V_a"]:
                cl, v = k.split(":")
                vals[cl].append(int(v))
            tg = allocate(lang, cov[lang], a.budget or sp["budget"], dict(vals))
            rep: dict = {"targets": {c: sum(n for (cc, _), n in tg.items() if cc == c) for c in CLASSES}}
            rows = plan_lang(lang, tg, a.tag, a.seed, rep, a.per_carrier)
            if a.table and rep["missing_texts"]:
                n = ensure_table(lang, rep["missing_texts"], a.dumper, a.out, a.table)
                print(lang, "grammar keys added", n, "for", len(rep["missing_texts"]), "sentences")
                rep = {"targets": rep["targets"]}
                rows = plan_lang(lang, tg, a.tag, a.seed, rep, a.per_carrier)
            rep.pop("missing_texts", None)
            rep["chars"] = sum(len(r["text"]) for r in rows)
            _write(os.path.join(a.out, f"planned_{lang}.jsonl"), rows)
            json.dump(rep, open(os.path.join(a.out, f"plan_report_{lang}.json"), "w"), indent=1)
            print(lang, json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
