"""Retrain-3 crop filters over the recropped shards (R3.12.2 D-1b / D-4 / label assert / unit abbreviations, R3.17 F1
filler restoration + F2 label-agreement report). CPU only; runs on a local copy of the small text files:

    <root>/crops3/<tag>.jsonl   recrop v2 output
    <root>/dec3/<tag>.json      {wav: {"stock": text, "v10": text}} (modal s3_decode_crops)
    <root>/rows3/<tag>.jsonl    relabel rows (span decisions, CTC margins)
    <root>/veto3/<tag>.json     HuBERT picks per multi-reading span (modal s3_veto --rows-dir d1/rows3)

Writes <root>/crops3_ok/<tag>.jsonl (kept crops, fillers restored) and <root>/filter3_report.json (counts, 30 listed
crops per filter, per-kind agreement).

    python filter3.py <root>
"""

import collections
import glob
import json
import os
import random
import re
import sys

import labeller as L
from common import NUMBER_WORDS, SPAN_CORE, WRITTEN_FORM, number_span
from restore_fillers import FILL, merge

UNIT_ABBR = re.compile(r"\b(?:" + NUMBER_WORDS.pattern[3:-3] + r"|one|oh)\s+(?:m|km|kg|g|mg|ms|mb|gb|tb|kb|cm|mm|ft|lb|lbs|"
                       r"oz|hz|khz|mhz|ghz|kw|mw|kwh|v|w|mph|kph|kmh|sq)\b", re.I)
D4_MARGIN = 6.0


def _words(text: str) -> list[str]:
    out = []
    for w in re.sub(r"[-–—]", " ", text).split():
        w = w.lower().strip(".,?!;:\"()[]")
        if w and w not in FILL:
            out.append(w)
    return out


SCALE = {"thousand": 10**3, "million": 10**6, "billion": 10**9, "trillion": 10**12}


def _canon(v) -> str:
    from decimal import Decimal

    try:
        return format(Decimal(str(v)).normalize(), "f")
    except Exception:
        return str(v)


def _spelled(span: list[str]):
    """Value of a spelled span; "two point five million" = 2.5e6 (spelled_value cannot scale a decimal)."""
    v = L.spelled_value(" ".join(span))
    if v is None and len(span) > 1 and span[-1] in SCALE:
        w = L.spelled_value(" ".join(span[:-1]))
        if w is not None:
            from decimal import Decimal

            return _canon(Decimal(str(w[0])) * SCALE[span[-1]])
    return _canon(v[0]) if v else None


def units(text: str) -> list[str]:
    """Number spans become one value unit ("#2011", "#5%"); other words stay words. "$5" -> "#5", "dollars"."""
    ws = _words(text)
    is_num = number_span(ws)
    out, i = [], 0
    while i < len(ws):
        if re.search(r"\d", ws[i]):
            raw = ws[i]
            v = re.sub(r"[^\d.]", "", raw.replace(",", "")).rstrip(".")
            pct = "%" in raw
            if i + 1 < len(ws) and ws[i + 1] in SCALE and v:  # "2.5 million" = one value
                from decimal import Decimal

                v = str(Decimal(v) * SCALE[ws[i + 1]])
                i += 1
            if i + 1 < len(ws) and ws[i + 1] == "percent":
                pct = True
                i += 1
            out.append("#" + _canon(v) + ("%" if pct else ""))
            if "$" in raw:
                out.append("dollars")
            i += 1
            continue
        if not is_num[i]:
            out.append(ws[i])
            i += 1
            continue
        j = i
        while j < len(ws) and is_num[j] and not re.search(r"\d", ws[j]):
            j += 1
        span = ws[i:j]
        pct = span[-1] == "percent"
        if pct:
            span = span[:-1]
        v = _spelled(span) if span else None
        out.append("#" + (v if v is not None else " ".join(span)) + ("%" if pct else ""))
        i = j
    return out


def edge_ok(label: str, hyp: str) -> bool:
    """D-1b: first and last unit plus the neighbouring unit agree (numbers by value, other words as words)."""
    a, b = units(label), units(hyp)
    if not a or not b:
        return False
    k = min(2, len(a))
    return a[:k] == b[:k] and a[-k:] == b[-k:]


def num_values(text: str) -> list[str]:
    return [u for u in units(text) if u.startswith("#")]


def has(text: str, w: str) -> bool:
    return re.search(r"(?<![\w'])" + re.escape(w) + r"(?![\w'])", text, re.I) is not None


def run(root: str) -> dict:
    rnd = random.Random(3)
    st = collections.Counter()
    ex = collections.defaultdict(list)
    agree = collections.defaultdict(lambda: collections.Counter())
    year_ratio = collections.Counter()
    os.makedirs(f"{root}/crops3_ok", exist_ok=True)

    def note(name, x, extra=""):
        st[name] += 1
        if len(ex[name]) < 200:
            ex[name].append(f"{x['text']}  ||  {extra}")

    for f in sorted(glob.glob(f"{root}/crops3/*.jsonl")):
        tag = os.path.basename(f)[:-6]
        if not os.path.exists(f"{root}/dec3/{tag}.json"):
            st["shards_without_decode"] += 1
            continue
        dec = json.load(open(f"{root}/dec3/{tag}.json"))
        rows = {r["utt_id"]: r for r in map(json.loads, open(f"{root}/rows3/{tag}.jsonl"))}
        vt = json.load(open(f"{root}/veto3/{tag}.json")) if os.path.exists(f"{root}/veto3/{tag}.json") else None
        if vt is None:
            st["shards_without_veto"] += 1
        keep = []
        for x in map(json.loads, open(f)):
            st["crops"] += 1
            st["onset_crops"] += x["onset"]
            d = dec.get(x["audio_filepath"])
            if d is None:
                note("no_decode", x)
                continue
            t, v10, stock = x["text"], d["v10"], d["stock"]
            # caption artifacts: "sixteen.Bolded" (no space after a stop), "Jiaga ," (space before punctuation)
            t2 = re.sub(r"\s+([,.?!;:])(?=\s|$)", r"\1", re.sub(r"([a-z])([.?!])([A-Z])", r"\1\2 \3", t))
            if t2 != t:
                st["caption_spacing_fixed"] += 1
                x = {**x, "text": t2}
                t = t2
            # F2: label agreement per span kind, before any filter (by value, digits or words)
            lv = num_values(t)
            for kind in x["kinds"]:
                agree[kind]["n"] += 1
                agree[kind]["v10"] += bool(lv) and lv == num_values(v10)
                agree[kind]["stock"] += bool(lv) and lv == num_values(stock)
            if WRITTEN_FORM.search(t) or "+" in t or "⁇" in t:
                note("label_written_form", x)
                continue
            if UNIT_ABBR.search(t):
                note("unit_abbreviation", x)
                continue
            rec = rows.get(x["src"], {})
            spans = [s for s in rec.get("spans") or [] if s.get("decided")]
            hub = (vt or {}).get("hubert", {}).get(x["src"], {})
            vetoed = (vt or {}).get("vetoed", {}).get(x["src"], [])
            if any(has(t, w) for w in vetoed):
                note("hubert_veto", x, " / ".join(vetoed))
                continue
            # D-4: "two thousand X" read where v10pt heard something else: keep only with margin AND HuBERT agreement
            d4_drop = False
            for i, s in enumerate(rec.get("spans") or []):
                if not s.get("decided") or s["kind"] != "year":
                    continue
                pick = s["cands"][s["pick"]]
                if not has(t, pick):
                    continue
                if pick.startswith("two thousand"):
                    year_ratio["two_thousand"] += 1
                elif pick.startswith("twenty"):
                    year_ratio["twenty"] += 1
                v10_digits = re.search(r"\b(?:19|20)\d\d\b", v10) is not None and not re.search(r"\btwenty\b", v10)
                if pick.startswith("two thousand") and not has(v10, pick) and not v10_digits:
                    h = hub.get(str(i))
                    ok = s.get("mc", 0) >= D4_MARGIN and h is not None and h["best"] == pick
                    st["d4_disagree"] += 1
                    st["d4_kept"] += ok
                    if not ok:
                        d4_drop = True
                        note("d4_year_disagree", x, f"v10: {v10}")
            if d4_drop:
                continue
            if not edge_ok(t, v10):
                note("d1b_edge", x, f"v10: {v10}")
                st["d1b_edge_onset"] += x["onset"]
                continue
            t2, n = merge(t, v10)
            if n:
                st["fillers_rows"] += 1
                st["fillers_words"] += n
                if len(ex["fillers_restored"]) < 200:
                    ex["fillers_restored"].append(f"{t}  ->  {t2}")
                x = {**x, "text": t2, "text_caption": t}
            keep.append(x)
            st["kept"] += 1
            st["kept_onset"] += x["onset"]
            st["kept_s"] += x["duration"]
        with open(f"{root}/crops3_ok/{tag}.jsonl", "w") as fo:
            for x in keep:
                fo.write(json.dumps(x, ensure_ascii=False) + "\n")
    st["kept_h"] = round(st.pop("kept_s", 0) / 3600, 2)
    rep = {"counts": dict(st), "year_ratio": dict(year_ratio),
           "agreement": {k: {"n": c["n"], "v10": round(c["v10"] / c["n"], 3), "stock": round(c["stock"] / c["n"], 3)}
                         for k, c in sorted(agree.items(), key=lambda kv: -kv[1]["n"])},
           "examples": {k: rnd.sample(v, min(30, len(v))) for k, v in ex.items()}}
    json.dump(rep, open(f"{root}/filter3_report.json", "w"), indent=1, ensure_ascii=False)
    return rep


if __name__ == "__main__":
    r = run(sys.argv[1])
    print(json.dumps({k: r[k] for k in ("counts", "year_ratio", "agreement")}, indent=1))
