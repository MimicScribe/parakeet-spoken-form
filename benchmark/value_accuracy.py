#!/usr/bin/env python3
"""Value accuracy on real held-out speech: is each number's VALUE right, and, when it is, is it written the way the
speaker said it?

usage: value_accuracy.py --refs DIR --arm stock=DIR [--arm B=DIR ...] [--arm oracle] [--base stock]
                         [--langs de,es,...] [--e22] [--min-oracle 95] [--json out.json]

--refs DIR   heldout_<lang>.jsonl label rows: {id, lang, src, caption, text, spans: [{text, kind, reading}]}.
             `caption` is the source dataset's own transcript (numbers as digits; for a piece cut from a longer clip
             the scored reference is ref_caption: the label with its spans written back as caption text), `text` the spoken-form label,
             `spans[i].text` the caption number, `spans[i].reading` the words heard in the audio.
--arm N=DIR  decodes of one model: DIR/heldout_<lang>.jsonl rows {id, hyp}. `--arm oracle` (no DIR) scores the
             spoken-form label itself as the hypothesis: the reader's recall on spoken readings (see oracle_recall.py).
--base N     the arm the others are compared with (paired, same spans); default the first arm.
--e22        also score English Earnings-22 decodes (DIR/e22_numbers.jsonl rows {id, ref, hyp}; reference numbers
             only, no form labels; reported on its own row, never pooled with the labelled spans).

Metrics, per span (one labelled number of the caption):
  value_right     the span's slots (common.span_slots: PH digit run, T clock, P price, D decimal, PCT, N number)
                  all match the hypothesis in caption order under score_values.match_slots, the matcher the seam
                  scorers use, over the reader readings of the hypothesis (common.readings_of); digits or words
                  both count.
  words_right     value right AND written as words (a number word of that value; for a digit run, the digits are
                  not written as digits).
  form_match      value right AND the hypothesis contains the label reading word for word (folded, in order).
  form_match_join the same with spaces and hyphens ignored ("venti uno" = "ventuno"; "zwei und zwanzig" = "zweiundzwanzig").
Scopes: all spans; scorable spans (the label reading itself scores value-right, so a digits model is not credited
where the reader could not credit a perfect spoken-form model); non-canonical spans (scorable, and the label reading
differs from the grammar's canonical reading `spans[i].canonical`: "zwanzig zehn" for 2010, "treize cents" for 1300).
Rates are reported per language and pooled; CIs are 95% cluster-bootstrap intervals (clusters: common.cluster_of).
Paired differences vs the base use the same clusters. A language whose oracle recall is below --min-oracle is
marked READER<bar and left out of the pooled rows: the reader cannot score that language's spoken forms.
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

N = C.N
RIGHT = ("WORDS_RIGHT_VALUE", "DIGITS_RIGHT_VALUE")
WAVE1 = ("de", "es", "fr", "it", "nl", "pt")


def _find_seq(hay: list, needle: list, start: int) -> int:
    """index of the first contiguous occurrence of needle in hay at or after start, else -1"""
    k = len(needle)
    if not k:
        return -1
    for i in range(start, len(hay) - k + 1):
        if hay[i:i + k] == needle:
            return i
    return -1


SCALES = tuple(10 ** k for k in (3, 6, 9, 12))


def _match_ref(rnum, iso) -> bool:
    """caption number rnum (read in context) stands for the isolated span number iso: a shared value, or the span's
    value times a scale word that follows it in the caption ("1,7 Milliarden" -> 1.7e9)"""
    if rnum.vals & iso.vals:
        return True
    return any(r == i * s for r in rnum.vals for i in iso.vals for s in SCALES)


def _digits_of(num) -> set:
    return {re.sub(r"\D", "", format(v.normalize(), "f")) for v in num.vals}


def ref_caption(row: dict) -> str:
    """The written reference of exactly the labelled audio: the spoken-form label with each span's reading replaced
    by the span's caption text (num_chars give the reading's place in the label). A row cut from a longer clip keeps
    its parent's whole caption in `caption`; this reference covers only the piece. Falls back to `caption` when the
    places are missing."""
    t, spans, nc = row.get("text") or "", row.get("spans") or [], row.get("num_chars") or []
    if not nc or len(nc) != len(spans):
        return row["caption"]
    out, k = [], 0
    for sp, (a, b) in sorted(zip(spans, nc), key=lambda x: x[1][0]):
        if a < k:
            return row["caption"]
        out += [t[k:a], sp["text"]]
        k = b
    return "".join(out + [t[k:]])


def tie_spans(row: dict, lang: str) -> tuple:
    """(R, ties): R = the reference's (ref_caption) numbers read IN CONTEXT by the public reader (a scale word after a number counts;
    "1.000" is 1000 or 1), ties = one entry per span: ("slot", base, value) for a digit run (phone, code, chain,
    version, alnum: PH) or a clock (T); ("nums", [R indices]) for anything else, the next caption numbers that share a
    value with the span read alone (or that value times a scale word); ("none",) when the span cannot be tied."""
    R = N.extract_numbers(ref_caption(row), lang, keep_weak=False)
    ptr, ties = 0, []
    for sp in row.get("spans") or []:
        kind = sp.get("kind") or ""
        slots = C.span_slots(sp, lang) if (kind in C.DIGIT_RUN_KINDS or kind == "clock") else []
        slots = [x for x in slots if x[0] in ("PH", "T")]
        if slots:
            base, v = slots[0]
            if base == "PH":  # skip the caption numbers of this digit run
                got, j = "", ptr
                while j < len(R) and len(got) < len(v):
                    nxt = next((x for x in sorted(_digits_of(R[j]), key=len) if v.startswith(got + x)), None)
                    if nxt is None:
                        break
                    got += nxt
                    j += 1
                if got == v:
                    ptr = j
            else:
                h, m = (int(x) for x in v.split(":"))
                for j in range(ptr, min(len(R), ptr + 3)):
                    if h in R[j].vals:
                        ptr = j + 1
                        if m and j + 1 < len(R) and m in R[j + 1].vals:
                            ptr = j + 2
                        break
            ties.append(("slot", base, v))
            continue
        iso = N.extract_numbers(sp.get("text") or "", lang, keep_weak=False)
        got = []
        for x in iso:
            for j in range(ptr, len(R)):
                if _match_ref(R[j], x):
                    got.append(j)
                    ptr = j + 1
                    break
        ties.append(("nums", got) if got and len(got) == len(iso) else ("none",))
    return R, ties


def span_outcomes(row: dict, hyp: str, lang: str) -> list:
    """One dict per span: {kind, n_nums, value_right, words_right, form_match, form_match_join, noncanonical}.

    Value (tie_spans): a span tied to caption numbers is right when number_outcomes.number_outcomes(caption, hyp)
    scores each of them right (value-aligned: digits or words both count); a whole number of 3+ digits that it misses
    also counts when said in digit groups (the slot matcher's PH rule: "vingt quatre zero trois" = 2403). A digit run or
    clock span is a score_values slot (PH: the hypothesis' parsed numbers, joined, equal the digit run; T: hour and
    minutes) matched against the reader readings of the hypothesis (common.readings_of). n_nums == 0: the span could
    not be tied to the caption (unscorable, counted but left out of the rates)."""
    R, ties = tie_spans(row, lang)
    outs = N.number_outcomes(ref_caption(row), hyp or "", lang)  # one row per R, in R order
    readings = None
    hdig = re.sub(r"\D", "", hyp or "")
    htok = C.fold_tokens(hyp)
    hjoin = "".join(htok)
    tptr = cptr = 0
    res = []
    for sp, tie in zip(row.get("spans") or [], ties):
        kind = sp.get("kind") or ""
        vr = wr = False
        n_nums = 0
        if tie[0] == "slot":
            readings = readings or C.readings_of(hyp or "", lang)
            base, v = tie[1], tie[2]
            vr, n_nums = C.SV.slot_any(base, v, readings), 1
            wr = vr and (v not in hdig if base == "PH" else not re.search(r"\d", hyp or ""))
        elif tie[0] == "nums":
            got = tie[1]
            n_nums = len(got)
            vr = all(outs[j]["outcome"] in RIGHT for j in got)
            wr = vr and all(outs[j]["outcome"] == "WORDS_RIGHT_VALUE" for j in got)
            d = re.sub(r"\D", "", sp.get("text") or "")
            if n_nums == 1 and not vr and kind in ("int", "unit") and len(d) >= 3 and \
                    re.fullmatch(r"[\d.,\s\u00a0]+", (sp.get("text") or "").strip().split(" ")[0] or "x"):
                readings = readings or C.readings_of(hyp or "", lang)
                vr = C.SV.slot_any("PH", d, readings)
                wr = vr and d not in hdig
        rtok = C.fold_tokens(sp.get("reading") or "")
        fm = fj = False
        if vr and rtok:
            i = _find_seq(htok, rtok, tptr)
            if i >= 0:
                fm, tptr = True, i + len(rtok)
            j = hjoin.find("".join(rtok), cptr)
            if j >= 0:
                fj, cptr = True, j + len("".join(rtok))
        canon = C.fold_tokens(sp.get("canonical") or "")
        res.append({"kind": kind, "n_nums": n_nums, "value_right": bool(vr), "words_right": bool(wr),
                    "form_match": fm, "form_match_join": fj,
                    "noncanonical": bool(canon) and "".join(canon) != "".join(rtok)})
    return res


def load_refs(refs_dir: Path, langs) -> dict:
    out = {}
    for p in sorted(refs_dir.glob("heldout_*.jsonl")):
        lang = p.stem.split("_", 1)[1]
        if langs and lang not in langs:
            continue
        rows = [r for r in C.read_jsonl(p) if r.get("spans")]
        if rows:
            out[lang] = rows
    return out


def load_hyps(d: Path, lang: str) -> dict:
    p = d / f"heldout_{lang}.jsonl"
    if not p.exists():
        return {}
    return {r["id"]: r.get("hyp", "") for r in C.read_jsonl(p)}


def score_arm(refs: dict, hyps_by_lang) -> dict:
    """{lang: [(cluster, row id, span index, outcome dict)]}; hyps_by_lang(lang) -> {id: hyp} or None for the oracle.
    A row without a decode is an error (never a silent drop)."""
    out = {}
    for lang, rows in refs.items():
        H = hyps_by_lang(lang)
        items = []
        missing = [r["id"] for r in rows if H is not None and r["id"] not in H]
        if missing:
            raise KeyError(f"{lang}: {len(missing)} rows without a decode, e.g. {missing[:3]}")
        for r in rows:
            hyp = r["text"] if H is None else H[r["id"]]
            for k, o in enumerate(span_outcomes(r, hyp, lang)):
                items.append((C.cluster_of(r["id"], r.get("src", "")), r["id"], k, o))
        out[lang] = items
    return out


def summarize(items: list) -> dict:
    sc = [x for x in items if x[3]["n_nums"] > 0]
    n = len(sc)
    vr = sum(x[3]["value_right"] for x in sc)
    cl = collections.defaultdict(lambda: [0, 0])
    for c, _i, _k, o in sc:
        cl[c][0] += o["value_right"]
        cl[c][1] += 1
    d = {"spans": len(items), "scored": n,
         "clusters": len(cl), "value_right": C.pct(vr, n), "value_right_ci95": C.ratio_ci({k: tuple(v) for k, v in cl.items()}),
         "words_right": C.pct(sum(x[3]["words_right"] for x in sc), n),
         "form_given_right": C.pct(sum(x[3]["form_match"] for x in sc), vr),
         "form_join_given_right": C.pct(sum(x[3]["form_match_join"] for x in sc), vr)}
    return d


def paired(run: list, base: list) -> dict:
    """value-right difference run - base over the spans both scored (same row id and span index)"""
    b = {(x[1], x[2]): x for x in base if x[3]["n_nums"] > 0}
    cl = collections.defaultdict(lambda: [0, 0, 0])
    better = worse = 0
    for c, i, k, o in run:
        if (i, k) not in b or o["n_nums"] == 0:
            continue
        ob = b[(i, k)][3]
        cl[c][0] += o["value_right"]
        cl[c][1] += ob["value_right"]
        cl[c][2] += 1
        better += o["value_right"] and not ob["value_right"]
        worse += ob["value_right"] and not o["value_right"]
    n = sum(v[2] for v in cl.values())
    a = sum(v[0] for v in cl.values())
    bb = sum(v[1] for v in cl.values())
    return {"n": n, "diff_pp": round(100.0 * (a - bb) / n, 2) if n else None,
            "ci95": C.paired_diff_ci({k: tuple(v) for k, v in cl.items()}), "better": better, "worse": worse}


def e22_items(d: Path) -> list:
    """English Earnings-22: every reference number (digits or words) of each row; outcome value_right only."""
    p = d / "e22_numbers.jsonl"
    if not p.exists():
        return []
    items = []
    for r in C.read_jsonl(p):
        ref = r.get("ref") or ""
        for k, o in enumerate(N.number_outcomes(ref, r.get("hyp", ""), "en")):
            vr = o["outcome"] in RIGHT
            items.append((C.cluster_of(r["id"]), r["id"], k, {"kind": None, "n_nums": 1, "value_right": vr,
                          "words_right": o["outcome"] == "WORDS_RIGHT_VALUE", "form_match": False,
                          "form_match_join": False}))
    return items


def _subset(items: list, keep: set) -> list:
    return [x for x in items if (x[1], x[2]) in keep]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refs", required=True, type=Path)
    ap.add_argument("--arm", action="append", required=True, help="NAME=DIR, or 'oracle'")
    ap.add_argument("--base")
    ap.add_argument("--langs", default="")
    ap.add_argument("--e22", action="store_true")
    ap.add_argument("--min-oracle", type=float, default=95.0)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    langs = [x for x in a.langs.split(",") if x]
    refs = load_refs(a.refs, langs)
    arms, dirs = {}, {}
    for spec in a.arm:
        name, _, d = spec.partition("=")
        dirs[name] = Path(d) if d else None
        arms[name] = score_arm(refs, (lambda lang: None) if name == "oracle" else (lambda lang, d=Path(d): load_hyps(d, lang)))
    oracle = arms.get("oracle") or score_arm(refs, lambda lang: None)
    recall = {L: summarize(oracle[L]) for L in refs}
    ok_langs = [L for L in refs if (recall[L]["value_right"] or 0) >= a.min_oracle]
    # reader-scorable spans: the label reading itself scores value-right (a digit model is not credited where the
    # reader could not have credited a perfect spoken-form model)
    keep = {(x[1], x[2]) for L in refs for x in oracle[L] if x[3]["value_right"]}
    if a.e22:
        for name, d in dirs.items():
            if d is not None:
                arms[name]["en_e22"] = e22_items(d)
        keep |= {(x[1], x[2]) for name in arms for x in arms[name].get("en_e22", [])}
    base = a.base or next(n for n in arms if n != "oracle")
    res = {"langs": {}, "reader_ok": ok_langs, "min_oracle": a.min_oracle,
           "oracle_recall": {L: recall[L]["value_right"] for L in refs}}
    groups = [(L, [L]) for L in list(refs) + (["en_e22"] if a.e22 else [])]
    groups += [("POOLED_ok", ok_langs), ("POOLED_wave1", [L for L in WAVE1 if L in ok_langs])]
    print(f"{'lang':13s} {'arm':10s} {'spans':>6s} {'clus':>5s} {'value%':>7s} {'CI95':>16s} {'words%':>7s} "
          f"{'form|right':>10s} {'join|right':>10s}  vs {base}: diff pp [CI95] better/worse")
    noncanon = {(x[1], x[2]) for L in refs for x in oracle[L] if x[3]["noncanonical"]} & keep
    notes = {"all": "", "scorable": " (the label reading scores right: the fair set for digits vs words)",
             "noncanonical": " (scorable spans said in a reading OTHER than the grammar's canonical one: where a verbalizer must guess)"}
    for scope in ("all", "scorable", "noncanonical"):
        print(f"--- {scope} spans" + notes[scope])
        for gname, members in groups:
            res["langs"].setdefault(gname, {})[scope] = {}
            for name, per in arms.items():
                items = [x for k in members for x in per.get(k, [])]
                bitems = [x for k in members for x in arms[base].get(k, [])]
                if scope != "all":
                    sel = keep if scope == "scorable" else noncanon
                    items, bitems = _subset(items, sel), _subset(bitems, sel)
                if not items:
                    continue
                s = summarize(items)
                if name != base:
                    s["vs_base"] = paired(items, bitems)
                res["langs"][gname][scope][name] = s
                flag = "" if gname.startswith("POOLED") or gname in ok_langs or gname == "en_e22" else f"  READER<{a.min_oracle:g}"
                vb = s.get("vs_base")
                vbs = f"  {vb['diff_pp']:+.2f} {vb['ci95']} {vb['better']}/{vb['worse']}" if vb and vb["n"] else ""
                print(f"{gname:13s} {name:10s} {s['scored']:6d} {s['clusters']:5d} {s['value_right']!s:>7s} "
                      f"{str(s['value_right_ci95']):>16s} {s['words_right']!s:>7s} {s['form_given_right']!s:>10s} "
                      f"{s['form_join_given_right']!s:>10s}{vbs}{flag}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    return res


if __name__ == "__main__":
    main()
