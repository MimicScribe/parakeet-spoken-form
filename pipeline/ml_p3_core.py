"""Pure (no Modal, no torch) parts of retrain-4 P3: row building from a reading decision, quiet-edge flag, stats."""

import re
import unicodedata
from collections import Counter, defaultdict

import ml_cands as M

MAX_CER = 0.25


def letters(s: str) -> str:
    s = s.lower().replace("ß", "ss")
    s = "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", s)


def cer(a: str, b: str) -> float:
    a, b = letters(a), letters(b)
    if not a:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1] / len(a)


def edge_quiet(x) -> bool:
    """mix_build.trim_edges rule: a quiet (< 0.3 x speech level) 20 ms frame in each outer 160 ms."""
    import numpy as np

    n = len(x) // 320
    if n < 20:
        return False
    fr = np.sqrt(np.mean(np.asarray(x[:n * 320]).reshape(n, 320) ** 2, axis=1) + 1e-12)
    speech = np.median(fr[fr > np.percentile(fr, 30)])
    q = 0.3 * speech
    return bool(fr[:8].min() <= q and fr[-8:].min() <= q)


def render_cap(cap, sp, picks):
    """M.render, capitalising a reading that opens a sentence ("2009 wurde" -> "Zweitausendneun wurde")."""
    out, pos = [], 0
    for s, c in zip(sp, picks):
        out.append(cap[pos:s.start])
        w = s.cands[c]
        if not re.search(r"[^\W\d_]", "".join(out)[-40:]) or re.search(r"[.!?]\s*$", "".join(out)):
            w = w[:1].upper() + w[1:]
        out.append(w)
        pos = s.end
    out.append(cap[pos:])
    return "".join(out)


def make_row(row: dict, r: dict, x=None) -> dict:
    """r = ml_decide.decide(...). Returns {"row": new row} or {"reason": why dropped}."""
    cap = row["text"]
    sp = M.spans(cap, row["lang"])
    ds = r["spans"]
    if not sp or len(sp) != len(ds):
        return {"reason": "no_spans" if not sp else "span_mismatch"}
    bad = [d for d in ds if not d["decided"]]
    if bad:
        return {"reason": "no_candidates" if any(d.get("why") for d in bad) else "undecided"}
    picks = [d["pick"] for d in ds]
    label = render_cap(cap, sp, picks)
    if re.search(r"\d", label):
        return {"reason": "digit_left"}
    c = cer(label, r["greedy"])
    if c > MAX_CER:
        return {"reason": "cer", "cer": c}
    spans = [{"text": d["text"], "kind": d["kind"], "reading": d["cands"][d["pick"]], "canonical": d["cands"][0],
              "margin": d["margin"], "n_cands": len(d["cands"])} for d in ds]
    nr = {"audio_filepath": row["audio_filepath"], "duration": row["duration"], "text": label, "lang": row["lang"],
          "src": row["src"], "id": row["id"], "origin": row["origin"], "caption": cap, "cer_ctc": round(c, 3),
          "spans": spans}
    if x is not None:
        nr["edge_quiet"] = edge_quiet(x)
    return {"row": nr}


class Acc:
    def __init__(self, lang):
        self.lang = lang
        self.rows_in = Counter()
        self.hours_in = Counter()
        self.rows_out = Counter()
        self.hours_out = Counter()
        self.reasons = Counter()
        self.span_tot = Counter()  # kind -> spans seen
        self.span_cand = Counter()  # kind -> spans with candidates
        self.span_dec = Counter()  # kind -> decided
        self.noncanon = Counter()
        self.noncanon_diffwords = []
        self.edge = Counter()
        self.examples = []

    def add_skip(self, row, why):
        self.rows_in[row["origin"]] += 1
        self.hours_in[row["origin"]] += row["duration"] / 3600
        self.reasons[f"{row['origin']}:{why}"] += 1

    def add(self, row, r, res):
        o = row["origin"]
        self.rows_in[o] += 1
        self.hours_in[o] += row["duration"] / 3600
        for d in r["spans"]:
            self.span_tot[(o, d["kind"])] += 1
            if "pick" in d:
                self.span_cand[(o, d["kind"])] += 1
                if d["decided"]:
                    self.span_dec[(o, d["kind"])] += 1
                    if d["pick"] != 0:
                        self.noncanon[(o, d["kind"])] += 1
        if "row" in res:
            nr = res["row"]
            self.rows_out[o] += 1
            self.hours_out[o] += nr["duration"] / 3600
            self.edge[(o, nr.get("edge_quiet"))] += 1
            for s in nr["spans"]:
                if s["reading"] != s["canonical"]:
                    from ml_decide import ending_only

                    if not ending_only(s["reading"], s["canonical"]) and len(self.noncanon_diffwords) < 40:
                        self.noncanon_diffwords.append({"id": nr["id"], "caption_span": s["text"], "kind": s["kind"],
                                                        "reading": s["reading"], "canonical": s["canonical"]})
            self.examples.append(nr)
        else:
            self.reasons[f"{o}:{res['reason']}"] += 1

    def stats(self, kept):
        import random

        def agg(c):
            return {f"{o}/{k}": v for (o, k), v in sorted(c.items())}

        def tot(c, o):
            return sum(v for (oo, _), v in c.items() if oo == o)

        out = {"lang": self.lang, "max_cer": MAX_CER,
               "rows_in": dict(self.rows_in), "rows_out": dict(self.rows_out),
               "hours_in": {k: round(v, 3) for k, v in self.hours_in.items()},
               "hours_out": {k: round(v, 3) for k, v in self.hours_out.items()},
               "drop_reasons": dict(sorted(self.reasons.items())),
               "span_total": agg(self.span_tot), "span_with_candidates": agg(self.span_cand),
               "span_decided": agg(self.span_dec), "decided_non_canonical": agg(self.noncanon),
               "edge_quiet": {f"{o}/{q}": v for (o, q), v in self.edge.items()}}
        for o in ("d2", "new"):
            c = tot(self.span_cand, o)
            out[f"decide_rate_{o}"] = round(tot(self.span_dec, o) / c, 3) if c else None
        c, dd = sum(self.span_cand.values()), sum(self.span_dec.values())
        out["decide_rate_all"] = round(dd / c, 3) if c else None
        out["hours_out_total"] = round(sum(self.hours_out.values()), 3)
        out["kinds_kept"] = dict(Counter(s["kind"] for nr in kept for s in nr["spans"]))
        out["noncanonical_different_words"] = self.noncanon_diffwords
        rnd = random.Random(7)
        by_kind = defaultdict(list)
        for nr in self.examples:
            for s in nr["spans"]:
                by_kind[s["kind"]].append((s["text"], s["reading"], nr["id"]))
        ex = []
        kinds = sorted(by_kind)
        while len(ex) < 10 and kinds:
            for k in list(kinds):
                if not by_kind[k]:
                    kinds.remove(k)
                    continue
                ex.append(f"[{k}] {self.examples[0]['lang']}  " + " -> ".join(by_kind[k].pop(rnd.randrange(len(by_kind[k])))[:2]))
                if len(ex) >= 10:
                    break
        out["examples"] = ex
        # sample whole rows for eyeballing
        out["row_samples"] = [{"caption": nr["caption"], "label": nr["text"]} for nr in rnd.sample(self.examples, min(15, len(self.examples)))]
        return out
