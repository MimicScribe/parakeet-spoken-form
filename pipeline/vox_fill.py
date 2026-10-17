"""Targeted parliament fill (Granary-VoxPopuli text index): million / billion figures, day-month dates and, for
Italian, clock times, for languages whose real speech is thin in those classes (de / it / nl / pt).

Pure selection (no audio): `classes(text, lang)` flags each caption, `select(rows, ...)` fills per-class quotas with
caps per sentence template and per session, never from a held-out day or an already-cut clip, and only from the
`years_per_lang` debate years that hold the most quota candidates (each debate year is one cut job).
ml_p3.granvox_fill runs it next to the index and cuts the clips (granvox_cut_lang).
"""

import collections
import random
import re

MONTHS = {
    "de": "januar|jänner|februar|märz|april|mai|juni|juli|august|september|oktober|november|dezember",
    "it": "gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre",
    "nl": "januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december",
    "pt": "janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro",
}
# day number + month as captions write it: de "am 6. Dezember", it "il 3 maggio" / "1° maggio", nl "op 12 maart",
# pt "3 de maio" / "1.º de maio"
DATE_RE = {
    "de": r"(?<![\d.,])(?:[12]?\d|3[01])\.\s?(?:{m})\b",
    "it": r"(?<![\d.,])(?:[12]?\d|3[01])°?\s(?:{m})\b",
    "nl": r"(?<![\d.,])(?:[12]?\d|3[01])\s(?:{m})\b",
    "pt": r"(?<![\d.,])(?:[12]?\d|3[01])(?:\.?º)?\s(?:de\s)?(?:{m})\b",
}
MILLION = {"de": r"millionen|million|mio\.?", "it": r"milioni|milione", "nl": r"miljoen", "pt": r"milhões|milhão"}
BILLION = {"de": r"milliarden|milliarde|mrd\.?", "it": r"miliardi|miliardo", "nl": r"miljard(?:en)?",
           "pt": r"mil milhões|bilhões|bilhão|biliões|bilião"}
# Italian clock: "alle 15.30", "ore 12", "dalle 9 alle 12.30"
IT_CLOCK = (r"\b(?:alle|dalle|ore|entro le|verso le|fino alle)\s+(?:ore\s+)?(?:[01]?\d|2[0-3])(?:[.:][0-5]\d)?(?![\d%]|[.,]\d)"
            r"(?!\s*(?:%|per cento|euro|milion|miliard|anni|mesi|giorni|volte|paesi|stati|deputati|membri))")
NUM = r"(?<![\w.,])\d{1,3}(?:[.,]\d{1,3})?\s?"


def classes(text: str, lang: str) -> dict:
    """Flags of one caption: scale_m / scale_b (a number + million / billion word), date (day + month), clock
    (Italian only; the other languages keep the parliament batch's clock selection)."""
    t = text.lower()
    t_m = t.replace("mil milhões", "") if lang == "pt" else t  # pt "mil milhões" is a billion
    out = {"scale_m": bool(re.search(NUM + r"(?:" + MILLION[lang] + r")(?!\w)", t_m)),
           "scale_b": bool(re.search(NUM + r"(?:" + BILLION[lang] + r")(?!\w)", t)),
           "date": bool(re.search(DATE_RE[lang].format(m=MONTHS[lang]), t)),
           "clock": lang == "it" and bool(re.search(IT_CLOCK, t))}
    return out


def template(text: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"[^\w\s]", "", text.lower()))[:60]


def session(utt_id: str) -> str:
    return utt_id.rsplit("_", 1)[0]


def select(rows: list, lang: str, quotas: dict, held: set, exclude: set, seed: int = 1, years_per_lang: int = 4,
           per_template: int = 3, per_session: int = 8, min_s: float = 3.0, max_s: float = 30.0) -> tuple[list, dict]:
    """Rarest class first; a clip counts toward every quota it carries. Returns (selected rows with "why", stats)."""
    rows = [r for r in rows if r["day"] not in held and r["utt_id"] not in exclude
            and min_s <= len(r["text"]) / 15 <= max_s]
    cls = {r["utt_id"]: classes(r["text"], lang) for r in rows}
    quotas = {k: v for k, v in quotas.items() if v > 0}
    cand = [r for r in rows if any(cls[r["utt_id"]][k] for k in quotas)]
    by_year = collections.Counter(r["day"][:4] for r in cand)
    years = {y for y, _ in by_year.most_common(years_per_lang)}
    pool = [r for r in cand if r["day"][:4] in years]
    random.Random(f"{seed}-{lang}-fill").shuffle(pool)
    t_n, s_n, sel = collections.Counter(), collections.Counter(), []
    have = collections.Counter()
    taken = set()
    for k in sorted(quotas, key=lambda k: sum(cls[r["utt_id"]][k] for r in pool)):
        for r in pool:
            if have[k] >= quotas[k]:
                break
            u = r["utt_id"]
            if u in taken or not cls[u][k] or t_n[template(r["text"])] >= per_template or s_n[session(u)] >= per_session:
                continue
            taken.add(u)
            t_n[template(r["text"])] += 1
            s_n[session(u)] += 1
            sel.append(dict(r, why=k))
            for kk in quotas:
                have[kk] += cls[u][kk]
    st = {"lang": lang, "clips": len(sel), "hours_est": round(sum(len(r["text"]) / 15 for r in sel) / 3600, 2),
          "classes": dict(have), "quotas": quotas, "debate_years": sorted(years),
          "index_avail": {k: sum(cls[r["utt_id"]][k] for r in rows) for k in quotas},
          "sessions": len(s_n), "excluded_prior": len(exclude), "held_days": len(held)}
    return sel, st


def pilot(sel: list, n: int = 40, seed: int = 1) -> list:
    """An even per-class pilot subset of the selection (round-robin over the "why" classes)."""
    by = collections.defaultdict(list)
    for r in sel:
        by[r["why"]].append(r)
    rng = random.Random(f"{seed}-pilot")
    for v in by.values():
        rng.shuffle(v)
    out, ks = [], sorted(by)
    while len(out) < n and any(by[k] for k in ks):
        for k in ks:
            if by[k] and len(out) < n:
                out.append(by[k].pop())
    return out
