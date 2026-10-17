"""Decision-set (held-out) parliament pull on UNUSED plenary days (2026-10-17): clips from the Granary-VoxPopuli text
index whose plenary day appears in no training or candidate pass of the language, picked by number class so the
decision set reaches >= `per_class` clips per class and `n_year` year clips. Real speech only, never trained on: the
labelled rows become a validation-only pass (val_<tag>.jsonl, which build4 adds to the held-out set) and the days go
to heldout_<lang>.txt as "# group vox:<day>" lines (granvox_select_lang / vox_en never select them again).

    modal run vox_heldout.py::select-cli --lang es --out sel_es.json --per-class 10 --n-year 40 --n-debate-years 12   # pilot
    modal run vox_heldout.py::select-cli --lang de --out sel_de.json --per-class 40 --classes phone,int_big   # long numbers
    modal run ml_p3.py::granvox_cut_ids --lang es --ids-file sel_es.json --tag <tag>
    ... second transcript, label2 --srcs <tag> --name val_<tag>, judge --name val_<tag>
    python vox_heldout.py held-lines <cands_<tag>.jsonl>  -> lines to append to heldout_<lang>.txt
"""
import collections
import json
import random
import re
import sys

VOL = "/vol"
MONTHS = {"es": "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre",
          "fr": "janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre",
          "en": "january|february|march|april|may|june|july|august|september|october|november|december",
          "de": "januar|jänner|februar|märz|april|mai|juni|juli|august|september|oktober|november|dezember",
          "it": "gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre",
          "nl": "januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december",
          "pt": "janeiro|fevereiro|março|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro"}
ORD = {"es": r"\b\d+\s?(?:º|ª|\.º|\.ª)|\b\d+(?:er|o|a)\b", "fr": r"\b\d+(?:e|er|ère|ere|ème|eme|re)\b",
       "en": r"\b\d+(?:st|nd|rd|th)\b",
       # de: "3." is an ordinal only before a noun that takes one (a bare "3." ends a sentence or is a date)
       "de": r"\b\d{1,2}\.(?=\s+(?:tagesordnungspunkt|artikel|änderungsantrag|absatz|jahrhundert|platz|mal|kapitel|"
             r"sitzung|lesung|punkt|stelle|ebene|rang|weltkrieg)\b)",
       "it": r"\b\d+(?:º|ª|°)|\b\d+(?:o|a|esimo|esima)\b", "nl": r"\b\d+(?:e|de|ste)\b",
       "pt": r"\b\d+\s?(?:º|ª|\.º|\.ª)|\b\d+(?:o|a)\b"}
DECADE = {"es": r"\baños\s+(?:19)?\d0\b|\b\d0s\b", "fr": r"\bannées\s+(?:19)?\d0\b|\b\d0s\b", "en": r"\b(?:19|20)?\d0s\b|'\d0s\b",
          "de": r"\bjahre[n]?\s+(?:19)?\d0\b|\b(?:19)?\d0er\b", "it": r"\banni\s+['’]?(?:19)?\d0\b",
          "nl": r"\bjaren\s+(?:19)?\d0\b|\b\d0s\b", "pt": r"\banos\s+(?:19)?\d0\b"}
# abbreviations only: the labeller makes a unit span of "40 km", a plain count of "40 kilómetros"
UNIT = r"\d\s?(?:km|kg|m2|m²|m3|m³|cm|mm|gw|mw|kw|kwh|twh|ha|t)\b|\d\s?°"
PHONE = r"(?<![\d.,])\d{2,4}(?:[ .-]\d{2,4}){2,4}(?![\d.,])"
ES_CLOCK = r"\b(?:a|desde|hasta|de)\s+las?\s+\d{1,2}(?:[.:]\d\d)?\s*(?:horas?|h\b|de la (?:mañana|tarde|noche))"
CLASSES = ("year", "dec", "pct", "cur", "clock", "phone", "ord", "date", "range", "decade", "unit")
# int_big_digit: int_big written in digits only (no "12 Millionen"); the decision-set gate counts these spans by value
ALL_CLASSES = CLASSES + ("int_big", "int_big_digit")
# count scale words that make "<digits> <scale>" an integer >= 1000 (digits before the word are required)
SCALE = {"de": r"millionen|million|milliarden|milliarde|billionen|billion|tausend|mio\.?|mrd\.?",
         "it": r"milioni|milione|miliardi|miliardo|mila|mln|mld|migliaia",
         "nl": r"miljoen|miljard|duizend|mln|mld",
         "pt": r"milhões|milhão|milhoes|milhao|mil milhões|mil milhoes|mil|biliões|bilião|bilhões|bilhão",
         "es": r"millones|millón|mil millones|mil|billones",
         "fr": r"millions|million|milliards|milliard|mille|mil",
         "en": r"millions?|billions?|trillions?|thousand|bn"}
THOUSAND_GROUPED = r"(?<![\d.,])\d{1,3}(?:[.   ]\d{3})+(?![\d])(?![.,]\d)"
DATE_NUMERIC = r"^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$"


def is_year(n: str) -> bool:
    return len(n) == 4 and n.isdigit() and 1800 <= int(n) <= 2099


def phone_spans(text: str) -> list:
    """PHONE-shaped digit groups that are really a phone number: >= 7 digits, not a thousands-grouped integer
    ("3 500 000", "1.250.000"), not a numeric date ("12.05.2019") and not a list of years ("2019 2020 2021")."""
    out = []
    for m in re.finditer(PHONE, text):
        g, digits = m.group(0), re.sub(r"\D", "", m.group(0))
        parts = re.split(r"[ .-]", g)
        if len(digits) < 7 or re.fullmatch(THOUSAND_GROUPED, g) or re.fullmatch(DATE_NUMERIC, g) or all(is_year(p) for p in parts):
            continue
        out.append(m.span())
    return out


def has_int_big(text: str, lang: str, scale: bool = True) -> bool:
    """An integer >= 1000 that is not a year: thousands-grouped ("3.500", "3 500 000"), a plain run of >= 4 digits
    that is not a 4-digit 1800-2099 year, or digits before a count scale word ("12 Millionen", "2 mil milhões").
    Phone numbers are blanked first; a decimal tail (",50") means a decimal / price, not an integer."""
    for a, b in reversed(phone_spans(text)):
        text = text[:a] + " " + text[b:]
    t = text.lower()
    if re.search(THOUSAND_GROUPED, text):
        if any(not (re.fullmatch(DATE_NUMERIC, m.group(0)) or m.group(0).startswith("0")) for m in re.finditer(THOUSAND_GROUPED, text)):
            return True
    for m in re.finditer(r"(?<![\d.,])\d{4,}(?![\d])(?![.,]\d)", text):
        if not is_year(m.group(0)):
            return True
    sc = SCALE.get(lang) if scale else None
    return bool(sc and re.search(r"(?<![\d.,])\d+(?:[.,]\d+)?\s?(?:" + sc + r")(?![\wäöüéèà])", t))


def classes(text: str, lang: str, gv_classes) -> dict:
    """ml_p3._gv_classes flags + the classes the decision set also needs (ord, date, decade, unit, phone, int_big)."""
    g = gv_classes(text, lang)
    t = text.lower()
    out = {"year": bool(g["years"]), "dec": g["dec"], "pct": g["pct"], "cur": g["cur"], "range": g["range"],
           # es: _gv_classes also flags durations ("durante 10 horas"); the decision set wants clock times only
           "clock": bool(re.search(ES_CLOCK, t)) if lang == "es" else g["clock"],
           "ord": bool(re.search(ORD[lang], t)), "decade": bool(re.search(DECADE[lang], t)),
           "unit": bool(re.search(UNIT, t)), "phone": bool(phone_spans(text)) and not g["years"],
           "int_big": has_int_big(text, lang), "int_big_digit": has_int_big(text, lang, scale=False),
           "date": bool(re.search(r"\b\d{1,2}(?:er|st|nd|rd|th)?\.?\s+(?:de\s+|di\s+|d'|of\s+)?(?:" + MONTHS[lang] + r")\b", t)
                        or (lang == "en" and re.search(r"\b(?:" + MONTHS["en"] + r")\s+\d{1,2}(?:st|nd|rd|th)?\b", t)))}
    return out


def day_of(i: str):
    m = re.match(r"(\d{8})-\d{4}", str(i))
    return m.group(1) if m else None


def select(rows: list, used_days: set, cls_of, per_class: int = 80, n_year: int = 330, n_debate_years: int = 2,
           per_session: int = 8, per_template: int = 3, seed: int = 1, min_s: float = 3.0, max_s: float = 25.0,
           only: tuple = ()) -> tuple:
    """rows: index rows {utt_id, text, day}. Only days not in used_days; only the n_debate_years debate years with the
    most unused-day rows that carry a rare class (fewer tars to stream). Rarest class first, then years; caps per
    session and per digit-masked template. `only` (e.g. ("phone", "int_big")) restricts the pull to rows that carry
    one of those classes and chases only those (no year / other quota); default () = the original CLASSES.
    Returns (selected rows with "why", stats)."""
    use = tuple(only) or CLASSES
    assert set(use) <= set(ALL_CLASSES), use
    rng = random.Random(seed)
    rows = [r for r in rows if r["day"] not in used_days and min_s <= len(r["text"]) / 15 <= max_s]
    cls = {r["utt_id"]: cls_of(r["text"]) for r in rows}
    if only:
        rows = [r for r in rows if any(cls[r["utt_id"]][k] for k in use)]
    rare = collections.Counter(r["day"][:4] for r in rows if any(cls[r["utt_id"]][k] for k in use if k != "year"))
    years = {y for y, _ in rare.most_common(n_debate_years)}
    rows = [r for r in rows if r["day"][:4] in years]
    rng.shuffle(rows)
    tmpl = lambda r: re.sub(r"\d+", "#", re.sub(r"[^\w\s]", "", r["text"].lower()))[:60]  # noqa: E731
    t_n, s_n, taken, sel = collections.Counter(), collections.Counter(), set(), []
    have = collections.Counter()
    want = {k: per_class for k in use}
    if not only:
        want["year"] = n_year

    def take(r, why):
        u = r["utt_id"]
        if u in taken or t_n[tmpl(r)] >= per_template or s_n[u.rsplit("_", 1)[0]] >= per_session:
            return False
        taken.add(u)
        t_n[tmpl(r)] += 1
        s_n[u.rsplit("_", 1)[0]] += 1
        sel.append(dict(r, why=why))
        for k in ALL_CLASSES:
            have[k] += bool(cls[u].get(k))
        return True

    avail = {k: sum(bool(cls[r["utt_id"]].get(k)) for r in rows) for k in ALL_CLASSES}
    for k in sorted(use, key=lambda k: avail[k]):
        for r in rows:
            if have[k] >= want[k]:
                break
            if cls[r["utt_id"]].get(k):
                take(r, k)
    st = {"debate_years": sorted(years), "rows_unused_days": len(rows), "clips": len(sel), "have": dict(have),
          "want": want, "avail": avail, "days": len({r["day"] for r in sel}),
          "hours_est": round(sum(len(r["text"]) / 15 for r in sel) / 3600, 2)}
    return sel, st


def held_lines(ids: list) -> str:
    """'# group vox:<day>' + the ids of that day, one block per day (carve_heldout.py format)."""
    by = collections.defaultdict(list)
    for i in ids:
        by[day_of(i)].append(i)
    return "".join(f"# group vox:{d}\n" + "".join(i + "\n" for i in sorted(v)) for d, v in sorted(by.items()) if d)


try:
    import modal

    app = modal.App("parakeet-vox-heldout")
    vol = modal.Volume.from_name("parakeet-spoken-form")
    image = modal.Image.debian_slim(python_version="3.11").add_local_python_source("ml_p3", "vox_heldout")

    @app.function(image=image, volumes={VOL: vol}, cpu=2, memory=16384, timeout=3600)
    def select_fn(lang: str, per_class: int, n_year: int, n_debate_years: int, seed: int, max_s: float = 25.0,
                  only: str = "") -> dict:
        import glob

        from ml_p3 import _gv_classes

        used = set()
        pats = [f"{VOL}/ml/p3v2/{lang}/*.jsonl"] if lang != "en" else [f"{VOL}/d1/en4_ok/*.jsonl", f"{VOL}/ml/p3v2/en/*.jsonl"]
        n_files = 0
        for pat in pats:
            for p in glob.glob(pat):
                n_files += 1
                for l in open(p, encoding="utf-8"):
                    m = re.search(r'"(?:id|src|utt_id)": "(\d{8})-\d{4}', l)
                    if m:
                        used.add(m.group(1))
        held = set()
        hp = f"{VOL}/ml/p3v2/heldout_{lang}.txt" if lang != "en" else f"{VOL}/d1/en4_ok/heldout_days_vox_en.txt"
        try:
            held = {m.group(1) for l in open(hp) if (m := re.match(r"(?:# group vox:)?(\d{8})", l.strip()))}
        except FileNotFoundError:
            pass
        rows = [json.loads(l) for l in open(f"{VOL}/ml/granvox/{lang}/index.jsonl", encoding="utf-8")]
        sel, st = select(rows, used | held, lambda t: classes(t, lang, _gv_classes), per_class, n_year, n_debate_years,
                         seed=seed, max_s=max_s, only=tuple(x for x in only.split(",") if x))
        st.update(lang=lang, used_days=len(used), held_days=len(held), files_scanned=n_files)
        by_year = collections.defaultdict(list)
        for r in sel:
            by_year[r["day"][:4]].append(r["utt_id"])
        return {"stats": st, "ids_by_year": dict(by_year), "days": sorted({r["day"] for r in sel})}

    @app.local_entrypoint()
    def select_cli(lang: str, out: str, per_class: int = 80, n_year: int = 330, n_debate_years: int = 2, seed: int = 1,
                   max_s: float = 25.0, classes: str = ""):
        """--classes phone,int_big pulls only clips carrying those classes (default: every class as before)."""
        res = select_fn.remote(lang, per_class, n_year, n_debate_years, seed, max_s, classes)
        print(json.dumps(res["stats"], ensure_ascii=False))
        json.dump(res["ids_by_year"], open(out, "w"))
        json.dump(res, open(out + ".full.json", "w"))
except ImportError:
    pass


if __name__ == "__main__":
    if sys.argv[1:2] == ["held-lines"]:
        ids = [json.loads(l)["id"] for l in open(sys.argv[2]) if l.strip()]
        sys.stdout.write(held_lines(ids))
