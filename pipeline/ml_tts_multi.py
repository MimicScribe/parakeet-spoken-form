"""Approach B for the 16 languages of ml_multi (2026-10-16): synthetic speech ONLY for number shapes real speech
lacks (prices with cents, clock times, phone numbers, dates with a month name, codes, versions, temperatures). Never
years or plain counts (a TTS reading prior once broke real speech); at most 25-30 % of any class.

1. `fragments`: written number fragments with values drawn here (so the value spread is ours, not the LLM's), in each
   language's own writing conventions (decimal comma, thousands grouping, currency position, phone grouping).
2. `write_rows`: an LLM (Gemini Flash-Lite, public text only) adapts a REAL number-bearing sentence (parliament
   captions) to carry the fragment verbatim, and writes the fully spoken form with the correct case / gender
   endings, plus the spoken words of each number piece.
3. `check_row`: the spoken words of every piece must parse back (ml_multi.parse_numbers) to the written values
   (digit strings for phones / codes), the spoken sentence must carry no digit, and outside the pieces the spoken
   sentence must equal the written one. A row failing any check is not voiced.
4. `synth`: Chirp 3 HD voices (Google Cloud TTS) at rates 0.9 / 1.0 / 1.15 / 1.35; 3 voices per language are kept
   for dev rows only.
5. `verify`: an external transcript of the synthetic clip must give the same values for every piece and match the
   label word for word (folded) above a threshold; the result is stored with the row.

    python ml_tts_multi.py plan  --lang pl --n 60 --out rows_pl.jsonl --sentences sents_pl.txt
    python ml_tts_multi.py synth --rows rows_pl.jsonl --wav-dir wav_pl
"""
import argparse
import base64
import json
import os
import random
import re
import sys

import ml_llm
import ml_multi as MM

# writing conventions: decimal mark, thousands mark, currency (symbol, after-amount?), phone patterns ("#" = digit)
FMT = {
    "bg": (",", " ", [("лв.", True), ("€", True)], ["0888 ### ###", "02 ### ####", "+359 87 ### ####"]),
    "hr": (",", ".", [("€", True), ("kn", True)], ["091 ### ####", "01 #### ###", "+385 98 ### ###"]),
    "cs": (",", " ", [("Kč", True), ("€", True)], ["777 ### ###", "+420 602 ### ###", "2## ### ###"]),
    "da": (",", ".", [("kr.", True), ("€", True)], ["## ## ## ##", "+45 ## ## ## ##"]),
    "et": (",", " ", [("€", True), ("eurot", True)], ["5### ####", "6## ####", "+372 5### ####"]),
    "fi": (",", " ", [("€", True), ("euroa", True)], ["040 ### ####", "050 ### ####", "09 ### ####"]),
    "el": (",", ".", [("€", True), ("ευρώ", True)], ["210 ### ####", "69## ### ###", "+30 69## ######"]),
    "hu": (",", " ", [("Ft", True), ("€", True)], ["06 30 ### ####", "+36 1 ### ####", "06 20 ### ####"]),
    "lv": (",", " ", [("€", True), ("eiro", True)], ["2### ####", "67## ####", "+371 2### ####"]),
    "lt": (",", " ", [("€", True), ("Eur", True)], ["8 6## #####", "+370 6## #####", "8 5 ### ####"]),
    "mt": (".", ",", [("€", False), ("ewro", True)], ["2### ####", "79## ####", "+356 99## ####"]),
    "pl": (",", " ", [("zł", True), ("€", True)], ["5## ### ###", "22 ### ## ##", "+48 6## ### ###"]),
    "ro": (",", ".", [("lei", True), ("€", True)], ["07## ### ###", "021 ### ####", "+40 7## ### ###"]),
    "sk": (",", " ", [("€", True), ("eur", True)], ["09## ### ###", "02 #### ####", "+421 9## ### ###"]),
    "sl": (",", ".", [("€", True), ("evrov", True)], ["041 ### ###", "01 ### ## ##", "+386 31 ### ###"]),
    "sv": (",", " ", [("kr", True), ("kronor", True), ("€", True)], ["08-### ## ##", "070-### ## ##", "031-## ## ##"]),
}
MONTHS_HINT = "write the month as a word"
YEARS_POOL: list = []  # year values the "years" class draws from (pop order); empty = a spread 1900-2060
CLASSES = {"price": 12, "clock": 12, "phone": 10, "date": 8, "code": 6, "version": 6, "temp": 6}
SPEEDS = [(0.9, 1), (1.0, 2), (1.15, 2), (1.35, 1)]
DEV_VOICES = 3
LOCALE = {"bg": "bg-BG", "hr": "hr-HR", "cs": "cs-CZ", "da": "da-DK", "et": "et-EE", "fi": "fi-FI", "el": "el-GR",
          "hu": "hu-HU", "lv": "lv-LV", "lt": "lt-LT", "pl": "pl-PL", "ro": "ro-RO", "sk": "sk-SK", "sl": "sl-SI",
          "sv": "sv-SE"}  # Maltese has no Chirp voice


def _group(n: int, sep: str) -> str:
    s = str(n)
    if n < 10000 and sep in ".":  # 4-digit amounts are usually written ungrouped
        return s
    out = ""
    while len(s) > 3:
        out = sep + s[-3:] + out
        s = s[:-3]
    return s + out


# the source sentence of each class comes from parliament captions of a related kind, so the shape fits its topic
SOURCE_KIND = {"years": "years", "dec": None, "pct": None, "ord": None, "range": None, "big": "cur", "price": "cur", "clock": "clock", "date": "years", "phone": None, "code": None, "version": None, "temp": None}
HINTS = {
    "years": "years the sentence compares or lists (each written in digits)",
    "dec": "a decimal number (a measurement, a rate, an average)",
    "pct": "a percentage",
    "ord": "an ordinal number (a place in a ranking, a floor, an anniversary, a point of an agenda)",
    "range": "a range of two numbers (from ... to ...)",
    "big": "a large amount (people, euros, tonnes, households, jobs, kilometres); put it in the middle or near the end "
           "of the sentence, never first",
    "price": "the price or cost of a concrete thing (a ticket, a product, a fee, a meal)",
    "clock": "a clock time at which something happens (a meeting, a vote, a train); read it as hour and minutes",
    "phone": "a phone number that people are asked to call",
    "date": "the calendar date of an event",
    "code": "a reference code, case number, room number or postcode",
    "version": "the version number of a software, regulation draft or document",
    "temp": "a temperature (weather, climate, a measurement)",
}


def fragment(cls: str, lang: str, rng: random.Random) -> dict:
    """One written fragment of class `cls` and its value check: {"written", "check": "values"|"digits", "hint"}."""
    dec, th, curs, phones = FMT[lang]
    if cls == "years":  # 2-3 years per sentence (cheaper per year span); YEARS_POOL (a planned spread) when set
        k = rng.choice([2, 3, 3])
        pool = YEARS_POOL or [rng.choice([rng.randint(1990, 2030), rng.randint(2031, 2060), rng.randint(1900, 1989)])
                              for _ in range(12)]
        ys = []
        while len(ys) < k:
            y = pool.pop() if YEARS_POOL else rng.choice(pool)
            if y not in ys:
                ys.append(y)
        return {"written": ", ".join(map(str, ys)), "check": "years", "years": ys,
                "hint": f"the years {', '.join(map(str, ys))}, each written in digits in the same sentence"}
    if cls == "dec":
        n = rng.choice([rng.randint(0, 99), rng.randint(100, 999)])
        f = rng.choice([str(rng.randint(1, 9)), f"{rng.randint(1, 99):02d}"])
        return {"written": f"{n}{dec}{f}", "check": "values", "hint": "a decimal number"}
    if cls == "pct":
        v = rng.choice([str(rng.randint(1, 100)), f"{rng.randint(0, 99)}{dec}{rng.randint(1, 9)}"])
        return {"written": f"{v} %" if rng.random() < 0.6 else f"{v}%", "check": "nums", "nums": [v], "hint": "a percentage"}
    if cls == "ord":
        n = rng.choice([rng.randint(2, 31), rng.randint(32, 120)])
        w = {"sv": f"{n}:e", "ro": f"al {n}-lea"}.get(lang, f"{n}.")
        return {"written": w, "check": "values", "hint": "an ordinal number"}
    if cls == "range":
        a = rng.choice([rng.randint(2, 90), rng.randint(100, 900), rng.randint(1000, 90000)])
        b = a + rng.choice([2, 3, 5, 10, 20, 50]) * max(1, a // 10)
        return {"written": f"{_group(a, th)}–{_group(b, th)}", "check": "nums", "nums": [_group(a, th), _group(b, th)],
                "hint": "a range of two numbers"}
    if cls == "big":
        while True:  # thousands to billions; never a 4-digit 1100-2099 (a year to the census)
            a = rng.choice([rng.randint(1000, 9999), rng.randint(10000, 999999), rng.randint(1000000, 999999999),
                            rng.randint(1, 999) * 10 ** 6,
                            rng.randint(1, 99) * 10 ** 9 + rng.choice([0, 0, rng.randint(1, 999) * 10 ** 6])])
            if not 1100 <= a <= 2099:
                break
        return {"written": _group(a, th), "check": "values", "hint": "a large number"}
    if cls == "price":
        a = rng.choice([rng.randint(1, 99), rng.randint(100, 999), rng.randint(1000, 99999)])
        c = rng.choice([rng.randint(1, 99), 50, 99, 5, 90])
        sym, after = rng.choice(curs)
        amt = f"{_group(a, th)}{dec}{c:02d}"
        w = f"{amt} {sym}" if after else f"{sym}{amt}"
        return {"written": w, "check": "values", "hint": "a price with cents"}
    if cls == "clock":
        h, m = rng.randint(0, 23), rng.choice([0, 5, 10, 15, 20, 25, 30, 40, 45, 50, 55, rng.randint(1, 59)])
        sep = rng.choice([":", "."])
        return {"written": f"{h}{sep}{m:02d}", "check": "values", "hint": "a clock time"}
    if cls == "phone":
        p = rng.choice(phones)
        w = "".join(str(rng.randint(0, 9)) if ch == "#" else ch for ch in p)
        return {"written": w, "check": "digits", "hint": "a phone number, read the way people of this language read phone numbers"}
    if cls == "date":
        d, y = rng.randint(1, 28), rng.choice([rng.randint(1950, 2024), rng.randint(2025, 2060)])
        mo = rng.randint(1, 12)
        return {"written": f"{d}.{mo}.{y}", "check": "date", "hint": f"a date: write it as day {d} (digits), the name of "
                f"month {mo} as a word, year {y} (digits), in the usual written order of the language", "day": d, "year": y,
                "month": mo}
    if cls == "code":
        n = rng.randint(3, 6)
        w = "".join(str(rng.randint(0, 9)) for _ in range(n))
        return {"written": w, "check": "digits", "hint": "a code, reference or room number read digit by digit or in groups"}
    if cls == "version":
        parts = [str(rng.randint(1, 20)), str(rng.randint(0, 12))] + ([str(rng.randint(0, 9))] if rng.random() < 0.5 else [])
        return {"written": ".".join(parts), "check": "digits", "hint": "a software or document version number"}
    if cls == "temp":
        t = rng.choice([rng.randint(-25, -1), rng.randint(0, 40)])
        dd = rng.random() < 0.3
        w = (f"{t}{dec}{rng.randint(1, 9)}" if dd else str(t)) + rng.choice([" °C", "°C", " stopni" if lang == "pl" else " °C"])
        return {"written": w.replace("-", "−") if rng.random() < 0.3 else w, "check": "values", "hint": "a temperature"}
    raise KeyError(cls)


SYSTEM = ("You write natural {name} sentences and their exact spoken form. You are a native speaker with perfect "
          "grammar: in the spoken form every number is written out in words with the case, gender and number that the "
          "sentence requires, exactly as a careful native speaker would say it aloud.")
PROMPT = """Here are real {name} sentences from parliament debates, each with a number fragment and what the fragment must be.

For each item:
1. "written": write ONE new, natural, sensible {name} sentence (8-30 words) on the same topic as the given sentence, in which the fragment is used exactly as described in "use as" and makes sense (a price is what something costs, a phone number is called, a clock time is when something happens). The fragment must appear EXACTLY as given (same digits, symbols and spacing). Do not add other numbers.
2. "spoken": the same sentence exactly as spoken aloud, with EVERY number (also the other numbers already in the sentence), symbol and abbreviation written out in words with correct grammatical endings for the context. No digits, no symbols like % € °. All other words identical to "written".
3. "pieces": a list of {{"written": <a substring of "written" holding digits>, "spoken": <the words that replace it in "spoken">}} covering every digit in "written".

Items:
{items}

Answer as a JSON list of objects with keys "id", "written", "spoken", "pieces"."""


def _shown(it: dict) -> str:
    """The fragment as the prompt shows it: a "years" item lists its years apart, so they are not written as one
    comma list (a list of years reads as one long number to the span finder)."""
    if it.get("check") == "years":
        return " / ".join(map(str, it["years"])) + " (each year on its own, in different places of the sentence, " \
            "joined by words, never as a comma list)"
    return it["written"]


def write_rows(lang: str, sentences: list, n: int, seed: int = 1, batch: int = 10) -> list:
    """Plan n rows: classes round-robin over CLASSES weights, a real sentence and a fragment each, written by the LLM."""
    rng = random.Random(f"{seed}-{lang}")
    sentences = [x for x in sentences if in_language(x, lang)] or sentences
    by_kind = {k: [x for x in sentences if (MM.classes(x, lang)[k] if k else True)] or sentences
               for k in set(SOURCE_KIND.values())}
    plan = [c for c, k in CLASSES.items() for _ in range(k)]
    rng.shuffle(plan)
    plan = (plan * (n // len(plan) + 1))[:n]
    items = []
    for i, cls in enumerate(plan):
        fr = fragment(cls, lang, rng)
        fr["hint"] = HINTS[cls] + (f" ({fr['hint']})" if cls in ("date", "years") else "")
        items.append({"id": f"{lang}_{seed}_{i:04d}", "cls": cls, "sentence": rng.choice(by_kind[SOURCE_KIND[cls]]), **fr})
    rows = []
    name = __import__("ml_multi_lexgen").NAMES[lang]
    for b in range(0, len(items), batch):
        chunk = items[b:b + batch]
        txt = "\n".join(json.dumps({"id": it["id"], "sentence": it["sentence"], "fragment": _shown(it), "use as": it["hint"]},
                                   ensure_ascii=False) for it in chunk)
        try:
            ans = json.loads(ml_llm.call(PROMPT.format(name=name, items=txt).replace("{hint}", "the 'use as' note"),
                                         system=SYSTEM.format(name=name), max_tokens=8000))
        except Exception as e:  # noqa: BLE001
            print("batch failed", repr(e)[:200], file=sys.stderr)
            continue
        by = {a.get("id"): a for a in ans if isinstance(a, dict)}
        for it in chunk:
            a = by.get(it["id"])
            if a:
                rows.append({**it, "text_written": a.get("written", ""), "text": a.get("spoken", "")})
    return rows


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def piece_ok(written: str, spoken: str, check: str, lang: str) -> tuple:
    """(ok, reason) for one number piece."""
    if re.search(r"\d", spoken):
        return False, "digit_in_spoken"
    nums = MM.parse_numbers(spoken, lang)
    if not nums:
        return False, "unparsed"
    if check == "digits":
        ds = "".join(str(int(n.value)) if n.value == int(n.value) else str(n.value).replace(".", "") for n in nums)
        return (ds == _digits(written)), f"digits {ds} vs {_digits(written)}"
    ok, _ = MM.values_match(written.split(), spoken, lang)
    if ok:
        return True, ""
    ws = [str(x) for a in MM.written_values(written.split())[:3] for x in a]
    return False, f"values {[str(n.value) for n in nums]} vs {ws}"


def _norm_frag(x: str) -> str:
    return re.sub(r"\s+", "", x.replace("−", "-").replace("–", "-"))


def check_row(r: dict, lang: str) -> tuple:
    """(ok, reasons). The written sentence is aligned word by word to the spoken one exactly as a real clip is
    aligned to its external transcript (ml_multi.label_row): every digit stretch must be read by spoken words that
    parse back to its value(s), nothing may be masked, the spoken sentence has no digit, and the class fragment is in
    the written sentence (dates: the day and the year, the month being a word)."""
    w, s = r.get("text_written", ""), r.get("text", "")
    why = []
    if r["check"] == "date":
        nums = re.findall(r"\d+", w)
        if str(r["day"]) not in nums or str(r["year"]) not in nums:
            why.append("fragment_missing")
    elif r["check"] == "nums":  # the numbers in the written sentence, however it joins them ("od 62 do 93", "84 procent")
        have = {_digits(m) for m in re.findall(r"\d[\d  .,]*\d|\d", w)}
        if not all(_digits(str(n)) in have for n in r["nums"]):
            why.append("fragment_missing")
    elif r["check"] == "years":
        nums = re.findall(r"\d+", w)
        if not all(str(y) in nums for y in r["years"]):
            why.append("fragment_missing")
    elif _norm_frag(r["written"]) not in _norm_frag(w):
        why.append("fragment_missing")
    if not s or re.search(r"\d", s):
        why.append("digit_in_spoken")
    if not why and lang == "ro":
        import ro_gender

        # a synthetic reading must be the standard form ("două mii", not the colloquial "doi mii"): the LLM's
        # masculine slips are repaired before labelling; whatever grammar_errors still finds below rejects the row
        s = ro_gender.repair(s)
    if not why:
        lab = MM.label_row(w, [{"w": t, "s": i, "e": i} for i, t in enumerate(s.split())], lang, max_extra=1)
        if lab["masked"]:
            why += [f"masked:{m['cap']}->{m['ext']} {m['values_ext']} vs {m['written'][:2]}" for m in lab["masked"]]
        else:
            # the label (and the text voiced) is the WRITTEN sentence with each digit stretch replaced by its checked
            # spoken words: casing, punctuation and every other word come from the written sentence, never from the
            # LLM's free rewrite
            r["llm_spoken"], r["text"] = s, lab["text"]
            r["rest_agreement"] = round(MM.word_agreement(lab["text"], [{"w": t} for t in s.split()]), 3)
        r["spans"] = lab["spans"]
        for sp in lab["spans"]:
            if not cents_ok(sp["cap"], sp["words"], lang):
                why.append(f"cents_bare:{sp['cap']}->{sp['words']}")
        why += [f"grammar:{c}" for c in grammar_errors(r, lang)]
    return (not why), why


CLS_KIND = {"years": "int", "dec": "dec", "pct": "pct", "ord": "ord", "range": "range", "big": "int", "price": "cur", "clock": "clock", "phone": "phone", "date": "date", "code": "code", "version": "version",
            "temp": "unit"}


def grammar_errors(r: dict, lang: str) -> list:
    """The deterministic grammar checks (grammar_rules: ro "de" linker, "o mie", number gender incl. ro_gender; pl
    clock minutes, genitive after a governing word, "dwie tysiące") on a checked row's spans: the value check alone
    lets "patruzeci și nouă lei" or "osiemnastej pięćdziesiątej piątej" through."""
    import grammar_rules as G

    out = []
    for sp in r.get("spans") or []:
        g = {"text": sp.get("cap", sp.get("text", "")), "reading": sp.get("words", sp.get("reading", "")),
             "kind": sp.get("kind") or CLS_KIND.get(r.get("cls"), r.get("cls"))}
        ctx = {"cls": r.get("cls"), "text": r.get("text", ""), "text_written": r.get("text_written", "")}
        out += G.span_errors(lang, g, ctx)
    return sorted(set(out))


def cents_ok(written: str, spoken: str, lang: str) -> bool:
    """A price with cents ("912,05 €") read as two bare numbers ("nouă sute doisprezece, cinci euro") passes the
    value check (912 + 05) but is heard as 912.5: the cents must follow a decimal word or a currency / cents word."""
    if not re.search(r"\d[.,]\d{2}(?!\d)", written) or not re.search(MM.CUR, written.lower()):
        return True
    nums = MM.parse_numbers(spoken, lang)
    if len(nums) < 2:
        return True
    toks = spoken.split()
    between = [MM._key(t) for t in toks[nums[0].end + 1:nums[1].start] if MM._key(t)]
    return bool(between)


SCRIPT = {"bg": "а-я", "el": "α-ωά-ώ", "pl": "ąęłśżźćńó", "ro": "ăâîșțşţ", "sv": "åäö", "hu": "áéíóöőúüű", "fi": "äö",
          "cs": "áčďéěíňóřšťúůýž", "sk": "áäčďéíĺľňóôŕšťúýž", "sl": "čšž", "hr": "čćđšž", "lv": "āčēģīķļņšūž",
          "lt": "ąčęėįšųūž", "et": "äöõüšž", "da": "æøå", "mt": "ċġħż"}


def in_language(sent: str, lang: str) -> bool:
    """Crude script check for source sentences (the parliament pseudo-labels sometimes carry English): at least two
    letters only this language's alphabet has."""
    return len(re.findall(f"[{SCRIPT[lang]}]", sent.lower())) >= 2


# ------------------------------------------------------------------ voices
def synth_rows(rows: list, lang: str, wav_dir: str, seed: int = 1, threads: int = 8, cycle: bool = False,
               voice: str = "") -> list:
    """Voice rows with the language's cloud voices (3 held out for dev rows); cycle: every voice in turn (round robin
    per split) instead of a random one per row; voice: every row in that one voice, split dev (the language's
    held-out decision-set voice, mix_build.HELDOUT_VOICES)."""
    import ml_tts_cloud as C
    from concurrent.futures import ThreadPoolExecutor

    auth = C.Auth()
    d = C._req(auth, f"{C.API}/voices?languageCode={LOCALE[lang]}")
    vs = sorted(v["name"] for v in d.get("voices", []) if "Chirp3-HD" in v["name"])
    rng = random.Random(f"{seed}-{lang}-voices")
    rng.shuffle(vs)
    dev_v, train_v = vs[:DEV_VOICES], vs[DEV_VOICES:]
    os.makedirs(wav_dir, exist_ok=True)
    speeds, wts = zip(*SPEEDS)
    turn = {"dev": 0, "train": 0}
    for i, r in enumerate(rows):
        r["split"] = "dev" if i % 10 == 9 else "train"
        if voice:
            assert voice in vs, f"{voice} is not a voice of {lang}"
            r["split"] = "dev"
        pool = dev_v if r["split"] == "dev" else train_v
        v = voice or (pool[turn[r["split"]] % len(pool)] if cycle else rng.choice(pool))
        turn[r["split"]] += 1
        r.update(voice=v, speed=rng.choices(speeds, wts)[0],
                 locale=LOCALE[lang], engine="presynth", origin="tts", wav=os.path.join(wav_dir, r["id"] + ".wav"))

    def one(r):
        if os.path.exists(r["wav"]) and os.path.getsize(r["wav"]) > 1000:
            return True
        body = {"input": {"text": r["text"]}, "voice": {"languageCode": r["locale"], "name": r["voice"]},
                "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 16000, "speakingRate": r["speed"]}}
        for attempt in range(4):
            try:
                audio = base64.b64decode(C._req(auth, f"{C.API}/text:synthesize", body)["audioContent"])
                with open(r["wav"], "wb") as f:  # written only after the request succeeded (a failed one left 0 bytes)
                    f.write(audio)
                return True
            except Exception:  # noqa: BLE001
                import time
                time.sleep(2 ** attempt)
        return False

    with ThreadPoolExecutor(threads) as ex:
        ok = list(ex.map(one, rows))
    return [r for r, k in zip(rows, ok) if k]


def verify_row(r: dict, ext_words: list, lang: str) -> dict:
    """External-transcript check of one synthetic clip: every piece's values must appear, in order, in the external
    transcript's numbers; word agreement of the label vs the external transcript."""
    ext = " ".join(w["w"] for w in ext_words)
    got = [n.value for n in MM.parse_numbers(ext, lang)]
    want = [n.value for sp in r.get("spans", []) for n in MM.parse_numbers(sp["words"], lang)]
    it = iter(got)
    in_order = all(any(g == v for g in it) for v in want)
    return {"values_ok": in_order, "want": [str(v) for v in want], "got": [str(v) for v in got],
            "word_agreement": round(MM.word_agreement(r["text"], ext_words), 3)}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("plan")
    p.add_argument("--lang")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--sentences")
    p.add_argument("--out")
    p.add_argument("--seed", type=int, default=1)
    s = sub.add_parser("synth")
    s.add_argument("--rows")
    s.add_argument("--wav-dir")
    a = ap.parse_args()
    if a.cmd == "plan":
        sents = [l.strip() for l in open(a.sentences) if l.strip()]
        rows = write_rows(a.lang, sents, a.n, a.seed)
        with open(a.out, "w") as f:
            for r in rows:
                ok, why = check_row(r, a.lang)
                f.write(json.dumps({**r, "check_ok": ok, "check_why": why}, ensure_ascii=False) + "\n")
    elif a.cmd == "synth":
        rows = [json.loads(l) for l in open(a.rows)]
        lang = rows[0]["id"].split("_")[0]
        good = synth_rows([r for r in rows if r.get("check_ok")], lang, a.wav_dir)
        with open(a.rows.replace(".jsonl", "_synth.jsonl"), "w") as f:
            for r in good:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(lang, "voiced", len(good), "chars", sum(len(r["text"]) for r in good))


if __name__ == "__main__":
    main()
