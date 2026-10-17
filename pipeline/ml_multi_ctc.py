"""Approach A for the 16 languages of ml_multi, CTC form (2026-10-16): the external transcript of these languages
writes numbers as DIGITS, so it can confirm a value but cannot give the spoken case ending. The ending comes from the
audio instead:

  1. candidates  an LLM (public caption text) proposes every plausible spoken reading of each number expression of the
                 caption (case / gender / number variants); every reading is then re-inflected from the
                 language's paradigm (ml_multi_cases: each number in every case / gender / ending, the unit word in
                 each form, a range with and without its connector), so the spoken form is offered even when the
                 LLM missed it (`expand_cands`); each must parse back to the written value (ml_multi.values_match)
                 and may hold only number words, the number's own unit / currency / percent word and connectors
                 (`word_allowed`; "ett punk ett" is out), and no written word may be lost ("COVID-19" read
                 "dziewiętnaście") (`reading_ok`, local);
  2. decision    a public wav2vec2 CTC reader of the language scores the whole clip with each candidate in place (the
                 same reader class and rule as ml_decide: best beats every rival with other words by MARGIN nats, and
                 its words fit the greedy transcript, accents folded) (`score_lang`, Modal GPU);
  3. value check the external transcript's digits at the span's place give the caption's value (`ext_value_ok`,
                 `digits_agree`: grouping, month names and the transcript's additive groups "720 1" = 721).

The scale-up driver (selection, resumable cut, windowed scoring, number-piece transcripts, assembly) is
ml_multi_scale.py.

A span that fails 2 or 3 is masked (the clip is cut around it at the external transcript's word times), never
guessed.

    modal run ml_multi_ctc.py --lang pl --items ml/multi/pl/pilot/ctc_items.jsonl
"""
import json
import os
import re

try:
    import modal
except ImportError:  # the local helpers (candidates, ext_value_ok) need no Modal
    modal = None

VOL = "/vol"
READERS = {  # public CTC fine-tunes (Apache-2.0 / CC0), 2026-10-16 inventory
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish", "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish", "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "ro": "gigant/romanian-wav2vec2", "sv": "KBLab/wav2vec2-large-voxrex-swedish",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250", "sk": "comodoro/wav2vec2-xls-r-300m-sk-cv8",
    "sl": "anton-l/wav2vec2-large-xlsr-53-slovenian", "et": "anton-l/wav2vec2-large-xlsr-53-estonian",
    "lt": "m3hrdadfi/wav2vec2-large-xlsr-lithuanian", "lv": "jimregan/wav2vec2-large-xlsr-latvian-cv",
    "bg": "DrishtiSharma/wav2vec2-large-xls-r-300m-bg-d2", "da": "NbAiLab/wav2vec2-large-danish-npsc-nst",
    "hr": "classla/wav2vec2-xls-r-parlaspeech-hr", "mt": "Akashpb13/xlsr_maltese_wav2vec2",
}
MARGIN = 3.0


CAND_SYS = ("You are a native {name} speaker and a careful grammarian. For numbers in transcripts you list how they can "
            "be spoken aloud, with the grammatical case, gender and number that the context allows.")
CAND_PROMPT = """For each {name} sentence, find every number expression (digits together with any symbol, unit or abbreviation attached to them and read with them, e.g. "5 mln zł", "3,5%", "2025 r.", "godz. 15.30"). For each give:
- "written": the exact substring of the sentence,
- "readings": 4 to 6 DIFFERENT spoken forms of exactly that substring (every number, symbol and abbreviation in words, nothing from outside the substring): the most likely form in this sentence first, then the same number in other grammatical cases / genders / numbers (e.g. nominative, genitive, locative, instrumental; cardinal and ordinal where both are possible) and other usual ways of reading it.
Items:
{items}
Answer as a JSON list of {{"id": ..., "numbers": [{{"written": ..., "readings": [...]}}]}}."""


def reading_ok(written: str, reading: str, lang: str, after: str = "", before: str = "") -> bool:
    """A reading of `written` must parse back to its value(s) and add no word from outside it: its non-number words
    are at most two per symbol / abbreviation / separator of the written text, and it may not end with the caption's
    next word ("tuhat euroa" for "1000" followed by "euroa")."""
    import ml_multi as MM

    ok, nums = MM.values_match(written.split(), reading, lang)
    if not ok:
        return False
    toks = reading.split()
    num_tok = {k for n in nums for k in range(n.start, n.end + 1)}
    extra = [t for k, t in enumerate(toks) if k not in num_tok and MM._key(t) not in MM.lexicon(lang).glue]
    sym = (len(re.findall(r"[^\d\s.,:'\-–]+", written)) + len(re.findall(r"\d[.,:]\d", written))
           + len(re.findall(r"\d\s?[-–]\s?\d", written)))  # a range dash is read as a word ("pięć do dziesięciu")
    nxt = after.split()[0] if after.split() else ""
    if nxt and toks and MM._key(toks[-1]) == MM._key(nxt):
        return False
    prv = re.split(r"[\s-]", before.strip())[-1] if before.strip() else ""
    if prv and toks and MM._key(toks[0]) == MM._key(prv) and not MM.parse_numbers(toks[0], lang):
        return False  # "cea de-a [14-a]" read "a paisprezecea": the article is already in the caption
    if any(not word_allowed(MM._key(t), written, lang) for k, t in enumerate(toks) if k not in num_tok and MM._key(t)):
        return False  # "ett punk ett" for "1.1": a word that is neither a number, its unit nor a connector
    if "%" in written and nums:  # "0,1%" read "zero procent i jeden": the percent word comes after the number
        pct = re.compile(MM.fold(MM.PCT.get(lang, "procent\\w*")).replace(r"\s+", "|"))
        pk = [i for i, t in enumerate(toks) if pct.fullmatch(MM._key(t)) or (lang == "ro" and MM._key(t) == "suta")]
        if pk and min(pk) < max(n.end for n in nums):
            return False
    if not written_words_kept(written, reading, lang):
        return False  # "COVID-19" read "dziewiętnaste": the name would be lost from the label
    return len(extra) <= 2 * sym


# Words a reading may hold besides number words (folded keys, ml_multi._key): connectors that stand inside numbers,
# and per symbol / abbreviation of the WRITTEN span the unit, currency or percent word it is read as. A word outside
# these makes the reading wrong ("ett punk ett"), whatever its value.
CONNECT = {"pl": r"i|oraz", "ro": r"si|de|dea|al|a|cel|cea|celui|celei", "sv": r"och"}
RANGE_WORDS = {"pl": r"do|od", "ro": r"la|pana|intre|si", "sv": r"till|fran"}
SEP_WORDS = {"pl": r"przecinek|koma|kropka|dwukropek", "ro": r"virgula|punct|si|intregi|intreg", "sv": r"komma|punkt"}
UNIT_WORDS = {  # (written pattern, allowed folded words)
    "pl": [(r"%|proc\.?|procent", r"procent\w*"), (r"zł|zl\b|złot", r"zlot\w*|grosz\w*"), (r"\bgr\b", r"grosz\w*"),
           (r"€|eur", r"euro|eurocent\w*|cent\w*"), (r"\$|usd|dolar", r"dolar\w*|cent\w*"), (r"\br\.|\brok", r"rok\w*|roku"),
           (r"godz|godzin", r"godzin\w*"), (r"\bmin\b|minut", r"minut\w*"), (r"\bnr\b|numer", r"numer\w*"),
           (r"\bart\.", r"artykul\w*"), (r"\bust\.", r"ustep\w*"), (r"\bpkt\b", r"punkt\w*"), (r"\bkm\b", r"kilometr\w*"),
           (r"\bkg\b", r"kilogram\w*"), (r"°|stop", r"stop\w*|celsjusz\w*"), (r"‰", r"promil\w*"), (r"\bha\b", r"hektar\w*"),
           (r"m²|m2\b", r"metr\w*|kwadrat\w*"), (r"/", r"na|lamane|przez")],
    "ro": [(r"%|la sută|procent", r"la|suta|procent\w*"), (r"\blei\b|\bleu\b|\bron\b", r"lei|leu|ban\w*"),
           (r"€|eur", r"euro|eurocent\w*|cent\w*"), (r"\$|usd|dolar", r"dolar\w*|cent\w*"), (r"\bora\b|\borele\b|\bh\b", r"or[aei]\w*"),
           (r"\bnr\.?", r"numar\w*|numarul"), (r"\bart\.", r"articol\w*"), (r"\bkm\b", r"kilometr\w*"),
           (r"\bkg\b", r"kilogram\w*"), (r"°|grad", r"grad\w*|celsius"), (r"/", r"pe|supra")],
    "sv": [(r"%|procent", r"procent"), (r"\bkr\b|kron", r"kron\w*|ore|oren"), (r"€|euro", r"euro|cent"),
           (r"\bkl\.?|klockan", r"klockan")],
}


def written_words_kept(written: str, reading: str, lang: str) -> bool:
    """Every word of the written span that is not a number abbreviation / unit / symbol word ("COVID" of "COVID-19",
    "Horyzont" of "Horyzont 2020") must still be in the reading (any ending)."""
    import ml_multi as MM

    rk = [MM._key(t) for t in reading.split()]
    abbr = {MM._key(a) for a in MM.SCALE_ABBR}
    suffixes = {m.group(1) for m in re.finditer(r"\d[-–]?([^\W\d_]{1,4})\b", written)}  # "5-tego", "2-lea", "14-a"
    for chunk in re.findall(r"[^\W\d_]+", written):
        k = MM._key(chunk)
        if chunk in suffixes and chunk.islower():
            continue
        if not k or k in abbr or re.fullmatch(MM.fold(MM.SCALE), k) or MM.lexicon(lang).morphs(k):
            continue
        if any(re.search(pat, chunk.lower()) or re.search(pat, written.lower()) and len(k) <= 4
               for pat, _w in UNIT_WORDS.get(lang, [])):
            continue  # "zł", "r.", "godz.", "nr": read as their unit word
        if len(k) <= 2 and not k.isupper():
            continue  # a case suffix written after the digits ("5-tego", "2-lea", "14-a")
        if k in MM.month_forms(lang) and any(MM._key(t) in MM.month_forms(lang) for t in reading.split()):
            continue
        stem = k[:max(3, len(k) - 3)] if len(k) >= 4 else k
        if not any(t == k or t.startswith(stem) for t in rk):
            return False
    return True


def word_allowed(k: str, written: str, lang: str) -> bool:
    """May the folded word `k` stand in a reading of `written` (outside its number words)?"""
    import ml_multi as MM

    lx = MM.lexicon(lang)
    if k in lx.glue or k in lx.dec or k in lx.half or k in lx.minus or k in lx.frac:
        return True
    pats = [CONNECT.get(lang, "")]
    w = written.lower()
    if re.search(r"\d\s?[-–]\s?\d", written):
        pats.append(RANGE_WORDS.get(lang, ""))
    if re.search(r"\d[.,:]\d", written):
        pats.append(SEP_WORDS.get(lang, ""))
    if "%" in written:
        pats.append(MM.fold(MM.PCT.get(lang, "procent\\w*")).replace(r"\s+", "|"))
    for pat, words in UNIT_WORDS.get(lang, []):
        if re.search(pat, w):
            pats.append(words)
    if any(p and re.fullmatch(f"(?:{p})", k) for p in pats):
        return True
    for t in re.findall(r"[^\W\d_]+", written):  # a word of the written span itself, in any ending ("euro", "miliardów")
        tk = MM._key(t)
        if len(tk) >= 2 and not re.search(r"\d", tk) and (k == tk or (len(tk) >= 4 and k.startswith(tk[:max(3, len(tk) - 3)]))):
            return True
    if re.search(r"(?<!\d)\d{1,2}[./]\d{1,2}[./](?:\d{2}|\d{4})(?!\d)", written) and k in MM.month_forms(lang):
        return True  # a numeric date is read with its month name
    return False


MAX_CANDS = 80


def ro_de_variants(reading: str) -> list:
    """Romanian: the reading with "de" put before / taken from the unit word after its last number."""
    import ml_multi as MM

    toks = reading.split()
    nums = MM.parse_numbers(reading, "ro")
    if not nums or nums[-1].end + 1 >= len(toks):
        return []
    k = nums[-1].end + 1
    if MM._key(toks[k]) == "de":
        return [" ".join(toks[:k] + toks[k + 1:])] if k + 1 < len(toks) else []
    return [" ".join(toks[:k] + ["de"] + toks[k:])]


RANGE_CONN = {"pl": ["do"], "ro": ["la", "până la"], "sv": ["till"]}


def range_variants(reading: str, lang: str) -> list:
    """A range reading with its connector between the two numbers removed, or each usual connector put there
    ("dwadzieścia do dwudziestu pięciu" / "dwadzieścia dwadzieścia pięć"): the speaker's choice is left to the CTC."""
    import ml_multi as MM

    toks = reading.split()
    nums = MM.parse_numbers(reading, lang)
    if len(nums) != 2:
        return []
    a, b = nums[0].end + 1, nums[1].start
    head, tail = toks[:a], toks[b:]
    out = [" ".join(head + tail)] + [" ".join(head + c.split() + tail) for c in RANGE_CONN.get(lang, [])]
    return [x for x in out if x != reading]


def expand_cands(written: str, readings: list, lang: str, after: str = "", before: str = "") -> list:
    """The value-checked candidate set of one span: the LLM's readings (its order first), then every lexicon-driven
    inflection of each (ml_multi_cases.expand: case / gender / ending of each number and of the unit word), and for a
    plain integer every case form and ordinal ending of its value. Each must pass reading_ok."""
    import ml_multi_cases as MC

    out, seen = [], set()

    def add(rd):
        rd = rd.strip()
        if rd and rd not in seen and len(out) < MAX_CANDS and not re.search(r"\d", rd):
            seen.add(rd)
            if reading_ok(written, rd, lang, after, before):
                out.append(rd)

    base = [str(x).strip() for x in readings if str(x).strip()]
    if lang == "ro":  # the spoken decimal word is "virgulă" ("virgula" = "the comma"; the reader cannot tell them apart)
        base = [re.sub(r"\bvirgula\b", "virgulă", x) for x in base]
    for rd in base:
        add(rd)
    if lang == "ro":  # "două miliarde (de) euro": the speaker's "de" before the unit word is left to the CTC
        for rd in list(out):
            for v in ro_de_variants(rd):
                add(v)
    if re.search(r"\d\s?[-–]\s?\d", written):  # a range dash may be read "do" / "la" or not at all
        for rd in list(out):
            for v in range_variants(rd, lang):
                add(v)
    for rd in list(out):
        for v in MC.expand(rd, lang)[1:]:
            add(v)
    clock = re.search(r"(?<![\d.,])([01]?\d|2[0-3])[.:]([0-5]\d)(?![\d.,]\d)", written)
    if clock and lang == "pl":  # Polish says the hour as an ORDINAL ("o jedenastej trzydzieści"): every ending,
        import ml_multi_cases as MC  # minutes as a cardinal (or nothing for ":00")

        h, m = int(clock.group(1)), int(clock.group(2))
        for e in MC.PL_ENDINGS:
            hw = MC.pl_ordinal(h, e) if h else None
            if hw:
                add(f"{hw} {MC.pl_cardinal(m, 'nom')}" if m else hw)
                if not m:
                    add(f"{hw} zero zero")
    plain = re.fullmatch(r"\d{1,3}(?:[  ]\d{3})+|\d+", written.strip())
    if plain and lang in ("pl", "ro"):
        n = int(re.sub(r"\D", "", written))
        for _k, v in MC._variants(n, False, lang) + MC._variants(n, True, lang):
            add(v)
    return out


def candidates(rows: list, lang: str, batch: int = 8) -> dict:
    """{row id: [{"written", "a", "b", "cands": [value-checked readings]}]} (local, LLM)."""
    import ml_llm
    import ml_multi as MM
    from ml_multi_lexgen import NAMES

    out = {}
    for b in range(0, len(rows), batch):
        ch = rows[b:b + batch]
        items = "\n".join(json.dumps({"id": r["id"], "sentence": r["text"]}, ensure_ascii=False) for r in ch)
        try:
            ans = json.loads(ml_llm.call(CAND_PROMPT.format(name=NAMES[lang], items=items),
                                         system=CAND_SYS.format(name=NAMES[lang]), max_tokens=12000, temperature=0.3))
        except Exception as e:  # noqa: BLE001
            print("candidate batch failed", repr(e)[:120])
            continue
        by = {a.get("id"): a for a in ans if isinstance(a, dict)}
        for r in ch:
            spans, used = [], 0
            for nb in (by.get(r["id"]) or {}).get("numbers") or []:
                w = str(nb.get("written", ""))
                a = r["text"].find(w, used) if w else -1
                if a < 0 or not re.search(r"\d", w):
                    continue
                cands = expand_cands(w, [str(x) for x in nb.get("readings") or []], lang, r["text"][a + len(w):],
                                     r["text"][:a])
                spans.append({"written": w, "a": a, "b": a + len(w), "cands": cands,
                              "offered": len(nb.get("readings") or [])})
                used = a + len(w)
            out[r["id"]] = spans
    return out


def render(text: str, spans: list, pick: list) -> str:
    out, pos = [], 0
    for s, k in zip(spans, pick):
        out.append(text[pos:s["a"]])
        out.append(s["cands"][k] if s["cands"] else s["written"])
        pos = s["b"]
    out.append(text[pos:])
    return "".join(out)


if modal is not None:
    vol = modal.Volume.from_name("parakeet-spoken-form")
    image = (modal.Image.debian_slim(python_version="3.11")
             .apt_install("ffmpeg", "libsndfile1")
             .pip_install("torch==2.4.1", "torchaudio==2.4.1", "transformers==4.44.2", "soundfile", "librosa", "numpy<2")
             .add_local_python_source("ml_cands", "ml_decide", "ml_multi")
             .add_local_file("ml_grammar_table.json", "/root/ml_grammar_table.json")
             .add_local_file("ml_multi_lex.json", "/root/ml_multi_lex.json")
             .add_local_file("ml_multi_months.json", "/root/ml_multi_months.json"))
    app = modal.App("parakeet-ml-multi-ctc")
    # CPU: wav2vec2-large scores a 25 s clip in a few seconds on 8 cores, and CPU is never queued behind other GPU jobs
    gpu_fn, entry = app.function(image=image, volumes={VOL: vol}, cpu=8, memory=16384, timeout=3 * 3600,
                                 secrets=[modal.Secret.from_name("hf")]), app.local_entrypoint()
else:
    gpu_fn = entry = (lambda f: f)


@gpu_fn
def score_lang(lang: str, items_path: str) -> dict:
    """Per row: CTC NLL of every candidate of every span (other spans at their current best, two passes), the
    decision (margin over rivals with different words, greedy fit) and the greedy transcript."""
    import soundfile as sf

    import ml_decide as D

    D.MODELS[lang] = READERS[lang]
    import torch

    class CaseCTC(D.LangCTC):
        """Some readers (KBLab Swedish, the Hungarian / Finnish XLSR fine-tunes) have an UPPERCASE alphabet; the
        base class looks letters up lowercased, which scores every reading as impossible. Look up in the
        alphabet's own case."""

        def ids(self, text):
            letters = [k for k in self.vocab if len(k) == 1 and k.isalpha()]
            if letters and sum(k.isupper() for k in letters) > len(letters) / 2:
                out = []
                for w in re.sub(r"[^\w' ]+", " ", text.upper()).split():
                    cs = [self.vocab[c] for c in w if c in self.vocab]
                    if cs:
                        if out and self.delim is not None:
                            out.append(self.delim)
                        out += cs
                return out
            return super().ids(text)

    ctc = CaseCTC(lang, device="cuda" if torch.cuda.is_available() else "cpu")
    rows = [json.loads(l) for l in open(f"{VOL}/{items_path}")]
    out = []
    for r in rows:
        x, sr = sf.read(r["audio_filepath"], dtype="float32")
        lp = ctc.lp(x)
        greedy = ctc.greedy(lp)
        sp = [s for s in r["spans"] if s["cands"]]
        pick = [0] * len(sp)
        res = []
        for _ in range(2):
            res = []
            for j, s in enumerate(sp):
                scores = []
                for ci in range(len(s["cands"])):
                    p = list(pick)
                    p[j] = ci
                    scores.append(ctc.nll(lp, render(r["text"], sp, p)))
                best = min(range(len(scores)), key=lambda i: scores[i])
                bk = D.key(s["cands"][best])
                rivals = [i for i in range(len(scores)) if D.key(s["cands"][i]) != bk]
                margin = min((scores[i] - scores[best] for i in rivals), default=float("inf"))
                res.append({"written": s["written"], "scores": [round(v, 2) for v in scores], "best": best,
                            "margin": round(margin, 2) if margin != float("inf") else None,
                            "fit": D.fits(s["cands"][best], greedy, fuzzy=True)})
            pick = [d["best"] for d in res]
        out.append({"id": r["id"], "greedy": greedy, "spans": res})
    return {"lang": lang, "rows": out}


@entry
def main(lang: str, items: str, out: str = ""):
    res = score_lang.remote(lang, items)
    p = out or f"ctc_{lang}.json"
    json.dump(res, open(p, "w"), ensure_ascii=False)
    print(lang, "rows", len(res["rows"]), "->", p)


def ext_value_ok(caption: str, span: dict, ext_words: list, lang: str) -> bool | None:
    """The external transcript (digits) at the span's place holds the span's value: the caption is aligned to it
    with the span's text as one block; None when the place cannot be found."""
    import difflib

    import ml_multi as MM

    pre = caption[:span["a"]].split()
    post = caption[span["b"]:].split()
    ck = [MM._key(t) for t in pre] + ["\x00SPAN"] + [MM._key(t) for t in post]
    ek = [MM._key(w["w"]) for w in ext_words]
    sm = difflib.SequenceMatcher(None, ck, ek, autojunk=False)
    si = len(pre)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if i1 <= si < i2 and tag != "equal":
            got = " ".join(w["w"] for w in ext_words[j1:j2])
            if not re.search(r"\d", got):
                return MM.values_match(span["written"].split(), got, lang)[0] if got else False
            return digits_agree(span["written"], got, lang)
    return None


def _additive(vals: list) -> list:
    """Join additive digit groups the external transcript writes for one spoken number ("720 1" for "șapte sute
    douăzeci și unu", "20 1" for 21): a group whose trailing zeros cover the next, shorter group absorbs it."""
    out = []
    for v in vals:
        prev = str(int(out[-1])) if out and out[-1] == int(out[-1]) else ""
        tz = len(prev) - len(prev.rstrip("0")) if prev and prev != "0" else 0
        if out and v == int(v) and 0 < v < 10 ** tz:
            out[-1] = out[-1] + v
        else:
            out.append(v)
    return out


def digits_agree(written: str, got: str, lang: str) -> bool:
    """The external transcript's words at a span's place (digits) give the written span's value(s): grouping,
    decimal comma and scale words as in ml_multi.written_values, month names as month numbers ("2 października 1998"
    for "2.10.1998"), additive groups joined ("720 1" = 721), and for digit strings (phones, codes) the same digits
    in any grouping."""
    import ml_multi as MM

    mf = MM.month_forms(lang)
    toks = [str(mf[MM._key(t)]) if MM._key(t) in mf else t for t in got.split()]
    if any(MM._key(t) in mf for t in got.split()):
        toks = [t for t in toks]  # month number joins the date groups below
        date_like = " ".join(toks)
        toks = [re.sub(r"[^\d]", "", t) for t in date_like.split() if re.search(r"\d", t)]
        toks = [".".join(toks)] if len(toks) == 3 else toks
    wv = MM.written_values(written.split(), lang)
    gv = MM.written_values(toks, lang)
    gv = gv + [_additive(a) for a in gv]
    if any(a == b for a in wv for b in gv):
        return True
    wd = re.sub(r"\D", "", written)
    gd = "".join(re.sub(r"\D", "", t) for t in toks)
    groups = re.findall(r"\d+", written)
    if len(groups) >= 3 or re.match(r"\D*0\d", written) or (len(wd) >= 5 and not re.search(r"[.,]", written)):
        if gd == wd:
            return True  # a phone number / code: the same digits, whatever the grouping
        ad = "".join(str(int(x)) for x in _additive([__import__("decimal").Decimal(x) for x in re.findall(r"\d+", " ".join(toks))]))
        return ad in (wd, wd.lstrip("0")) or ("0" + ad) == wd
    return False
