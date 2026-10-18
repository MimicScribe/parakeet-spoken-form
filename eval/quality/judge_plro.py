"""pl / ro judge (next tier, like the ru judge): the second transcript (sliced to the row span; it writes digits) must
(1) hold every span's VALUE (caption digits) as a number, or the span's reading as words, checked on EVERY row, and (2)
for real speech, the tightening's run-scoped cer_ext <= 0.10. The slice's word agreement with the label (WER of the
non-number words) is a cross-check only: a real row over 0.25 is kept but FLAGGED ("wer_flag" in judge_rows, counted in
judge_stats "kept_wer_flagged"): on the local sample those rows were slicing / scale-word artefacts of the WER, not label
errors, and rejecting them dropped ~6% of the real rows. Synthetic rows (census_common.synthetic: the transcript is of
the whole clip) get the value check only. Writes jr/judge_stats.json + reject_ids; with --upload merges the rejects into
ml/p3v2/<lang>/reject_ids.txt and writes the judged_<pass>.json marker build4 requires.
A re-judge with --upload replaces the pass's own rejects in the merged list (other passes keep theirs).
Work dir: $SPOKEN_FORM_JUDGE_ROOT (default /tmp/spoken_form_judge)/<lang>/<pass>/ with <pass>.jsonl and ext.jsonl.
usage: judge_plro.py <lang> <pass> [--upload]"""
import difflib, json, re, sys
from pathlib import Path
import os  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import volume_io as V  # noqa: E402
sys.path.insert(0, str(V.PIPE))
from census_common import synthetic  # noqa: E402
lang, name = sys.argv[1], sys.argv[2]
up = "--upload" in sys.argv
W = Path(os.environ.get("SPOKEN_FORM_JUDGE_ROOT", "/tmp/spoken_form_judge")) / lang / name
# this pass's rejects as last UPLOADED into the merged list: a re-judge takes back only these (ids can repeat across
# passes). Read from jr/uploaded_reject_ids.txt, written only after a successful upload (jr/reject_ids.txt is rewritten
# by every run, uploaded or not, so it is not what the merged list holds)
_OLD_OWN = ({x.strip() for x in open(W / "jr/uploaded_reject_ids.txt") if x.strip()}
            if (W / "jr/uploaded_reject_ids.txt").exists() else set())
ext = {json.loads(l)["id"]: json.loads(l) for l in open(W / "ext.jsonl")}
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.U)


def nums(s):
    """numbers written in s: '20.000' -> 20000, '2,5' -> 2.5, clock '18.00' / '14.45' -> 18 / 14, 45,
    '94-46-2002-58' -> 94, 46, 2002, 58 (a plain space never joins: '2020 179' is two numbers)"""
    out = set()
    for m in re.finditer(r"\b(\d{1,2})[.:](\d{2})\b(?![.,]\d)", s):
        out.add(float(m.group(1)))
        if int(m.group(2)):
            out.add(float(m.group(2)))
    s = re.sub(r"\b\d{1,2}[.:]\d{2}\b(?![.,]\d)", " ", s)
    s = re.sub(r"(?<=\d)[.\u00a0](?=\d{3}\b)", "", s)
    for m in re.finditer(r"(?<![\d.])(\d+)\.(\d{1,2})(?![\d.])", s):  # "13.5" written with a dot: one decimal
        out.add(float(f"{m.group(1)}.{m.group(2)}"))
    s = re.sub(r"(?<![\d.])\d+\.\d{1,2}(?![\d.])", " ", s)
    for m in re.finditer(r"\d+(?:,\d+)?", s):
        try:
            out.add(float(m.group(0).replace(",", ".")))
        except ValueError:
            pass
    return out


# extra ways a span value may be found (each only ADDS acceptances; the rule above is unchanged), 2026-10-17 for uk.
# Each one needs the parts of the value ADJACENT in the transcript, never scattered numbers:
#  - a space-grouped figure in the caption span ("5 500", "6 000 000") is one number, not "5" and "500";
#  - a figure said as scale parts ("1 мільйон 200 тисяч" for 1 200 000): one whole run, no partial sums;
#  - a date (dd.mm.yyyy): the day, then the month NAME or the numeric month right after it ("18 квітня", "18.04"),
#    and the year as a number;
#  - a price with cents: major and minor next to each other ("18,50", "18 євро 50 центів"), not a clock "18:50";
#  - a clock time: hour and minutes next to each other ("18:05", "18.05", "18-05", "18 і 05", "9 0 6"), or with a
#    comma ("18,05") when no currency word follows;
#  - a phone / code run of >= 7 digits (all the row's code / chain / phone spans, joined) inside ONE phone-looking
#    digit sequence of the transcript (groups separated by spaces / hyphens / dots / brackets), in order. Synthetic
#    rows (no cer_ext): the digits, also as said ("800 90" for "вісімсот дев'яносто" = 890); real rows: the caption's
#    digit groups themselves, since the reading says the groups.
SCALE_WORD = re.compile(r"(?:тисяч[аіу]?|тысяч[аиу]?|мільйон(?:а|и|ів|у)?|миллион(?:а|ы|ов)?|мільярд(?:а|и|ів|у)?|"
                        r"миллиард(?:а|ы|ов)?|"
                        # 2026-10-17 long-number sets: pl ro sv cs sk hu fi hr bg ("23 miliárd", "600 tisíc", "817 tysięcy")
                        r"tysi[ąę]c\w*|tys|tisíc\w*|tisuć\w*|tisuć|хиляд\w*|mii|tusen|ezer|tuhat\w*|"
                        r"milion\w*|milión\w*|milioan\w*|milijun\w*|милион\w*|miljon\w*|millió\w*|miljoon\w*|mln|"
                        r"miliard\w*|miliárd\w*|milijard\w*|милиард\w*|miljard\w*|milliárd\w*|mld)")
SCALE_OF = (("мільярд", 1e9), ("миллиард", 1e9), ("мільйон", 1e6), ("миллион", 1e6), ("тисяч", 1e3), ("тысяч", 1e3),
            ("miliard", 1e9), ("miliárd", 1e9), ("milijard", 1e9), ("милиард", 1e9), ("miljard", 1e9), ("milliárd", 1e9), ("mld", 1e9),
            ("milion", 1e6), ("milión", 1e6), ("milioan", 1e6), ("milijun", 1e6), ("милион", 1e6), ("miljon", 1e6), ("millió", 1e6),
            ("miljoon", 1e6), ("mln", 1e6), ("tysi", 1e3), ("tisíc", 1e3), ("tisuć", 1e3), ("хиляд", 1e3), ("mii", 1e3),
            ("tusen", 1e3), ("ezer", 1e3), ("tuhat", 1e3), ("tys", 1e3))  # "tys": the Polish abbreviation "4 tys."
CUR_WORD = re.compile(r"(?:грн|гривн\w*|гривен\w*|долар\w*|доллар\w*|євро|евро|центи?\w*|копій\w*|копе\w*|руб\w*|zł\w*|"
                      r"złot\w*|lei|leu|euro\w*|dolar\w*|\$|€|₴)", re.I)


def scale_sums(et):
    """'1 мільйон 200 тисяч' -> {1200000}; '1 мільйон 200 тисяч 300' -> {1200300}: each whole run of number + scale
    word (+ one adjacent plain number), never a partial sum."""
    out, tot, prev_end = set(), 0.0, None
    toks = list(re.finditer(r"(\d+(?:,\d+)?)(?:\s+([^\W\d_]+))?", et))
    for m in toks:
        v = float(m.group(1).replace(",", "."))
        w = (m.group(2) or "").lower()
        adjacent = prev_end is not None and not et[prev_end:m.start()].strip()
        if w and SCALE_WORD.fullmatch(w):
            if tot and not adjacent:
                out.add(tot)
                tot = 0.0
            tot += v * next(f for st, f in SCALE_OF if w.startswith(st))
            prev_end = m.end()
            continue
        if tot:
            out.add(tot + v if adjacent else tot)
        tot, prev_end = 0.0, None
    if tot:
        out.add(tot)
    return out


def said_digits(seq):
    """Digits of one digit sequence with said compounds joined: '097 800 90 29 48' -> '0978902948' (800 + 90 = 890)."""
    toks = re.findall(r"\d+", seq)
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if i + 1 < len(toks):
            n = toks[i + 1]
            if re.fullmatch(r"[1-9]00", t) and len(n) <= 2 and n[0] != "0":
                if i + 2 < len(toks) and re.fullmatch(r"[1-9]0", n) and len(toks[i + 2]) == 1 and toks[i + 2] != "0":
                    out.append(str(int(t) + int(n) + int(toks[i + 2]))); i += 3; continue
                out.append(str(int(t) + int(n))); i += 2; continue
            if re.fullmatch(r"[1-9]0", t) and len(n) == 1 and n != "0":
                out.append(str(int(t) + int(n))); i += 2; continue
        out.append(t); i += 1
    return "".join(out)


def phone_runs(et):
    """Each phone-looking digit sequence of the transcript (digit groups joined by spaces / hyphens / dots / brackets):
    its plain digits and its said-compound digits."""
    out = []
    for m in re.finditer(r"\(?\d+\)?(?:[\s\-.]+\(?\d+\)?)*", et):
        out += [re.sub(r"\D", "", m.group(0)), said_digits(m.group(0))]
    return out


MONTHS = {"uk": "січня лютого березня квітня травня червня липня серпня вересня жовтня листопада грудня",
          "ru": "января февраля марта апреля мая июня июля августа сентября октября ноября декабря",
          "pl": "stycznia lutego marca kwietnia maja czerwca lipca sierpnia września października listopada grudnia",
          "ro": "ianuarie februarie martie aprilie mai iunie iulie august septembrie octombrie noiembrie decembrie",
          # 2026-10-17 long-number sets: the month as a transcript writes it after a day number (case forms of the
          # day + month phrase; up to two more letters are tolerated below, e.g. hr "studenog" / "studenoga")
          "sv": "januari februari mars april maj juni juli augusti september oktober november december",
          "cs": "ledna února března dubna května června července srpna září října listopadu prosince",
          "sk": "januára februára marca apríla mája júna júla augusta septembra októbra novembra decembra",
          "hu": "január február március április május június július augusztus szeptember október november december",
          "fi": "tammikuuta helmikuuta maaliskuuta huhtikuuta toukokuuta kesäkuuta heinäkuuta elokuuta syyskuuta "
                "lokakuuta marraskuuta joulukuuta",
          "hr": "siječnja veljače ožujka travnja svibnja lipnja srpnja kolovoza rujna listopada studenog prosinca",
          "bg": "януари февруари март април май юни юли август септември октомври ноември декември"}
# languages that put the month name BEFORE the day ("2021. március 18-án")
MONTH_FIRST = {"hu"}
# minor currency units: "96 551 zł i 90 groszy", "72 kronor och 99 öre" (a longer gap is only accepted when one follows)
MINOR_WORD = r"(?:groszy|grosze|grosz|bani|ban|öre|копеек|копейки|копейка|копійок|копійки|копійка|центів|центи|цент|центов|цента|cents?|centów|centy|halierov|haléřů|fillér|senttiä|centů|eurocentů|centov|eurocentov|centi|centa|lipa|lipe|lipi|стотинки|стотинка|senttiä|sentin)"


def strict(sp, row_phone):
    """A span whose value has several parts (date, clock, cents price, phone run of >= 7 digits): the plain rule
    ("every number of the span somewhere in the transcript") would pass scattered numbers, so only alt_ok decides."""
    t, k = sp.get("text", "").strip().rstrip(",.;:!?"), sp.get("kind")  # "18.3.2021," -> the date itself
    return bool((k == "date" and re.fullmatch(r"\d{1,2}[./]\d{1,2}[./]\d{4}", t))
                or (k == "clock" and re.fullmatch(r"\d{1,2}:\d{2}", t))
                or (k == "cur" and re.fullmatch(r"\D*?\d+,\d{2}\D*", t))
                or (k in ("code", "chain", "phone") and len(row_phone) >= 7))


def alt_ok(sp, et, en, scaled, row_phone="", synthetic=False, row_groups=()):
    t, k = sp.get("text", "").strip().rstrip(",.;:!?"), sp.get("kind")  # a trailing comma / dot is not part of the value
    tl = et.lower()
    # 2026-10-17 long-number sets: a range of grouped figures ("420 000 000 000–630 000 000 000"), heard with scale
    # words ("420 miliárd až 630 miliárd") or one shared scale word ("300 és 750 ezer"): EACH endpoint's whole value must
    # be a transcript number or a whole number + scale-word run; a bare first endpoint may take the second one's scale
    rg = re.fullmatch(r"\D*?(\d{1,3}(?:[ .\u00a0\u202f]\d{3})*)\s*[-–]\s*(\d{1,3}(?:[ .\u00a0\u202f]\d{3})*)\D*", t)
    if k in ("range", "int") and rg:
        lo, hi = (float(re.sub(r"\D", "", x)) for x in rg.groups())
        runs = scale_sums(et)
        have = en | runs
        shared = any(hi in runs and lo % f == 0 and hi % f == 0 and lo / f in en and hi / f in en for f in (1e3, 1e6, 1e9))
        if hi in have and (lo in have or shared):
            return True
    g = re.fullmatch(r"\D*?(\d{1,3}(?:[ \u00a0\u202f]\d{3})+)(?:,(\d+))?\D*", t)
    if k in ("int", "cur", "pct", "unit", "dec") and g:
        v = float(re.sub(r"\D", "", g.group(1)) + ("." + g.group(2) if g.group(2) else ""))
        if v in en or (not g.group(2) and k in ("int", "cur") and v in scale_sums(et)):
            return True
    d = re.fullmatch(r"(\d{1,2})[./](\d{1,2})[./](\d{4})", t.strip())
    if k == "date" and d:
        day, mon, yr = int(d.group(1)), int(d.group(2)), int(d.group(3))
        if not (1 <= mon <= 12 and 1 <= day <= 31) or float(yr) not in en:
            return False
        mw = MONTHS.get(lang, "").split()
        name = mw[mon - 1] if len(mw) == 12 else None
        return bool((name and re.search(rf"(?<!\d)0?{day}\.?\s+{name}[^\W\d_]{{0,2}}(?![^\W\d_])", tl))  # "15. maaliskuuta"
                    or (name and lang in MONTH_FIRST and re.search(rf"(?<![^\W\d_]){name}[^\W\d_]{{0,2}}\s+0?{day}(?!\d)", tl))
                    or re.search(rf"(?<!\d)0?{day}[./\s]0?{mon}(?!\d)", tl))
    c = re.fullmatch(r"\D*?(\d+),(\d{2})\D*", t)
    if k == "cur" and c:
        a, b = int(c.group(1)), int(c.group(2))
        return bool(re.search(rf"(?<![\d:]){a}[,.]{b:02d}(?![\d:])", tl)
                    or re.search(rf"(?<![\d:]){a}\s+[^\W\d_]+\s+0?{b}(?![\d:])", tl)
                    or re.search(rf"(?<![\d:]){a}\s+[^\W\d_]+\.?\s+(?:i|och|și|si|и|і|and|et|y|e|und|en|a|és|ja)\s+0?{b}\s+(?:de\s+)?{MINOR_WORD}(?![^\W\d_])", tl))
    h = re.fullmatch(r"(\d{1,2}):(\d{2})", t.strip())
    if k == "clock" and h:
        hh, mm = int(h.group(1)), h.group(2)
        mpat = rf"(?:{mm}|0\s{int(mm)})" if mm.startswith("0") and mm != "00" else mm
        if re.search(rf"(?<![\d,.:]){hh}(?:[:.]|\s*[-–]\s*|\s+(?:і|и|i|și|si|och|és|ja|a)\s+|\s+){mpat}(?![\d,.:])", tl) or (
                mm == "00" and re.search(rf"(?<![\d,.:]){hh}(?![\d,.:])", tl)):
            return True
        m = re.search(rf"(?<![\d,.:]){hh},{mm}(?![\d,.:])\s*(\S*)", tl)
        return bool(m) and not CUR_WORD.match(m.group(1))
    if k in ("code", "chain", "phone"):
        sd = row_phone or re.sub(r"\D", "", t)
        if len(sd) < 7:
            return False
        if synthetic:  # the label is what the voice was given: the digits, in order, in one sequence
            return any(sd in run for run in phone_runs(et))
        # real speech: the caption's digit GROUPS, in order, inside one transcript sequence (the reading says groups:
        # "1698-2005" heard for a caption "16-98-2005" is a different reading of the same digits)
        gs = list(row_groups)
        for m in re.finditer(r"\(?\d+\)?(?:[\s\-.]+\(?\d+\)?)*", et):
            ts = re.findall(r"\d+", m.group(0))
            if any(ts[i:i + len(gs)] == gs for i in range(len(ts) - len(gs) + 1)):
                return True
        return False
    return False


def scaled_values(en, et, sp_text):
    """The transcript numbers a span value may match, scale words taken into account only where one is written: x * k
    when the transcript writes x right before a scale word of factor k ("4 tysięcy" for a caption "4000"; scale_sums,
    or with the Romanian "de" between them), then / k when the caption span itself carries a scale word of factor k
    ("4 tys." with a transcript "4000")."""
    out = set(en)
    fs = {next(f for st, f in SCALE_OF if w.startswith(st))
          for w in re.findall(r"[^\W\d_]+", sp_text.lower()) if SCALE_WORD.fullmatch(w)}
    # a number + scale word with the Romanian linker "de" between them ("900 de milioane"): x * k
    for m in re.finditer(r"(\d+(?:,\d+)?)\s+de\s+([^\W\d_]+)", et):
        w = m.group(2).lower()
        if SCALE_WORD.fullmatch(w):
            out.add(float(m.group(1).replace(",", ".")) * next(f for st, f in SCALE_OF if w.startswith(st)))
    # a figure grouped by two or more commas ("29,660,000,000") is one whole number (no decimal has two commas)
    out |= {float(m.group(0).replace(",", "")) for m in re.finditer(r"(?<![\d,.])\d{1,3}(?:,\d{3}){2,}(?![\d,])", et)}
    out |= scale_sums(et)
    return out | {x / f for x in out for f in fs}  # the span's own scale word: "20.000 de milioane" = 20 miliarde


def reading_heard(rd, el, label_words):
    """The span's reading, as whole words, at least as often in the transcript as in the label (a reading word that is
    also an ordinary word elsewhere in the sentence does not stand in for the number); an empty reading never is."""
    if not rd.strip():
        return False
    lab = " " + " ".join(label_words) + " "
    return el.count(rd) >= max(1, lab.count(rd))


def wer(a, b):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in sm.get_opcodes() if t != "equal") / max(1, len(a))


rej, reasons, rows_out, n = [], {"no_ext": 0, "span_missing": 0, "cer_ext": 0}, [], 0
n_wer_flag = 0
ids_seen = []
wers = []
for l in open(W / f"{name}.jsonl"):
    r = json.loads(l); n += 1
    ids_seen.append(r["id"])
    e = ext.get(r["id"])
    et = (e or {}).get("text") or ""
    if not et.strip():
        rej.append(r["id"]); reasons["no_ext"] += 1; rows_out.append({"id": r["id"], "reason": "no_ext"}); continue
    spans = r.get("spans") or []
    if isinstance(spans, str):
        import ast
        spans = ast.literal_eval(spans)
    en, el = nums(et), " " + " ".join(WORD.findall(et.lower())) + " "
    lw = WORD.findall(r["text"].lower())
    lab_words = list(lw)
    for s_ in spans:  # drop each span's reading words before the word check
        rw = WORD.findall((s_.get("reading") or "").lower())
        for k in range(len(lw) - len(rw) + 1):
            if rw and lw[k:k + len(rw)] == rw:
                del lw[k:k + len(rw)]; break
    ew = WORD.findall(re.sub(r"\d[\d.,%-]*", " ", et.lower()))
    w = wer(lw, ew); wers.append(w)
    # words: the tightening already measured each kept run against this transcript (cer_ext, run-scoped); the slice
    # here is only a cross-check, and a slice that disagrees badly is a slicing error, not evidence against the row
    syn = synthetic(r)  # an explicit marker (src / origin tts, synthetic=True), not a missing field
    ce = float(r.get("cer_ext") or 0)
    if ce > 0.10:
        rej.append(r["id"]); reasons["cer_ext"] += 1; rows_out.append({"id": r["id"], "reason": "cer_ext", "cer_ext": ce}); continue
    bad = None
    # values: every span of every row (a slice that disagrees with the row is no reason to skip the check)
    for s_ in spans:
        v = nums(s_.get("text", ""))
        rd = " " + " ".join(WORD.findall((s_.get("reading") or "").lower())) + " "
        scaled = scaled_values(en, et, s_.get("text", ""))  # "4 tys." / "4 tysięcy": only with a written scale word
        ph_sp = [x for x in spans if x.get("kind") in ("code", "chain", "phone")]
        phone = "".join(re.sub(r"\D", "", x.get("text", "")) for x in ph_sp)
        groups = [g_ for x in ph_sp for g_ in re.findall(r"\d+", x.get("text", ""))]
        heard = reading_heard(rd, el, lab_words)
        if strict(s_, phone):  # multi-part values: scattered numbers never count (only the adjacent forms or words)
            ok = heard or alt_ok(s_, et, en, scaled, phone, syn, groups)
        else:
            ok = (v and all(x in scaled for x in v)) or heard or alt_ok(s_, et, en, scaled, phone, syn, groups)
        if not ok:
            bad = s_.get("text"); break
    if bad is not None:
        rej.append(r["id"]); reasons["span_missing"] += 1; rows_out.append({"id": r["id"], "reason": "span_missing", "span": bad}); continue
    # words (real speech): a slice that disagrees with the label (WER > 0.25) is flagged for a listen, not rejected (the
    # values above were checked on it all the same; cer_ext already measured the words run by run)
    flag = not syn and w > 0.25
    n_wer_flag += flag
    rows_out.append({"id": r["id"], "reason": "kept", "wer": round(w, 3)} | ({"wer_flag": True} if flag else {}))
(W / "jr").mkdir(exist_ok=True)
wers.sort()
st = {"lang": lang, "rows": n, "rejected": len(rej), "reject_rate": round(len(rej) / max(1, n), 4), "reasons": reasons,
      "judge": "plro: tightening cer_ext<=0.10 + every span value in the slice (slice WER > 0.25 flagged)",
      "kept_wer_flagged": n_wer_flag, "wer_p50": wers[len(wers) // 2] if wers else None,
      "wer_p90": wers[int(.9 * len(wers))] if wers else None}
json.dump(st, open(W / "jr/judge_stats.json", "w"), indent=1)
with open(W / "jr/judge_rows.jsonl", "w") as f:
    for x in rows_out:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
(W / "jr/reject_ids.txt").write_text("".join(i + "\n" for i in rej))
print(json.dumps(st))
if up:
    import fcntl
    merged = W.parent / "reject_ids.txt"
    lang_lock = open(W.parent / ".reject_ids.lock", "w")  # the same per-language lock run_judge takes
    fcntl.flock(lang_lock, fcntl.LOCK_EX)
    prev = set()
    # fail closed: a merged list that exists but cannot be read is an error (V.read_shared raises), never an empty start
    if V.read_shared(f"ml/p3v2/{lang}/reject_ids.txt", merged):
        prev = {x.strip() for x in open(merged) if x.strip()}
    # a re-judge REPLACES this pass's own earlier rejects; other passes' rejects stay, even for an id this pass also
    # holds (ids can repeat across passes, e.g. a variety pass re-cutting a recovery pass's clip)
    allr = sorted((prev - (_OLD_OWN - set(rej))) | set(rej))
    merged.write_text("".join(x + "\n" for x in allr))
    V.put(merged, f"ml/p3v2/{lang}/reject_ids.txt")
    V.put(W / "jr/judge_stats.json", f"ml/p3v2/{lang}/judged_{name}.json")
    (W / "jr/uploaded_reject_ids.txt").write_text("".join(i + "\n" for i in rej))  # what the merged list now holds
    print(f"{lang}/{name}: uploaded; reject_ids.txt now {len(allr)}")
