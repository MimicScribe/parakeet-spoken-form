"""English label consistency fixes for the mix (2026-10-18 audit): one spelling for what is said.

  R2  titles said in full: "Mr." -> "Mister", "Mrs." -> "Missus", "Dr." / "Prof." (before a capitalised name) -> "Doctor" /
      "Professor". Ambiguous abbreviations (St., Ms., Jr., Sr., "Dr." as Drive) are left alone and counted.
  R3  "per cent" -> "percent".
  R5  a number word capitalised in the middle of a stitched window / seam pair (the next clip's sentence start after a
      join that is no sentence end) -> lower case. Window and seam parts only.
  R6  stray symbols the caption kept: "-one point four" -> "minus one point four"; a row with a multiplication sign, a
      "=" / "==" markup run or a minus glued to a letter ("f-one") has no spoken form that is known -> dropped.
  R7  hyphenated fractions: "one-third" -> "one third" (other hyphenated words stay).
  R8  the number of a letter code written as the homophone "to": "B to B" / "B to C" / "D to C" / "P to P" -> "B two B" /
      "B two C" / "D two C" / "P two P" (letter+digit policy: the letters as written, the number as the spoken word
      "two"). Only a closed list of codes, so "from A to B" / "X to Y" / "E to P" stay.

    python en_relabel.py build --part <name>=<manifest.jsonl> [--part ...] --out relabel.jsonl --drops drops.jsonl

A relabelled row is the original row with its new "text" (keys kept; "text_old" and "relabel" added). Rows are keyed by
"audio_filepath". A timed row (word_times) whose token count would change is dropped instead of relabelled.
"""
import argparse
import collections
import json
import re

NUMW = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
        "eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion trillion").split()
CAP_NUM = re.compile(r"^(?:%s)$" % "|".join(w.capitalize() for w in NUMW))
TITLES = [(re.compile(r"\bMr\."), "Mister", False), (re.compile(r"\bMrs\."), "Missus", False),
          (re.compile(r"\bDr\.(?=\s+[A-Z])"), "Doctor", True), (re.compile(r"\bProf\.(?=\s+[A-Z])"), "Professor", True)]
UNRESOLVED = re.compile(r"\b(?:St|Ms|Jr|Sr)\.|\bDr\.(?!\s+[A-Z])|\bProf\.(?!\s+[A-Z])")
PER_CENT = re.compile(r"\b(per) (cent)\b", re.I)
FRAC = re.compile(r"\b((?:one|two|three|four|five|six|seven|eight|nine|ten|a))-((?:third|thirds|half|halves|quarter|quarters|"
                  r"fourth|fourths|fifth|fifths|sixth|sixths|seventh|sevenths|eighth|eighths|ninth|ninths|tenth|tenths))\b", re.I)
MINUS_LEAD = re.compile(r"(?<![\w−-])−(?=[a-z])")
# R8: letter-to-letter codes (B2B, B2C, B2P, B2G, B2E, D2C, P2P, C2C, G2G, O2O, M2M). Capital letters take the whole list; a
# lower-case text (a caption without capitals) only the two business codes, where "b to b" cannot be a range of letters.
CODE_PAIRS_UPPER = {"BB", "BC", "BP", "BG", "BE", "DC", "PP", "CC", "GG", "OO", "MM"}
CODE_PAIRS_LOWER = {"BB", "BC"}
NAME_TO = re.compile(r"(?<![\w'])([A-Za-z]) (to|too|tu)(?= [A-Za-z](?![\w']))( )([A-Za-z])(?![\w'])")


def _name_to(m) -> str:
    a, b = m.group(1), m.group(4)
    pair = (a + b).upper()
    ok = pair in (CODE_PAIRS_UPPER if a.isupper() and b.isupper() else CODE_PAIRS_LOWER if a.islower() and b.islower() else ())
    return f"{a} two {b}" if ok else m.group(0)


def fix_text(text: str, joined: bool = False) -> tuple:
    """(new text, [rules]) or (None, [reason]) for a row to drop. joined: a window / seam part (R5 applies)."""
    rules = []
    t = text
    if "×" in t or re.search(r"={2,}", t) or re.search(r"\w−", t):
        return None, ["R6_drop"]
    if "−" in t:
        t2 = MINUS_LEAD.sub("minus ", t)
        if "−" in t2:
            return None, ["R6_drop"]
        t, rules = t2, rules + ["R6"]
    for rx, word, _ in TITLES:
        t2 = rx.sub(word, t)
        if t2 != t:
            t, rules = t2, rules + ["R2"]
    t2 = PER_CENT.sub(lambda m: "Percent" if m.group(1)[0].isupper() else "percent", t)
    if t2 != t:
        t, rules = t2, rules + ["R3"]
    t2 = FRAC.sub(lambda m: m.group(1) + " " + m.group(2), t)
    if t2 != t:
        t, rules = t2, rules + ["R7"]
    t2 = NAME_TO.sub(_name_to, t)
    if t2 != t:
        t, rules = t2, rules + ["R8"]
    if joined:
        toks = t.split(" ")
        for i in range(1, len(toks)):
            core = re.sub(r"[^\w]", "", toks[i])
            if CAP_NUM.match(core) and not re.search(r"[.?!]['\"”’)]*$", toks[i - 1]):
                toks[i] = toks[i].replace(core, core.lower(), 1)
        t2 = " ".join(toks)
        if t2 != t:
            t, rules = t2, rules + ["R5"]
    return t, rules


def relabel_rows(rows: list, part: str, joined: bool) -> tuple:
    """(relabelled rows, dropped rows, counts)."""
    out, drops, c = [], [], collections.Counter()
    seen = set()
    for r in rows:
        key = r.get("audio_filepath")
        if UNRESOLVED.search(r.get("text", "")):
            c["unresolved_abbreviation"] += 1
        new, rules = fix_text(r["text"], joined)
        if new is None:
            drops.append({"part": part, "audio_filepath": key, "src": r.get("src"), "id": r.get("id"), "reason": rules[0],
                          "text": r["text"]})
            c["dropped_" + rules[0]] += 1
            continue
        if new == r["text"]:
            continue
        if r.get("word_times") is not None and len(new.split()) != len(r["text"].split()):
            drops.append({"part": part, "audio_filepath": key, "src": r.get("src"), "id": r.get("id"),
                          "reason": "token_count_changed", "text": r["text"]})
            c["dropped_token_count"] += 1
            continue
        assert key not in seen, f"{part}: audio_filepath {key} twice"
        seen.add(key)
        out.append(dict(r, text=new, text_old=r["text"], relabel=rules, part=part))
        for x in rules:
            c[x] += 1
    return out, drops, dict(c)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--part", action="append", required=True, help="<volume path>=<local manifest>")
    b.add_argument("--out", required=True)
    b.add_argument("--drops", required=True)
    a = ap.parse_args()
    tot = collections.Counter()
    with open(a.out, "w", encoding="utf-8") as fo, open(a.drops, "w", encoding="utf-8") as fd:
        for spec in a.part:
            part, path = spec.split("=", 1)
            rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
            joined = part.rsplit("/", 1)[-1] in ("win_en.jsonl", "seam_en.jsonl")
            out, drops, c = relabel_rows(rows, part, joined)
            for r in out:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            for r in drops:
                fd.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(part, len(rows), "rows;", c)
            tot.update(c)
    print("TOTAL", dict(tot))


if __name__ == "__main__":
    main()
