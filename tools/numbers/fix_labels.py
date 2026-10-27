"""Label fixes (owner 2026-10-24) for every synthetic sentence row, keyed by audio row id:
  - digit-string labels without dashes (real labels never have them unless "dash" is spoken);
  - codes stay digits only when the label is ONE letter-digit token (3D, B12, W-2, COVID-19); word+number codes and bare
    numbers ("Chapter 11", "Windows 10", "911", "529 plan") are labelled as spoken words;
  - drop rows whose digit/code reading is not a whole-word match, occurs twice, or touches a stray number word
    (would give half-digit labels).
  python3 fix_labels.py --root DATA --src v2=sentences/v2.jsonl --src sup1=sentences/sup1.jsonl   -> labels_fix.json {row_id: label|null}"""
import collections, hashlib, json, re

NUMW = set("zero oh one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen "
           "nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand double triple".split())


def keep_digits(n):
    lab = n.get("label") or ""
    if n["kind"] == "product or code name":
        return bool(re.search(r"[A-Za-z]", lab)) and " " not in lab
    return bool(lab) and lab != n["reading"] and re.fullmatch(r"[\d-]+", lab) is not None


def relabel(row):
    said, lab = row["said"], row["said"]
    for n in row["numbers"]:
        if not keep_digits(n): continue
        pat = re.compile(r"(?<![A-Za-z'])" + re.escape(n["reading"]) + r"(?![A-Za-z'])", re.I)
        hits = list(pat.finditer(lab))
        if len(hits) != 1: return None, "reading_not_once"
        a, b = hits[0].span()
        before = re.findall(r"[A-Za-z']+", lab[:a])[-1:]; after = re.findall(r"[A-Za-z']+", lab[b:])[:1]
        if (before and before[0].lower() in NUMW) or (after and after[0].lower() in NUMW): return None, "adjacent_number_word"
        new = n["label"].replace("-", "") if n["kind"] != "product or code name" else n["label"]
        lab = lab[:a] + new + lab[b:]
    return lab, "ok"


def main():
    import argparse, os
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); ap.add_argument("--prefix", default="pack")
    ap.add_argument("--src", action="append", required=True, help="tag=path of a sentences JSONL")
    a = ap.parse_args(); os.chdir(a.root)
    out, why, changed = {}, collections.Counter(), collections.Counter()
    for tag, path in (x.split("=", 1) for x in a.src):
        for row in map(json.loads, open(path)):
            rid = f"{a.prefix}_{tag}_" + hashlib.sha1((row["cell"] + row["said"] + row["id"]).encode()).hexdigest()[:10]
            lab, w = relabel(row); why[w] += 1; out[rid] = lab
            if lab is not None and lab != row["label_v3"]: changed[tag] += 1
    json.dump(out, open("labels_fix.json", "w"))
    print("rows", len(out), "outcome", dict(why), "labels changed by tag", dict(changed))


if __name__ == "__main__": main()
