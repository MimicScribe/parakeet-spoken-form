"""Find the existing synthetic rows whose label (and so whose AUDIO) has a known generator grammar slip
(grammar_rules), and write their ids as ml/p3v2/<lang>/grammar_reject_ids.txt, which mix_build.build4 keeps out of
training. The audio speaks the wrong form, so these rows are excluded, never relabelled.

    python3 grammar_scan.py <root holding ml/<set>/<lang>/rows.jsonl> --out <dir> [--report scan.json]

<root> is a copy of the volume's synthetic sets (mix_build.TTS_SETS). Per language the list is written to
<out>/ml/p3v2/<lang>/grammar_reject_ids.txt; an existing list is never overwritten (the script stops instead).
"""

import argparse
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import grammar_rules as G  # noqa: E402
from mix_build import GRAMMAR_REJECTS, TTS_SETS  # noqa: E402


def scan(root: str, sets=None) -> dict:
    """{"ids": {lang: {id: [codes]}}, "counts": {lang: {code: {set: rows}}}, "rows": {lang: {set: rows scanned}}}."""
    sets = sets or [t for t, _ in TTS_SETS]
    ids, counts, seen = {}, {}, {}
    for tag in sets:
        for p in sorted(glob.glob(f"{root}/ml/{tag}/*/rows.jsonl")):
            lang = os.path.basename(os.path.dirname(p))
            for line in open(p, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                seen.setdefault(lang, collections.Counter())[tag] += 1
                errs = G.row_errors(lang, r)
                if not errs:
                    continue
                ids.setdefault(lang, {})[r["id"]] = errs
                for c in errs:
                    counts.setdefault(lang, {}).setdefault(c, collections.Counter())[tag] += 1
    return {"ids": ids, "counts": {l: {c: dict(v) for c, v in cs.items()} for l, cs in counts.items()},
            "rows": {l: dict(v) for l, v in seen.items()}}


def write_lists(res: dict, out: str) -> dict:
    """One sorted id list per language with hits; refuses to overwrite an existing list. Returns {lang: path}."""
    paths = {}
    for lang, rows in sorted(res["ids"].items()):
        p = f"{out}/ml/p3v2/{lang}/{GRAMMAR_REJECTS}"
        if os.path.exists(p):
            raise FileExistsError(p)
        paths[lang] = p
    for lang, p in paths.items():
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write("# synthetic rows whose audio speaks a generator grammar slip (grammar_scan.py)\n")
            f.writelines(f"{i}\n" for i in sorted(res["ids"][lang]))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report")
    a = ap.parse_args()
    res = scan(a.root)
    paths = write_lists(res, a.out)
    if a.report:
        json.dump(res, open(a.report, "w"), ensure_ascii=False, indent=1)
    for lang in sorted(res["counts"]):
        print(lang, len(res["ids"][lang]), "rows", {c: sum(v.values()) for c, v in res["counts"][lang].items()},
              "->", paths.get(lang))


if __name__ == "__main__":
    main()
