"""Replay data: real English speech whose labels contain no written-form characters.

LibriSpeech-PC (CC BY 4.0) supplies punctuated, cased labels for LibriSpeech audio (CC BY 4.0).
Writes, under /vol/replay:
  lspc_train.jsonl          train-clean-100 rows, digit-free, chapter headings dropped
  lspc_train_numwords.jsonl the subset containing English number words (upweighted in training)
  lspc_dev.jsonl            dev-clean rows (validation)
"""

import json
import os
import re
import subprocess

from common import NUMBER_WORDS, WRITTEN_FORM

SLR = "https://www.openslr.org/resources"
HEADING = re.compile(r"\b[A-Z]{2,}\b(?:\s+[A-Z]{2,}\b)+")

# American spelling for the replay labels (Gutenberg books mix in British spellings; one rule).
US_OUR = ("honour favour labour humour colour neighbour behaviour harbour rumour vapour vigour splendour valour "
          "odour ardour savour endeavour clamour parlour armour candour fervour rigour tumour saviour").split()
US_ISE = ("realise recognise organise apologise civilise criticise sympathise emphasise characterise "
          "authorise memorise surprise").split()


def us_spelling(text: str) -> str:
    """-our -> -or for the listed stems; -ise -> -ize only before a verb ending, so
    criticism, characteristic, emphasis and organism stay as they are."""
    def fix(m):
        w = m.group(0)
        low = w.lower()
        out = None
        for stem in US_OUR:
            if low.startswith(stem) and re.fullmatch(r"(s|ed|ing|ite|ites|able|ably|ful|less|er|ers|ers)?", low[len(stem):]):
                out = stem[:-3] + "or" + low[len(stem):]
        for stem in US_ISE:
            base = stem[:-3]
            tail = low[len(base) + 2:]
            if stem != "surprise" and low.startswith(base + "is") and re.fullmatch(r"(e|ed|es|ing|ation|ations|er|ers)", tail):
                out = base + "iz" + tail
        if out is None:
            return w
        return out.capitalize() if w[0].isupper() else out
    return re.sub(r"[A-Za-z]+", fix, text)


def prepare(vol: str, splits=("train-clean-100", "dev-clean")) -> None:
    root = f"{vol}/replay"
    ls = f"{vol}/librispeech"
    if not any(f.endswith(".json") for _, _, fs in os.walk(f"{root}/manifests") for f in fs):
        _fetch_tar(f"{SLR}/145/manifests.tar.gz", f"{root}/manifests")
    for split in splits:
        if not os.path.isdir(f"{ls}/LibriSpeech/{split}"):
            _fetch_tar(f"{SLR}/12/{split}.tar.gz", ls)
    found = {}
    for dirpath, _, files in os.walk(f"{root}/manifests"):
        for fn in files:
            found[fn] = os.path.join(dirpath, fn)
    print("manifests:", sorted(found))

    def load(split):
        rows, dropped = [], 0
        for line in open(found[f"{split}.json"]):
            r = json.loads(line)
            if WRITTEN_FORM.search(r["text"]) or HEADING.search(r["text"]):
                dropped += 1
                continue
            path = f"{ls}/LibriSpeech/{r['audio_filepath']}"
            rows.append({"audio_filepath": path, "duration": r["duration"], "text": us_spelling(r["text"]),
                         "id": os.path.basename(path)[:-5]})
        missing = sum(not os.path.exists(r["audio_filepath"]) for r in rows[:200])
        print(f"{split}: {len(rows)} rows, {sum(r['duration'] for r in rows) / 3600:.1f} h, "
              f"dropped {dropped}, missing audio in first 200: {missing}")
        return rows

    def write(name, rows):
        with open(f"{root}/{name}", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    train = load("train-clean-100")
    write("lspc_train.jsonl", train)
    num = [r for r in train if NUMBER_WORDS.search(r["text"])]
    print(f"number-word rows: {len(num)}, {sum(r['duration'] for r in num) / 3600:.1f} h")
    write("lspc_train_numwords.jsonl", num)
    write("lspc_dev.jsonl", load("dev-clean"))
