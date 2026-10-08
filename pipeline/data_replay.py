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
