"""Expand carriers x readings into training rows.

Each row has the target text (what the model should write: the words spoken) and the text given
to the synthesizer (same words, with pronunciation hints). Rows are deterministic for a seed.
Dev rows use held-out carriers; voices are split separately in voice.py.

    python gen.py --rows 2000 --seed 1 --out rows.jsonl
"""

import argparse
import json
import os
import random
import re
import zlib

import readings

HERE = os.path.dirname(os.path.abspath(__file__))
SLOT = re.compile(r"\{(\w+)\}")

# Sentences where "one", "point", "second", "quarter", "for", "to" are words, not numbers.
CONTROLS = [
    "No one wants to own that service.", "At this point the fix is obvious.",
    "That was a one off, I promise.", "Point taken, let's move on.", "Give me a second to check.",
    "The quarter ended on a high note.", "One of us should write it down.",
    "I'll go for the simpler option.", "We need to talk to the vendor.", "That's too much to ask.",
    "Which one did you pick?", "She won the bid in the end.", "I ate before the call.",
    "It's a two way street.", "The point of the demo is speed.", "Someone has to say no.",
    "I'm all for it, to be honest.", "Let's take this one step at a time.",
    "Everyone was there except Sam.", "We went back and forth for a while.",
    "It's one thing to plan and another to ship.", "Those two are the same.",
    "Could you send it to me too?", "This one's on me.", "Second, the tests are flaky.",
    "First, let's agree on scope.", "He made a good point earlier.", "I won't be able to make it.",
]


def load_carriers(path=os.path.join(HERE, "carriers.txt")):
    return [l.rstrip("\n") for l in open(path) if l.strip() and not l.startswith("#")]


def is_dev_template(text: str, dev_frac: float) -> bool:
    return (zlib.crc32(text.encode()) % 1000) / 1000 < dev_frac


def fill(carrier: str, r: random.Random):
    target, tts, kinds = carrier, carrier, []
    for m in list(SLOT.finditer(carrier)):
        kind = m.group(1)
        slot = readings.KINDS[kind](r)
        kinds.append(kind)
        target = target.replace(m.group(0), slot, 1)
        tts = tts.replace(m.group(0), readings.tts_form(slot), 1)
    if carrier.startswith("{"):
        target, tts = target[0].upper() + target[1:], tts[0].upper() + tts[1:]
    return target, tts, kinds


def rows(n: int, seed: int, dev_frac: float = 0.2, control_frac: float = 0.1):
    r = random.Random(seed)
    carriers = load_carriers()
    for i in range(n):
        if r.random() < control_frac:
            text = r.choice(CONTROLS)
            yield {"id": f"r{seed}_{i}", "split": "dev" if is_dev_template(text, dev_frac) else "train",
                   "template": text, "kinds": ["control"], "text": text, "tts_text": text}
            continue
        carrier = r.choice(carriers)
        target, tts, kinds = fill(carrier, r)
        yield {"id": f"r{seed}_{i}", "split": "dev" if is_dev_template(carrier, dev_frac) else "train",
               "template": carrier, "kinds": kinds, "text": target, "tts_text": tts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.out, "w") as f:
        for row in rows(a.rows, a.seed):
            assert not readings_written_form(row["text"]), row
            f.write(json.dumps(row) + "\n")


def readings_written_form(text: str) -> bool:
    return bool(re.search(r"[0-9$%€£/]", text))


if __name__ == "__main__":
    main()
