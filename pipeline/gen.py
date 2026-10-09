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

# Sentences where number-like words are ordinary words, not numbers. The model must not turn
# them into number words (kill2 heard "EBIT" as "eight").
CONTROLS = [
    "I ate before the meeting.", "We want to go too.", "This is for you.", "They won the account.",
    "Let's queue it up for later.", "I'd like to see the EBIT bridge.", "The tin was empty.",
    "Wait for the signal.", "Fore! Watch out!", "He was too late to fix it.", "Who won? We did.",
    "I'm on it.", "Eat first, then we talk.", "Before or after lunch?", "To be honest, it's fine.",
    "The weight is the issue.", "Our net margin improved.", "Tutor the new hires.", "The freight arrived late.",
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


def dev_carriers(carriers: list[str], dev_frac: float) -> set[str]:
    """Held-out carriers: the hash split, then adjusted so every slot kind has at least one
    carrier in train and one in dev (a pure hash split left some kinds untrained or unscored)."""
    dev = {c for c in carriers if is_dev_template(c, dev_frac)}
    kinds = sorted({k for c in carriers for k in SLOT.findall(c)})
    crc = lambda c: zlib.crc32(c.encode())  # noqa: E731
    for k in kinds:
        with_k = sorted((c for c in carriers if "{" + k + "}" in c), key=crc)
        if len(with_k) < 2:
            raise ValueError(f"slot kind {k!r} needs at least two carriers")
        if all(c in dev for c in with_k):
            dev.discard(with_k[0])
        if not any(c in dev for c in with_k):
            dev.add(next(c for c in with_k if sum(c2 not in dev for c2 in with_k) > 1))
    return dev


GENERIC_KINDS = ["int_small", "int_tens", "int_hundreds", "int_4digit", "year", "big", "decimal", "money",
                 "percent", "clock", "date", "ordinal", "range", "fraction", "measure", "quarter", "and_acronym",
                 "title_name", "roman", "acronym", "identifier", "version", "code", "alnum", "multiplier",
                 "title_name", "and_acronym", "title_name", "and_acronym"]  # weighted: v3 under-learned these


def fill(carrier: str, r: random.Random):
    carrier = carrier.replace("{any}", "{" + r.choice(GENERIC_KINDS) + "}")
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


# Spoken lead-ins, so carriers do not always start the same way.
# No "Um,"/"Uh,": the synthesizer voices them as "Ah", so the label would not match the audio.
LEAD_INS = ["So ", "Okay, so ", "And ", "Yeah, ", "I think ", "Well, ", "Right, so ", "Honestly, ",
            "Basically, ", "Look, ", "Now, "]


# A short sentence before the carrier, so the number OPENS a second sentence and keeps its capital
# ("Okay. Two point seven five ..."). Without these no label had a number word right after a full stop,
# and the fine-tune wrote "Okay. two point seven five" (2026-10-09 census, both arms).
SENTENCE_LEAD_INS = ["Okay. ", "Right. ", "Sure. ", "Got it. ", "Yes. ", "Thanks. ", "Good. "]


def lead_in(text: str, r: random.Random, p: float = 0.3) -> str:
    if r.random() >= p:
        return text
    pick = r.choice(LEAD_INS + SENTENCE_LEAD_INS)
    if pick.endswith(". "):
        return pick + text
    if text[:2].isupper():
        return text
    return pick + text[0].lower() + text[1:]


def rows(n: int, seed: int, dev_frac: float = 0.2, control_frac: float = 0.1, multi_frac: float = 0.0):
    """`multi_frac` of the rows join two or three sentences, so sentence boundaries occur inside a
    clip (single-sentence clips taught the model that every clip ends in one period)."""
    single = _rows(n, seed, dev_frac, control_frac)
    r = random.Random(seed + 7)
    for row in single:
        if r.random() >= multi_frac:
            yield row
            continue
        parts = [row] + [x for x, _ in zip(single, range(r.choice([1, 1, 2])))]
        parts = [p for p in parts if p["split"] == row["split"]] or [row]
        yield {"id": row["id"], "split": row["split"], "template": " || ".join(p["template"] for p in parts),
               "kinds": [k for p in parts for k in p["kinds"]], "text": " ".join(p["text"] for p in parts),
               "tts_text": " ".join(p["tts_text"] for p in parts)}


def _rows(n: int, seed: int, dev_frac: float, control_frac: float):
    r = random.Random(seed)
    carriers = load_carriers()
    dev = dev_carriers(carriers, dev_frac)
    for i in range(n):
        if r.random() < control_frac:
            text = r.choice(CONTROLS)
            yield {"id": f"r{seed}_{i}", "split": "dev" if is_dev_template(text, dev_frac) else "train",
                   "template": text, "kinds": ["control"], "text": text, "tts_text": text}
            continue
        carrier = r.choice(carriers)
        target, tts, kinds = fill(carrier, r)
        state = r.getstate()
        target = lead_in(target, r)
        r.setstate(state)
        tts = lead_in(tts, r)
        yield {"id": f"r{seed}_{i}", "split": "dev" if carrier in dev else "train",
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
