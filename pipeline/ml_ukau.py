"""UK / Australian spoken number forms for an English synthetic set (2026-10-17): phone numbers read with "oh" and
"double", "nought", clock times in words ("half past three", "twenty to six"), "the fifth of May" / "May the fifth",
and pounds-and-pence / dollars-and-cents amounts ("four pounds fifty", "fifty p"). Labels are the spoken words exactly
as voiced; each reading carries the value it means (checked against number_outcomes in the tests).
"""
import random
import zlib

import readings as R

D = R.ONES[:10]
FORMS = ("phone", "nought", "clock", "date", "money")
CARRIERS = {
    "phone": ["My number's {x}, ring me this evening.", "You can reach the office on {x}.", "The landline is {x}.",
              "Give them a bell on {x}.", "Her mobile is {x}, I think.", "Call {x} and ask for the duty manager.",
              "The booking line is {x}.", "Try {x} if the other line's engaged.", "It's {x}, write that down.",
              "Leave a message on {x}.", "The surgery number is {x}.", "Text me on {x} when you're outside."],
    "nought": ["The margin came in at {x}.", "Rates are sitting at {x}.", "It reads {x} on the gauge.",
               "We measured {x} on the second run.", "The error was only {x}.", "Inflation ticked up to {x}.",
               "Set the tolerance to {x}.", "The tablet dose is {x}.", "They scored {x} in the last round.",
               "The gap is about {x}."],
    "clock": ["The train leaves at {x}.", "Let's meet at {x}.", "I'll pick you up at {x}.", "The shop shuts at {x}.",
              "We kicked off at {x}.", "Doors open at {x}.", "She rang me at {x}.", "Lunch is at {x} today.",
              "The bus comes at {x}.", "It's {x} already.", "I woke up at {x}.", "The match starts at {x}."],
    "date": ["The party's on {x}.", "We fly out on {x}.", "Her birthday is {x}.", "Rent is due on {x}.",
             "The course starts on {x}.", "I'm off from {x}.", "The meeting moved to {x}.", "School goes back on {x}."],
    "money": ["That came to {x}.", "It was {x} at the market.", "A coffee here is {x}.", "The ticket cost {x}.",
              "He charged me {x} for that.", "It's on sale for {x}.", "Postage was {x}.", "I paid {x} for the lot.",
              "The fare is {x} each way.", "Parking was {x}."],
}


def _phone(r):
    """(reading, digit string): UK 0xxxx / AU 04xx numbers read digit by digit, 'oh' for zero, 'double' for pairs."""
    au = r.random() < 0.5
    digits = ("04" + "".join(str(r.randint(0, 9)) for _ in range(8))) if au else (
        "0" + r.choice(["20", "161", "113", "7"]) + "".join(str(r.randint(0, 9)) for _ in range(7)))
    # make a double likely: copy one digit onto its neighbour
    i = r.randint(2, len(digits) - 2)
    digits = digits[:i] + digits[i - 1] + digits[i + 1:]
    groups = [digits[:4], digits[4:7], digits[7:]] if au else [digits[:len(digits) - 7], digits[-7:-4], digits[-4:]]
    out = []
    for g in groups:
        j = 0
        ws = []
        while j < len(g):
            w = "oh" if g[j] == "0" else D[int(g[j])]
            if j + 1 < len(g) and g[j + 1] == g[j]:
                ws.append(f"double {w}")
                j += 2
            else:
                ws.append(w)
                j += 1
        out.append(" ".join(ws))
    return ", ".join(out), digits


def _nought(r):
    k = r.random()
    if k < 0.5:
        f = r.choice(["five", "two", "two five", "seven five", "three", "oh five", "one"])
        v = "0." + "".join(str(D.index(w)) if w != "oh" else "0" for w in f.split())
        return f"nought point {f}" + (" percent" if r.random() < 0.4 else ""), v
    n, m = r.randint(1, 9), r.randint(1, 9)
    if k < 0.75:
        return f"{D[n]} point nought {D[m]}", f"{n}.0{m}"
    return f"nought point nought {D[n]}", f"0.0{n}"


def _clock(r):
    h = r.randint(1, 12)
    nxt = h % 12 + 1
    form = r.choice(["half past", "quarter past", "quarter to", "ten past", "twenty past", "twenty to", "five to",
                     "ten to", "twenty five past"])
    hh = nxt if " to" in form else h
    val = {"half past": (h, 30), "quarter past": (h, 15), "quarter to": (h, 45), "ten past": (h, 10),
           "twenty past": (h, 20), "twenty to": (h, 40), "five to": (h, 55), "ten to": (h, 50),
           "twenty five past": (h, 25)}[form]
    pre = "a " if form.startswith("quarter") and r.random() < 0.3 else ""
    return f"{pre}{form} {R.cardinal(hh)}", f"{val[0]}:{val[1]:02d}"


def _date(r):
    m = r.choice(R.MONTHS)
    d = r.randint(1, 28)
    reading = f"the {R.ordinal(d)} of {m}" if r.random() < 0.6 else f"{m} the {R.ordinal(d)}"
    return reading, f"{d} {m}"


def _money(r):
    k = r.random()
    if k < 0.45:
        p, c = r.randint(1, 20), r.choice([5, 10, 20, 25, 49, 50, 75, 80, 95, 99])
        return f"{R.cardinal(p)} {'pound' if p == 1 else 'pounds'} {R.cardinal(c)}", f"£{p}.{c:02d}"
    if k < 0.7:
        c = r.choice([20, 30, 40, 50, 60, 75, 80, 90, 99])
        return f"{R.cardinal(c)} p", f"{c}p"
    d, c = r.randint(2, 60), r.choice([10, 20, 25, 50, 75, 95])
    return f"{R.cardinal(d)} dollars {R.cardinal(c)}", f"${d}.{c:02d}"


GEN = {"phone": _phone, "nought": _nought, "clock": _clock, "date": _date, "money": _money}


def rows(per_form: dict, seed: int, dev_frac: float = 0.2) -> list:
    """per_form {form: n} -> gen.py-shaped rows (template keeps the {slot}, kinds = ["ukau_<form>"], value = written form)."""
    r = random.Random(f"ukau|{seed}")
    out, seen = [], set()
    for form in FORMS:
        n, tries = int(per_form.get(form, 0)), 0
        while sum(1 for x in out if x["cls"] == form) < n and tries < 50 * n + 50:
            tries += 1
            carrier = r.choice(CARRIERS[form])
            reading, value = GEN[form](r)
            text = carrier.replace("{x}", reading)
            if carrier.startswith("{x}"):
                text = text[0].upper() + text[1:]
            if text in seen:
                continue
            seen.add(text)
            tts = text.replace(" p.", " pee.").replace(" p ", " pee ")
            split = "dev" if (zlib.crc32(carrier.encode()) % 1000) / 1000 < dev_frac else "train"
            out.append({"id": f"u{seed}_{form}_{len(out)}", "split": split, "template": carrier.replace("{x}", "{ukau_" + form + "}"),
                        "kinds": ["ukau_" + form], "text": text, "tts_text": tts, "value": value, "cls": form})
    return out
