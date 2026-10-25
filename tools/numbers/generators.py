"""Number-reading brief generator for synthetic spoken-number speech data.

A program fixes every number (value, exact spoken reading, training label, kind, repeat structure, position) and the context
(setting, register, form, name); an LLM then writes only the carrier sentence around the fixed readings (see sentences.py).
Coverage comes from an explicit spec (SPEC below) so a census can verify every cell before any synthesis is paid for.
Labels: digit strings -> digits without dashes; spelled alphanumerics -> letters+digits; single-token letter-digit codes
(3D, B12, W-2) -> code; every other number -> the spoken words.
  python3 generators.py --tag v3 [--scale 1.0] [--root DIR]   -> DIR/briefs/<tag>.jsonl + DIR/spec_<tag>.json
"""
import argparse, collections, hashlib, json, math, os, random, re, types

# ---------------------------------------------------------------- reading helpers
U = "zero one two three four five six seven eight nine".split()
T = "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TY = "twenty thirty forty fifty sixty seventy eighty ninety".split()


def w99(n):
    if n < 10: return U[n]
    if n < 20: return T[n - 10]
    return TY[n // 10 - 2] + ("" if n % 10 == 0 else " " + U[n % 10])


def w999(n):
    if n < 100: return w99(n)
    return U[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + w99(n % 100))


def card(n):
    if n == 0: return "zero"
    out = []
    for s, name in ((10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        if n >= s: out.append(w999(n // s) + " " + name); n %= s
    if n: out.append(w999(n))
    return " ".join(out)


def year_reading(y, R):
    if 2000 <= y <= 2009: return R.choice([f"two thousand {U[y - 2000]}" if y > 2000 else "two thousand", f"two thousand and {U[y - 2000]}" if y > 2000 else "two thousand", f"twenty oh {U[y - 2000]}" if y > 2000 else "two thousand"])
    if 2010 <= y <= 2099: return R.choice([f"twenty {w99(y - 2000)}", f"two thousand {w99(y - 2000)}"])
    hi, lo = divmod(y, 100)
    if lo == 0: return f"{w99(hi)} hundred"
    if lo < 10: return f"{w99(hi)} oh {U[lo]}"
    return f"{w99(hi)} {w99(lo)}"


def digits_reading(ds, R, style):
    """digit-by-digit reading with optional double/triple, oh/zero, grouped pairs"""
    out, i = [], 0
    while i < len(ds):
        d = ds[i]
        z = R.choice(["oh", "zero"]) if style.get("oh", 0.5) > R.random() else "zero"
        if i + 2 < len(ds) and ds[i] == ds[i + 1] == ds[i + 2] and R.random() < style.get("triple", 0.4):
            out.append("triple " + (z if d == "0" else U[int(d)])); i += 3; continue
        if i + 1 < len(ds) and ds[i] == ds[i + 1] and R.random() < style.get("double", 0.35):
            out.append("double " + (z if d == "0" else U[int(d)])); i += 2; continue
        out.append(z if d == "0" else U[int(d)]); i += 1
    return " ".join(out)


def rep_digits(R, n):
    """digit string of length n with forced repeats (pairs and sometimes a triple)"""
    ds = [str(R.randint(0, 9)) for _ in range(n)]
    for _ in range(R.randint(1, max(1, n // 4))):
        k = R.randint(0, n - 2); ds[k + 1] = ds[k]
        if R.random() < 0.25 and k + 2 < n: ds[k + 2] = ds[k]
    return "".join(ds)


def grouped(ds, R, groups):
    """read groups with a pause (comma) between; returns (reading, label_with_dashes)"""
    parts, lab, i = [], [], 0
    for g in groups:
        seg = ds[i:i + g]; i += g
        if not seg: break
        if len(seg) == 2 and seg[0] != "0" and R.random() < 0.3: parts.append(w99(int(seg)))
        else: parts.append(digits_reading(seg, R, {}))
        lab.append(seg)
    return ", ".join(parts), "-".join(lab)


# ------------------------------------------------------------------ cells: each returns (sentence_words, label_v3, kind, vkeys)


CODES = [("three D", "3D"), ("two D", "2D"), ("four K", "4K"), ("five G", "5G"), ("four G", "4G"), ("Q three", "Q3"), ("Q four", "Q4"),
         ("B two B", "B2B"), ("B two C", "B2C"), ("COVID nineteen", "COVID-19"), ("A ten", "A-10"), ("F thirty five", "F-35"),
         ("K twelve", "K-12"), ("MP three", "MP3"), ("CO two", "CO2"), ("H two O", "H2O"), ("four A", "4A"), ("P S five", "PS5"),
         ("Web three", "Web3"), ("V eight", "V8"), ("Core i seven", "Core i7"), ("R two D two", "R2-D2"), ("G seven", "G7"), ("HTML five", "HTML5")]
NUMW = set("zero oh one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion point double triple and first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth half quarter".split())

P = types.SimpleNamespace(U=U, T=T, TY=TY, w99=w99, w999=w999, card=card, year_reading=year_reading, digits_reading=digits_reading,
                          rep_digits=rep_digits, grouped=grouped, CODES=CODES, NUMW=NUMW)

# ---------------------------------------------------------------- context pools and v2 fixed generators (ordinary2 etc.)
SETTINGS = ["customer support call", "team standup", "quarterly sales review", "doctor's office", "pharmacy counter", "shipping and returns",
            "bank phone line", "airport gate", "hotel front desk", "classroom lecture", "history podcast", "sports radio", "IT help desk",
            "law office", "real estate showing", "science lab notes", "car dealership", "restaurant order", "insurance claim",
            "school admin office", "construction site", "warehouse inventory", "startup pitch", "board meeting", "tax appointment",
            "veterinary clinic", "train station", "radio traffic report", "cooking show", "fitness class", "game stream", "city council",
            "product demo", "engineering design review", "recruiting call", "wedding planning", "museum tour", "news briefing",
            "reading a shared screen in a video call", "voicemail"]
REGISTERS = ["casual", "work-casual", "formal", "reading aloud", "dictating slowly", "phone agent script", "rushed", "tired", "cheerful"]
FORMS = ["statement", "question", "instruction", "list of items", "interrupted sentence", "answer to an unheard question",
         "two short sentences", "aside in brackets", "confirming by repeating", "data entry (spelling things out)", "announcement"]
LENGTHS = [("very short", "3-6 words"), ("short", "7-12 words"), ("medium", "13-22 words"), ("long", "23-40 words")]
POSITIONS = ["first words of the sentence", "early", "middle", "late", "last words of the sentence"]
CODES = P.CODES


def digit_item(R, control=False):
    kind = R.choice(["phone number", "ISBN", "account number", "case number", "order number", "tracking number", "serial number", "member ID", "confirmation code"])
    n = {"phone number": R.choice([7, 10, 11]), "ISBN": R.choice([10, 13])}.get(kind, R.randint(6, 12))
    ds = "".join(str(R.randint(0, 9)) for _ in range(n)) if control else P.rep_digits(R, n)
    if kind == "ISBN" and n == 13: ds = "978" + ds[3:]
    groups = R.choice([[n], [3, 3, n - 6] if n > 6 else [n], [2] * (n // 2) + ([1] if n % 2 else []), [4, 4, n - 8] if n > 8 else [n], [3, n - 3]])
    reading, lab = P.grouped(ds, R, [g for g in groups if g > 0])
    lab = lab if R.random() < 0.5 else lab.replace("-", "")
    return dict(kind=kind, reading=reading, label=lab, value=ds, repeat=not control)


def short_id(R):
    ds = P.rep_digits(R, R.choice([3, 4])); r = P.digits_reading(ds, R, {"double": 0.3, "triple": 0.2})
    return dict(kind=R.choice(["room", "gate", "extension", "flight", "suite", "bus route", "apartment unit", "platform", "PIN hint", "form"]) + " number (an ID, not a quantity)", reading=r, label=r, value=ds, repeat=True)


def repeat_words(R):
    k = R.choice(["year", "clock", "price", "reference code", "score"])
    if k == "year": y = R.choice([2020, 2121, 1919, 1818, 1717]); r = f"{P.w99(y // 100)} {P.w99(y % 100)}"; kind = "a year"
    elif k == "clock": h = R.randint(1, 12); r = f"{P.w99(h)} {P.w99(h)}" if h > 9 else f"{P.U[h]} oh {P.U[h]}"; kind = "a time of day"
    elif k == "price": a = R.choice([9, 19, 29, 49, 99]); r = f"{P.w99(a)} {P.w99(a)}"; kind = "a price in dollars and cents said without the words dollars or cents"
    elif k == "reference code": a = R.randint(11, 99); r = f"{P.w99(a)} {P.w99(a)}"; kind = "a reference or room code"
    else: a = R.randint(1, 9); r = f"{P.U[a]} {P.U[a]}"; kind = "a tied game score"
    return dict(kind=kind, reading=r, label=r, value=None, repeat=True)


def _core(s):  # pull the number words out of a c_R12 sentence
    import re
    m = re.findall(r"(?:\b(?:" + "|".join(sorted(P.NUMW, key=len, reverse=True)) + r")\b[ ,]*)+", s)
    return max(m, key=len).strip(" ,") if m else s


def ordinary(R):
    k = R.choice(["year", "amount", "count", "decimal", "percent", "range", "clock", "date", "score"])
    if k == "year": y = R.randint(1800, 2060); r = P.year_reading(y, R)
    elif k == "amount": r = R.choice([P.card(R.randint(2, 999999)) + " dollars", P.w99(R.randint(1, 99)) + " point " + P.U[R.randint(1, 9)] + " million", P.w99(R.randint(11, 99)) + " hundred bucks"])
    elif k == "count": r = P.card(R.randint(2, 5000))
    elif k == "decimal": r = P.w99(R.randint(0, 99)) + " point " + " ".join(P.U[R.randint(0, 9)] for _ in range(R.randint(1, 2)))
    elif k == "percent": r = P.w99(R.randint(1, 99)) + " percent"
    elif k == "range": return rng(R)
    elif k == "clock": r = f"{P.w99(R.randint(1, 12))} {R.choice(['fifteen', 'thirty', 'forty five', 'oh five', ''])}".strip() + R.choice([" a m", " p m", "", " in the morning"])
    elif k == "date": r = R.choice(["January", "March", "June", "September", "November"]) + " " + R.choice(["first", "third", "twelfth", "twenty second", "thirtieth"])
    else: a, b = R.randint(10, 45), R.randint(1, 40); r = f"{P.w99(max(a, b))} {R.choice(['to ', ''])}{P.w99(min(a, b))}"; k = "game score"
    kind = {"year": "a year", "amount": "an amount of money", "count": "a count of things", "decimal": "a measurement with a decimal", "percent": "a percentage",
            "clock": "a time of day", "date": "a date"}.get(k, k)
    return dict(kind=kind, reading=r, label=r, value=None, repeat=False)


def rng(R):
    k = R.choice(["years_short", "years_full", "numbers", "pages"])
    if k == "years_short":
        y = R.randint(1900, 2023)
        while (y + 1) % 100 == 0: y = R.randint(1900, 2023)
        r = f"{P.year_reading(y, R)} {R.choice(['to', 'through', ''])} {P.w99((y + 1) % 100) if (y + 1) % 100 >= 10 else 'oh ' + P.U[(y + 1) % 100]}".replace("  ", " "); kind = "a span of years (season or fiscal year)"
    elif k == "years_full": y = R.randint(1900, 2020); r = f"{P.year_reading(y, R)} to {P.year_reading(y + R.randint(2, 30), R)}"; kind = "a span of years"
    elif k == "numbers": a = R.randint(2, 60); r = f"{P.w99(a)} to {P.w99(min(99, a + R.randint(2, 30)))}"; kind = "a range of quantities"
    else: a = R.randint(10, 400); r = f"{P.card(a)} to {P.card(a + R.randint(2, 40))}"; kind = "a page range"
    return dict(kind=kind, reading=r, label=r, value=None, repeat=False)


def code(R):
    sp, lab = R.choice(CODES); return dict(kind="product or code name", reading=sp, label=lab, value=lab, repeat=False)


CELLS = {  # name: (count, builder of the main number list, extra instruction)
    "digits_repeat": (4500, lambda R: [digit_item(R)], ""),
    "digits_control": (1500, lambda R: [digit_item(R, control=True)], ""),
    "short_id": (2500, lambda R: [short_id(R)], ""),
    "repeat_words": (1500, lambda R: [repeat_words(R)], ""),
    "clip_start": (3500, lambda R: [ordinary(R)] + ([ordinary(R)] if R.random() < 0.4 else []), "The FIRST number must be the very first words of the sentence (no lead-in word)."),
    "pairs": (1500, lambda R: [rng(R)], ""),
    "codes": (2000, lambda R: [code(R)] + ([ordinary(R)] if R.random() < 0.3 else []), "Use a nearby ordinary word that sounds like part of the code when natural (free/three, for/four, tan/ten)."),
    "soundalike": (1000, lambda R: [dict(kind="small count", reading=R.choice(["two", "four", "one", "eight"]), label=None, value=None, repeat=False)], "Next to the number, naturally use one of the sound-alike words to, too, for, won, ate in its normal meaning (for example: went to two, for four days)."),
    "filler": (1000, lambda R: [ordinary(R)], "Put a hesitation (um or uh) or a self-correction INSIDE or right after the number, and keep it in the sentence."),
    "no_number": (2200, lambda R: [], "Use NO numbers at all. Naturally use ONE or TWO of these words in their normal meaning: to, too, for, won, ate, free, tan, fore, tree, tin."),
    "window_src": (3000, lambda R: [digit_item(R) if R.random() < 0.4 else ordinary(R), ordinary(R)], "Write 2 to 3 sentences that flow together, about 25-45 words in total."),
}


# ---------------------------------------------------------------- supplement (census v2 fixes, 2026-10-24); v2 builders above are unchanged
MONTHS = "January February March April May June July August September October November December".split()
DAYS_IN = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
WEEKDAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
ORDW = "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth".split()


def ord_word(d):
    if d < 20: return ORDW[d - 1]
    if d % 10 == 0: return {20: "twentieth", 30: "thirtieth"}[d]
    return P.TY[d // 10 - 2] + " " + ORDW[d % 10 - 1]


def logu(R, lo, hi): return int(round(10 ** R.uniform(math.log10(lo), math.log10(hi))))


def year2(R):
    x = R.random()
    y = R.randint(2010, 2035) if x < 0.45 else R.randint(2036, 2075) if x < 0.57 else R.randint(1950, 2009) if x < 0.80 else R.randint(1800, 1949) if x < 0.95 else R.randint(1500, 1799)
    return P.year_reading(y, R)


def mins(m): return "oh " + P.U[m] if m < 10 else P.w99(m)


PERIOD_HOURS = {" in the morning": set(range(5, 12)), " in the afternoon": {12, 1, 2, 3, 4, 5}, " in the evening": {5, 6, 7, 8, 9}, " at night": {9, 10, 11, 12, 1, 2, 3}}


def period_ok(h12, opts):
    return [o for o in opts if o not in PERIOD_HOURS or h12 in PERIOD_HOURS[o]]


def clock2(R):
    h12, m = R.randint(1, 12), R.choice([R.randint(1, 59), R.randint(1, 59), R.choice([0, 15, 30, 45])])
    x = R.random()
    if x < 0.40: return f"{P.w99(h12)}{'' if m == 0 else ' ' + mins(m)}" + R.choice(period_ok(h12, [" A M", " P M", " in the morning", " in the evening", " in the afternoon", " at night", ""]))
    if x < 0.52: return f"{P.w99(h12)} o'clock" + R.choice(period_ok(h12, ["", " in the morning", " at night", " in the evening", " in the afternoon"]))
    if x < 0.64: return R.choice([f"quarter past {P.w99(h12)}", f"quarter to {P.w99(h12)}", f"half past {P.w99(h12)}",
                                  f"{P.w99(R.choice([5, 10, 20, 25]))} past {P.w99(h12)}", f"{P.w99(R.choice([5, 10, 20, 25]))} to {P.w99(h12)}"])
    if x < 0.70: return R.choice(["noon", "midnight", "twelve noon"])
    h24 = R.randint(0, 23)
    hw = ("oh " + P.U[h24]) if h24 < 10 else P.w99(h24)
    return f"{hw} hundred" + R.choice(["", " hours"]) if m == 0 else f"{hw} {mins(m)}"


def date2(R):
    mi = R.randrange(12); d = R.randint(1, DAYS_IN[mi]); M, O = MONTHS[mi], ord_word(d)
    base = f"the {O} of {M}" if R.random() < 0.25 else f"{M} {O}"
    x = R.random()
    if x < 0.30: base += " " + year2(R)
    elif x < 0.45: base = R.choice(WEEKDAYS) + " " + base
    elif x < 0.52: base = f"the {O}"
    return base


def money2(R):
    x = R.random()
    if x < 0.38:
        d, c = R.randint(0, 99), R.randint(1, 99)
        if d == 0: return f"{P.w99(c)} cents"
        if d == 1 and R.random() < 0.3: return f"a buck {P.w99(c)}"
        return R.choice([f"{P.w99(d)} dollars and {P.w99(c)} cents", f"{P.w99(d)} dollars and {P.w99(c)} cents", f"{P.w99(d)} {mins(c)}",
                         f"{P.w99(d)} dollars {P.w99(c)}" if c >= 10 else f"{P.w99(d)} dollars and {P.w99(c)} cents"])
    if x < 0.55: n = logu(R, 1, 999); return f"{P.card(n)} {R.choice(['dollars', 'dollars', 'bucks'])}"
    if x < 0.70:
        n = logu(R, 1000, 99999)
        if n < 10000 and R.random() < 0.3:
            h = n // 100
            if h % 10: return f"{P.w99(h)} hundred bucks"
        return f"{P.card(n)} dollars"
    if x < 0.85: return f"{P.card(logu(R, 100000, 999999))} dollars"
    unit = R.choice([" dollars", " dollars", " bucks", ""])  # mostly say the currency so the writer cannot turn it into an item count
    return R.choice([f"{P.w99(R.randint(1, 99))} point {P.U[R.randint(1, 9)]} million{unit}", f"{P.card(R.randint(1, 50))} million dollars",
                     f"{P.U[R.randint(1, 9)]} point {P.U[R.randint(0, 9)]} {P.U[R.randint(1, 9)]} billion{unit}"])


def decimal2(R):
    ds = [R.randint(0, 9) for _ in range(R.choice([1, 1, 2, 2, 3]) - 1)] + [R.randint(1, 9)]  # no trailing zero
    frac = " ".join(R.choice(["zero", "oh"]) if d == 0 else P.U[d] for d in ds)
    x = R.random()
    if x < 0.15: return "point " + frac
    if x < 0.25: return "zero point " + frac
    if x < 0.80: return f"{P.w99(R.randint(1, 99))} point {frac}"
    return f"{P.card(logu(R, 100, 99999))} point {frac}"


def percent2(R):
    x = R.random()
    if x < 0.55: return f"{P.w99(R.randint(1, 99))} percent"
    if x < 0.85: return f"{P.w99(R.randint(0, 99))} point {P.U[R.randint(1, 9)]} percent" if R.random() < 0.7 else f"point {P.U[R.randint(1, 9)]} percent"
    return f"{P.card(100 if R.random() < 0.2 else R.randint(101, 400))} percent"


def span2(R):
    y = R.randint(2000, 2030) if R.random() < 0.6 else R.randint(1950, 1999)
    if R.random() < 0.5:
        while (y + 1) % 100 == 0: y += 1
        nx = (y + 1) % 100
        return "a span of years (season or fiscal year)", f"{P.year_reading(y, R)} {R.choice(['to', 'through', ''])} {P.w99(nx) if nx >= 10 else 'oh ' + P.U[nx]}".replace("  ", " ")
    return "a span of years", f"{P.year_reading(y, R)} to {P.year_reading(y + R.randint(1, 12), R)}"


CODES2 = list(CODES) + [
    ("W two", "W-2"), ("I nine", "I-9"), ("W four", "W-4"), ("ten ninety nine", "1099"), ("four oh one K", "401(k)"), ("five twenty nine plan", "529 plan"),
    ("A four", "A4"), ("B twelve", "B12"), ("vitamin D three", "vitamin D3"), ("omega three", "omega-3"), ("H one B", "H-1B"), ("N ninety five", "N95"),
    ("iPhone fifteen", "iPhone 15"), ("iPhone sixteen", "iPhone 16"), ("Pixel eight", "Pixel 8"), ("Galaxy S twenty four", "Galaxy S24"),
    ("Windows eleven", "Windows 11"), ("Windows ten", "Windows 10"), ("Python three", "Python 3"), ("Java seventeen", "Java 17"), ("iOS eighteen", "iOS 18"),
    ("PS four", "PS4"), ("Xbox three sixty", "Xbox 360"), ("Boeing seven thirty seven", "Boeing 737"), ("Boeing seven eight seven", "Boeing 787"),
    ("Airbus A three twenty", "Airbus A320"), ("Route sixty six", "Route 66"), ("I four oh five", "I-405"), ("Interstate ninety five", "Interstate 95"),
    ("Highway one oh one", "Highway 101"), ("Model three", "Model 3"), ("RTX forty ninety", "RTX 4090"), ("WD forty", "WD-40"), ("Formula one", "Formula 1"),
    ("Area fifty one", "Area 51"), ("Apollo eleven", "Apollo 11"), ("Catch twenty two", "Catch-22"), ("Studio fifty four", "Studio 54"),
    ("Section eight", "Section 8"), ("Title nine", "Title IX"), ("Chapter eleven", "Chapter 11"), ("Chapter seven", "Chapter 7"), ("Level three", "Level 3"),
    ("USB three", "USB 3"), ("Wi-Fi six", "Wi-Fi 6"), ("HDMI two", "HDMI 2"), ("IPv six", "IPv6"), ("IPv four", "IPv4"), ("Web two", "Web 2"),
    ("M three", "M3"), ("Ryzen seven", "Ryzen 7"), ("Ryzen nine", "Ryzen 9"), ("Core i nine", "Core i9"), ("Core i five", "Core i5"),
    ("GPT four", "GPT-4"), ("Halo three", "Halo 3"), ("Fallout four", "Fallout 4"), ("Mario Kart eight", "Mario Kart 8"), ("Final Fantasy seven", "Final Fantasy VII"),
    ("Top forty", "Top 40"), ("Fortune five hundred", "Fortune 500"), ("S and P five hundred", "S&P 500"), ("Russell two thousand", "Russell 2000"),
    ("Big Ten", "Big Ten"), ("Gate B twenty two", "Gate B22"), ("Pier thirty nine", "Pier 39"), ("Track nine", "Track 9"), ("Zone three", "Zone 3"),
    ("nine one one", "911"), ("three one one", "311"), ("twenty four seven", "24/7"), ("Office three sixty five", "Office 365"), ("seven eleven", "7-Eleven"),
    ("Forever twenty one", "Forever 21"), ("Blink one eighty two", "Blink-182"), ("Windows ninety five", "Windows 95"), ("C three PO", "C-3PO"),
    ("V twelve", "V12"), ("T one hundred", "T-100"), ("F one fifty", "F-150"), ("E coli O one fifty seven", "E. coli O157"), ("A one C", "A1C"),
    ("Q one", "Q1"), ("Q two", "Q2"), ("H one", "H1"), ("H two", "H2"), ("FY twenty five", "FY25"), ("FY twenty six", "FY26"), ("Gen Z", "Gen Z"),
    ("K nine", "K-9"), ("M sixteen", "M16"), ("AK forty seven", "AK-47"), ("B fifty two", "B-52"), ("U two", "U2"), ("Matchbox twenty", "Matchbox 20"),
    ("Maroon five", "Maroon 5"), ("Jackson five", "Jackson 5"), ("Ocean's eleven", "Ocean's Eleven"), ("Agent forty seven", "Agent 47"), ("Lot seven", "Lot 7"),
    ("Room one oh one", "Room 101"), ("Studio B", "Studio B"), ("SPF fifty", "SPF 50"), ("five K run", "5K run"), ("ten K", "10K"), ("eight K", "8K"),
    ("seventy two hundred RPM", "7200 RPM"), ("A five", "A5"), ("B two", "B2"), ("C sharp", "C#"), ("P one", "P1"), ("P zero", "P0"), ("S three", "S3"),
    ("EC two", "EC2"), ("L two", "L2"), ("L one", "L1"), ("T three", "T3"), ("X two", "X2"), ("Z ten", "Z10"), ("G four", "G4"), ("DDR five", "DDR5"),
    ("LTE", "LTE"), ("H two sixty four", "H.264"), ("MP four", "MP4"), ("PDF A", "PDF/A"), ("ISO nine thousand one", "ISO 9001"), ("SOC two", "SOC 2"),
    ("PCI DSS four", "PCI DSS 4"), ("TLS one point three", "TLS 1.3"), ("HTTP two", "HTTP/2"), ("OAuth two", "OAuth 2"), ("ES six", "ES6"), ("Vue three", "Vue 3"),
]
CODES2 = [(s, l) for s, l in CODES2 if any(c.isdigit() for c in l)]  # codes must contain a number


def code2(R):
    sp, lab = R.choice(CODES2); return dict(kind="product or code name", reading=sp, label=lab, value=lab, repeat=False)


def ordinary2(R):
    k = R.choices(["year", "amount", "count", "decimal", "percent", "span", "clock", "date", "score", "range", "rep_price"],
                  weights=[13, 15, 10, 10, 8, 8, 13, 12, 4, 4, 3])[0]
    if k == "span": kind, r = span2(R); return dict(kind=kind, reading=r, label=r, value=None, repeat=False)
    if k == "rep_price":  # no multiples of ten: "twenty twenty" reads as a year
        a = R.choice([v for v in range(11, 100) if v % 10]); r = f"{P.w99(a)} {P.w99(a)}"
        return dict(kind="a price in dollars and cents said without the words dollars or cents", reading=r, label=r, value=None, repeat=True)
    if k == "range":
        a = logu(R, 2, 5000); r = f"{P.card(a)} to {P.card(a + max(1, int(a * R.uniform(0.1, 1.5))))}"
        return dict(kind=R.choice(["a range of quantities", "a page range"]), reading=r, label=r, value=None, repeat=False)
    if k == "score":
        a, b = R.randint(0, 150), R.randint(0, 150); r = f"{P.card(max(a, b))} {R.choice(['to ', ''])}{P.card(min(a, b))}"
        return dict(kind="game score", reading=r, label=r, value=None, repeat=False)
    r = {"year": year2, "amount": money2, "decimal": decimal2, "percent": percent2, "clock": clock2, "date": date2, "count": lambda R: P.card(logu(R, 1, 999999))}[k](R)
    kind = {"year": "a year", "amount": "an amount of money", "count": "a count of things", "decimal": "a measurement with a decimal",
            "percent": "a percentage", "clock": "a time of day", "date": "a date"}[k]
    lab = r.replace(" A M", " AM").replace(" P M", " PM") if k == "clock" else r
    return dict(kind=kind, reading=r, label=lab, value=None, repeat=False)


def _win_nums(R):
    out = []
    for _ in range(R.choice([2, 2, 3])):
        x = R.random(); out.append(digit_item(R) if x < 0.15 else code2(R) if x < 0.22 else ordinary2(R))
    return out


SUP_CELLS = {
    "clip_start": (1200, lambda R: [ordinary2(R)] + ([ordinary2(R)] if R.random() < 0.3 else []), "The FIRST number must be the very first words of the sentence (no lead-in word)."),
    "clip_end": (1000, lambda R: ([ordinary2(R)] if R.random() < 0.3 else []) + [ordinary2(R)], "The LAST number must be the very last words of the sentence (nothing after it but punctuation)."),
    "window_src": (1200, _win_nums, "Write 2 to 3 sentences that flow together, about 25-45 words in total."),
    "filler": (600, lambda R: [ordinary2(R)], "Put a hesitation (um or uh) or a self-correction INSIDE or right after the number, and keep it in the sentence."),
}



# ---------------------------------------------------------------- name / place pools
FIRST = ("Aisha Akira Alejandro Amara Ananya Andrei Anika Arjun Astrid Beatriz Bilal Bongani Camila Chen Chidi Chloe Darius Deepa Diego Dmitri "
         "Elif Emeka Esther Farah Fatima Felipe Freya Gabriel Giulia Hamza Hana Hiroshi Ibrahim Imani Ines Ivan Jamal Javier Jin Joaquin Kai "
         "Kamala Kenji Khadija Kofi Lars Layla Leila Lucia Luis Mateo Mei Mohammed Nadia Naledi Nikhil Nina Noor Olga Omar Oscar Pablo Priya "
         "Quentin Rafael Rania Ravi Rosa Rustam Saanvi Salma Santiago Sanjay Sena Sofia Sunita Tariq Thandiwe Tomas Uma Valentina Vikram "
         "Wei Ximena Yara Yusuf Zainab Zara Brian Karen Greg Megan Tyler Brittany Doug Heather Kevin Laura Marcus Nicole Owen Paige Raymond "
         "Shannon Travis Wendy Colin Fiona Declan Siobhan Hamish Morag Rhys Bethan Liam Grace Nathan Hannah Ethan Olivia Jack Ruby Noah Isla").split()
LAST = ("Abara Adeyemi Alvarez Andersen Bakshi Becker Bianchi Chowdhury Costa Dubois Eriksen Fernandes Fischer Gallagher Garcia Gupta Haddad "
        "Hansen Hernandez Hoang Ibarra Ito Jankowski Jensen Kaur Kim Kowalski Kumar Lambert Larsen Le Lindqvist Lopez Mahlangu Martins Mendoza "
        "Moreau Murphy Nakamura Nguyen Novak Obi Okafor Olsen Ortiz Osei Park Patel Petrov Popescu Quinn Ramirez Reyes Rossi Saito Santos Schmidt "
        "Sato Shah Silva Singh Sokolov Suzuki Tanaka Torres Tran Vargas Volkov Wagner Walsh Yamamoto Yilmaz Zhang Zhou Brooks Carter Coleman "
        "Foster Hughes Jenkins Kelly Morgan Murray Parker Reid Robertson Stewart Sullivan Turner Ward Watson Wright").split()
PLACES = ("Austin Denver Phoenix Atlanta Seattle Portland Miami Dallas Houston Nashville Detroit Toronto Vancouver Montreal Calgary Dublin Cork "
          "Glasgow Edinburgh Manchester Leeds Bristol Cardiff Sydney Melbourne Brisbane Auckland Wellington Johannesburg Cape Town Nairobi Lagos "
          "Accra Mumbai Delhi Bangalore Singapore Manila Tokyo Osaka Seoul Berlin Munich Hamburg Madrid Barcelona Lisbon Porto Rome Milan Vienna "
          "Prague Warsaw Stockholm Oslo Helsinki Copenhagen Amsterdam Brussels Zurich Geneva Lyon Marseille Istanbul Dubai Cairo Lima Bogota "
          "Santiago Mexico City Sacramento Tucson Omaha Tulsa Boise Albany Hartford Raleigh Charlotte Pittsburgh Cleveland Minneapolis Milwaukee").split(" ")

# ---------------------------------------------------------------- v3 coverage spec
SETTINGS = SETTINGS + [
    "weekly team sync on a video call", "one-on-one with a manager", "sales discovery call", "vendor negotiation call", "support escalation call",
    "pharmacy refill phone line", "dentist appointment booking", "utility company outage line", "airline rebooking desk", "car rental counter",
    "parcel locker pickup", "bank fraud department", "mortgage pre-approval call", "school parent-teacher conference", "university registrar office",
    "hospital discharge briefing", "lab results phone call", "IT password reset call", "conference keynote Q&A", "podcast interview",
    "earnings call", "sprint planning meeting", "incident post-mortem", "construction bid review", "insurance adjuster visit",
    "property management voicemail", "gym membership desk", "hotel concierge", "train conductor announcement", "stadium announcer",
    "election results broadcast", "weather forecast", "cooking class", "hardware store counter", "electronics repair shop",
    "auto mechanic estimate", "DMV appointment", "court clerk office", "notary appointment", "tax preparer office", "travel agent call"]
REGISTERS = REGISTERS + ["hesitant", "fast talker", "careful phone voice", "reading from a screen"]
FORMS = FORMS + ["read-back to confirm", "spelling over the phone", "correction of an earlier number", "list of two or three items"]
LENGTHS = {"short": "6-12 words", "medium": "13-24 words", "long": "25-38 words", "passage": "40-60 words, 2 to 4 sentences"}
POS = {"start": "first words of the sentence", "end": "last words of the sentence", "middle": "middle of the sentence"}
ID_KINDS = ["account number", "order number", "tracking number", "serial number", "member ID", "confirmation number", "case number",
            "policy number", "invoice number", "reference number", "ticket number", "patient ID"]
GROUP = {5: [[5], [2, 3]], 6: [[3, 3], [6]], 7: [[3, 4], [7]], 8: [[4, 4], [8]], 9: [[3, 3, 3], [9]], 10: [[3, 3, 4], [10]],
         11: [[3, 4, 4], [1, 3, 3, 4]], 12: [[4, 4, 4], [3, 3, 3, 3]], 13: [[3, 10], [4, 4, 5], [13]]}


# ---------------------------------------------------------------- digit strings
def norep(R, n, first=None):
    ds = [first if first is not None else str(R.randint(0, 9))]
    while len(ds) < n:
        d = str(R.randint(0, 9))
        if d != ds[-1]: ds.append(d)
    return "".join(ds)


def with_repeat(R, n, kind):
    ds = list(norep(R, n))
    k = R.randint(0, n - 2)
    if kind == "pair": ds[k + 1] = ds[k]
    elif kind == "triple" and n >= 3: k = min(k, n - 3); ds[k + 1] = ds[k + 2] = ds[k]
    elif kind == "multi":
        for k in R.sample(range(0, n - 1, 2), min(2 + (n > 9), len(range(0, n - 1, 2)))): ds[k + 1] = ds[k]
    return "".join(ds)


def read_digits(seg, R, style):
    if len(seg) == 2 and seg[0] != "0" and R.random() < style.get("pairs", 0.0): return P.w99(int(seg))
    return P.digits_reading(seg, R, {"double": style["double"], "triple": style["triple"], "oh": 0.5})


def grouped_reading(ds, groups, R, style):
    out, i = [], 0
    for g in groups:
        out.append(read_digits(ds[i:i + g], R, style)); i += g
    return ", ".join(out)


def repeat_kind(R, cross_ok=True):
    return R.choices(["none", "pair", "triple", "multi", "cross"], weights=[50, 20, 10, 12, 8 if cross_ok else 0])[0]


def make_digits(R, n, groups, kind, first=None):
    rk = repeat_kind(R, len(groups) > 1)
    if rk == "none": ds = norep(R, n, first)
    elif rk == "cross":  # identical digits across a group boundary ("…five, five…")
        ds = list(norep(R, n, first)); b = groups[0]; ds[b] = ds[b - 1]; ds = "".join(ds)
    else: ds = with_repeat(R, n, rk)
    if first is not None: ds = first + ds[1:]
    has = bool(re.search(r"(\d)\1", ds))
    if rk == "none" and has: return make_digits(R, n, groups, kind, first)  # forced first digit created a repeat: redraw
    if rk != "none" and not has: rk = "none"
    style = {"double": R.choice([0.0, 0.0, 0.5, 0.9]), "triple": R.choice([0.0, 0.5, 0.9]), "pairs": R.choice([0.0, 0.0, 0.3])}
    return ds, grouped_reading(ds, groups, R, style), rk


def digit_number(R):
    x = R.random()
    if x < 0.30:  # phones
        y = R.random()
        if y < 0.55: n, g = 10, [3, 3, 4]
        elif y < 0.70: n, g = 11, [1, 3, 3, 4]
        elif y < 0.85: n, g = 7, [3, 4]
        else:  # toll-free, "one eight hundred" / "eight hundred"
            rest, rk = norep(R, 7), "none"
            a = R.choice(["eight hundred", "eight eight eight", "eight seven seven", "eight six six"])
            lead = R.choice(["one ", "1 ", ""]).replace("1 ", "one ")
            body = grouped_reading(rest, [3, 4], R, {"double": 0.5, "triple": 0.5})
            lab = ("1" if lead else "") + {"eight hundred": "800", "eight eight eight": "888", "eight seven seven": "877", "eight six six": "866"}[a] + rest
            return dict(kind="phone number (toll-free)", reading=f"{lead}{a}, {body}", label=lab, value=lab, repeat=False, rk="none", groups="tollfree")
        ds, r, rk = make_digits(R, n, g, "phone", first=str(R.randint(2, 9)) if n != 11 else "1")
        if n == 11:
            ds = "1" + str(R.randint(2, 9)) + ds[2:]; r = grouped_reading(ds, g, R, {"double": 0.5, "triple": 0.5})
            rk = rk if re.search(r"(\d)\1", ds) else "none"
        return dict(kind="phone number", reading=r, label=ds, value=ds, repeat=rk != "none", rk=rk, groups="-".join(map(str, g)))
    if x < 0.37:  # zip codes
        if R.random() < 0.8: ds, r, rk = make_digits(R, 5, [5], "zip"); g = "5"
        else: ds, r, rk = make_digits(R, 9, [5, 4], "zip"); g = "5-4"
        return dict(kind="ZIP code", reading=r, label=ds, value=ds, repeat=rk != "none", rk=rk, groups=g)
    if x < 0.44:  # ISBN
        n = R.choice([10, 13]); g = R.choice([[3, 10], [13]]) if n == 13 else [10]
        ds, r, rk = make_digits(R, n, g, "isbn")
        if n == 13: ds = "978" + ds[3:]; r = grouped_reading(ds, g, R, {"double": 0.5, "triple": 0.5})
        return dict(kind="ISBN", reading=r, label=ds, value=ds, repeat=rk != "none", rk=rk, groups="-".join(map(str, g)))
    n = R.choices([5, 6, 7, 8, 9, 10, 11, 12, 13], weights=[18, 12, 12, 12, 10, 12, 8, 10, 6])[0]
    g = R.choice(GROUP[n])
    ds, r, rk = make_digits(R, n, g, "id")
    return dict(kind=R.choice(ID_KINDS), reading=r, label=ds, value=ds, repeat=rk != "none", rk=rk, groups="-".join(map(str, g)))


def digit_number_tagged(R):
    n = digit_number(R)
    has = bool(re.search(r"(\d)\1", n["value"]))
    if n["rk"] == "none" and has: n["rk"] = "prefix"  # 800 / 888 / 978 prefixes carry their own repeats
    n["repeat"] = has
    return n


def minimal_pair(R):
    """two short codes from the SAME base digits: one with an adjacent repeat, the other with the repeat moved or removed
    ("two two four" vs "two four four" / "two four"); or 'two two' vs 'twenty two' vs 'two twenty'. 3-4 digits -> word labels."""
    if R.random() < 0.5:
        n = R.choice([3, 4]); base = list(norep(R, n)); k = R.randint(0, n - 2)
        A = base.copy(); A[k + 1] = A[k]
        if R.random() < 0.5 and n >= 3:
            j = R.choice([i for i in range(n - 1) if i != k]); B = base.copy(); B[j + 1] = B[j]   # repeat moved
        else:
            B = base   # repeat removed
        A, Bd = "".join(A), "".join(B)
        if A == Bd: Bd = "".join(base)
        def mk(d):
            rd = P.digits_reading(d, R, {"double": 0.0, "triple": 0.0, "oh": 0.5})
            return dict(kind="short code (digits, an ID)", reading=rd, label=rd, value=d, repeat=bool(re.search(r"(\d)\1", d)), rk="minpair", groups=str(n))
        return [mk(A), mk(Bd)], "Contrast the two codes naturally (a correction or read-back: not this one, that one)."
    d = R.randint(2, 9)
    opts = [f"{U[d]} {U[d]}", P.w99(d * 10 + d), f"{U[d]} {P.w99(d * 10)}"]
    two = R.sample(opts, 2)
    return [dict(kind="short number said aloud", reading=t, label=t, value=None, repeat=t.split()[0] == t.split()[-1], rk="minpair", groups="-") for t in two], \
        "Contrast the two numbers naturally (a correction or read-back)."


SHORT_ID_KINDS = ["room number", "extension", "PIN", "flight number", "gate number", "suite number", "apartment number", "bus route",
                  "platform number", "form number", "locker number", "table number"]


def short_id(R):
    """3-4 digit IDs read digit by digit in natural contexts ("room two two oh one"); word labels (< 5 digits rule)"""
    n = R.choice([3, 4]); rk = R.choices(["none", "pair", "triple"], weights=[50, 38, 12])[0]
    ds = norep(R, n) if rk == "none" else with_repeat(R, n, rk if n >= 3 else "pair")
    rd = P.digits_reading(ds, R, {"double": R.choice([0.0, 0.5]), "triple": R.choice([0.0, 0.5]), "oh": 0.5})
    kind = R.choice(SHORT_ID_KINDS)
    return dict(kind=f"{kind} (digits read one by one)", reading=rd, label=rd, value=ds, repeat=bool(re.search(r"(\d)\1", ds)), rk=rk, groups=str(n))


# ---------------------------------------------------------------- alphanumerics and codes
LET = "B C D E F G H J K L M N P Q R S T U V W X Y Z".split()  # no A (read as the article), no I/O (vs one/oh)


def alnum(R):
    x = R.random()
    if x < 0.35: pat = R.choice(["LDLDLD", "LLDDDD", "DLLDLL", "LDDLLD"])  # confirmation codes
    elif x < 0.6: pat = R.choice(["LLDD DLL", "LDD DLL", "LLD DLL"])  # UK-style postcodes
    elif x < 0.8: pat = R.choice(["DLLLDDD", "LLLDDDD", "LLDDLLL"])  # licence plates
    else: pat = R.choice(["LL DDDD", "LL DDD"])  # flight / room codes
    lab, rd = [], []
    for c in pat:
        if c == " ": lab.append(" "); rd.append(","); continue
        if c == "L": l = R.choice(LET); lab.append(l); rd.append(l)
        else: d = R.randint(0, 9); lab.append(str(d)); rd.append(R.choice(["oh", "zero"]) if d == 0 else U[d])
    reading = " ".join(rd).replace(" ,", ",")
    return dict(kind="alphanumeric code (letters and digits, spelled out)", reading=reading, label="".join(lab), value="".join(lab), repeat=False, rk="none", groups=pat)


CODES3 = [(f"{l} {P.w99(n)}", f"{l}{n}") for l in "B C D F G J K M N P Q R S T V X Z".split() for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 21, 30, 40, 50)] + \
         [(f"{P.w99(n)} {l}", f"{n}{l}") for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 21) for l in "B C D F G K M P S X".split()]


def code3(R):
    if R.random() < 0.45:  # aisle B twelve, apartment four B, model X five: single-token letter-digit codes
        sp, lab = R.choice(CODES3); return dict(kind="letter-digit code (like a model, aisle or seat code)", reading=sp, label=lab, value=lab, repeat=False, rk="none", groups="-")
    sp, lab = R.choice(CODES2)
    if " " in lab or not re.search(r"[A-Za-z]", lab):  # owner 10-24: word+number codes and bare numbers are labelled as words
        return dict(kind="name with a number in it", reading=sp, label=sp, value=None, repeat=False, rk="none", groups="-")
    return dict(kind="product or code name", reading=sp, label=lab, value=lab, repeat=False, rk="none", groups="-")


# ---------------------------------------------------------------- ordinary numbers (labels = spoken words)
UNITS = ["kilometers", "miles", "meters", "feet", "inches", "pounds", "kilograms", "grams", "milligrams", "liters", "gallons", "degrees",
         "minutes", "hours", "days", "weeks", "megabytes", "gigabytes", "volts", "watts", "calories", "steps", "seats", "people"]


def ordinary3(R):
    k = R.choices(["o2", "or", "between", "frac", "approx", "neg", "unit", "ordinal"], weights=[60, 5, 5, 6, 4, 3, 10, 7])[0]
    if k == "o2":
        n = ordinary2(R); return dict(n, rk="word" if n["repeat"] else "none", groups="-")
    if k == "or":
        a = R.randint(1, 20); r = f"{P.card(a)} or {P.card(a + R.choice([1, 1, 2, 5]))}"; kind = "rough count (X or Y)"
    elif k == "between":
        a = logu(R, 2, 5000); r = f"between {P.card(a)} and {P.card(a + max(1, int(a * R.uniform(0.2, 1.0))))}"; kind = "range (between X and Y)"
    elif k == "frac":
        fixed = ["three quarters", "a third", "two thirds", "a quarter"]
        para = [f"{P.card(R.randint(1, 30))} and a half", f"{P.card(R.randint(1, 20))} and a quarter", f"{P.card(R.randint(1, 12))} and three quarters",
                f"{P.card(R.randint(2, 5))} thirds", f"{P.card(R.randint(1, 7))} eighths", f"{P.card(R.randint(1, 50))} and a half million",
                f"{P.card(R.randint(1, 9))} and a half billion", f"one in {P.card(R.choice([3, 4, 5, 8, 10, 20, 100]))}"]
        r = R.choice(fixed) if R.random() < 0.08 else R.choice(para); kind = "fraction or 'and a half'"
    elif k == "approx":
        r = R.choice(["a hundred", "a thousand", "a couple hundred", "a few hundred", "a few thousand", "a dozen", "half a dozen", "a couple of dozen",
                      "a hundred and fifty", "a million", "half a million", "a quarter million", "a hundred thousand", "a couple thousand", "a billion",
                      "a thousand or so", "a hundred or so", "nearly a thousand", "almost a million", "well over a hundred", "a few dozen",
                      f"about {P.card(R.choice([20, 30, 40, 50, 60, 100, 200, 500]))}", f"roughly {P.card(R.choice([10, 15, 25, 75, 150, 250, 1000]))}"]); kind = "approximate amount"
    elif k == "neg":
        r = f"minus {P.card(R.randint(1, 40))}" + R.choice([" degrees", "", " percent"]); kind = "negative number"
    elif k == "unit":
        v = R.choice([P.card(logu(R, 2, 5000)), f"{P.card(R.randint(1, 99))} point {U[R.randint(1, 9)]}"]); r = f"{v} {R.choice(UNITS)}"; kind = "measurement with a unit"
    else:
        r = R.choice([f"{ord_word(R.randint(1, 31))} floor", f"the {ord_word(R.randint(1, 21))} century", f"{ord_word(R.randint(1, 10))} place",
                      f"the {ord_word(R.randint(1, 30))} time", f"{ord_word(R.randint(2, 12))} grade", f"the {ord_word(R.randint(1, 25))} anniversary"]); kind = "ordinal"
    return dict(kind=kind, reading=r, label=r, value=None, repeat=False, rk="none", groups="-")


def with_um(n, R):
    """'three hundred, um, fifty two': hesitation inside a word-labelled reading (label keeps it, like the spoken text)"""
    ws = n["reading"].split()
    if n["label"] != n["reading"] or len(ws) < 2: return n, False
    numish = NUMW | {"oh", "zero", "point"}
    big = {"hundred", "thousand", "million", "billion", "and"}
    cut = [k for k in range(1, len(ws)) if ws[k - 1].lower().strip(",") in numish and ws[k].lower() in numish
           and ws[k].lower() not in big and ws[k - 1].lower().strip(",") not in ("and", "one")]
    if not cut: return n, False
    k = R.choice(cut); h = R.choice(["um", "uh"])
    r = " ".join(ws[:k]) + f", {h}, " + " ".join(ws[k:])
    return dict(n, reading=r, label=r), True


def filler_nums(R):
    n = ordinary3(R)
    n2, inside = with_um(n, R)
    if inside: return [n2], "Keep the hesitation exactly where it is inside the number."
    d = digit_number_tagged(R)
    return [d], "Put a hesitation (um or uh) right before or right after the number, never inside it, and do not repeat the number."


# ---------------------------------------------------------------- repeats inside word numbers, across clauses, teen/ty pairs
TEENS = {13: 30, 14: 40, 15: 50, 16: 60, 17: 70, 18: 80, 19: 90}


def word_repeat(R):
    x = R.random()
    if x < 0.30:  # repeated digits after a decimal point: "fifty five point five five", "point oh oh five"
        a = R.randint(1, 99); d = R.randint(0, 9); w = R.choice(["oh", "zero"]) if d == 0 else U[d]
        r = R.choice([f"{P.w99(a)} point {w} {w}", f"{P.w99(a)} point {w} {w} {U[R.randint(1, 9)]}", f"point {w} {w}"])
        return [dict(kind="decimal with a repeated digit", reading=r, label=r, value=None, repeat=True, rk="word", groups="-")], ""
    if x < 0.55:  # the same number twice in a row across a clause boundary, adjacent by construction: "five, five"
        v = R.choice([R.randint(1, 20), R.randint(21, 99), R.randint(100, 999)]); r = f"{P.card(v)}, {P.card(v)}"
        return [dict(kind="a count said twice in a row", reading=r, label=r, value=None, repeat=True, rk="cross_clause", groups="-")], \
            "The number is said twice in a row across a clause boundary; keep the reading exactly as given (like: we need five, five is plenty)."
    if x < 0.80:  # clock / year / range word repeats: "ten ten", "eleven eleven", "twenty twenty to twenty twenty one"
        h = R.randint(10, 12); y = R.choice([20, 21])
        r, kind = R.choice([(f"{P.w99(h)} {P.w99(h)}", "a time of day"), (f"{P.w99(y)} {P.w99(y)} to {P.w99(y)} {P.w99(y + 1)}", "a span of years"),
                            (f"{P.w99(h)} {P.w99(h)} A M", "a time of day")])
        return [dict(kind=kind, reading=r, label=r.replace(" A M", " AM"), value=None, repeat=True, rk="word", groups="-")], ""
    kind = R.choice(["a price said without dollars", "a year", "a time of day", "a model number said aloud"])  # confusable teen/ty neighbours
    t = R.choice([13, 14, 15] if kind == "a time of day" else list(TEENS))
    r = f"{P.w99(t)} {P.w99(TEENS[t])}" + ("" if kind == "a time of day" else R.choice(["", " " + U[R.randint(1, 9)]]))
    return [dict(kind=kind, reading=r, label=r, value=None, repeat=False, rk="teen_ty", groups="-")], ""


SOUNDALIKE = ["to", "too", "for", "won", "ate", "tree", "tin", "free", "tan"]
REPEAT_WORDS = ["no no", "very very", "bye bye", "that that", "had had", "really really", "yeah yeah", "okay okay", "so so", "now now", "wait wait", "go go"]


# ---------------------------------------------------------------- spec
SPEC = {  # cell: (target rows, builder -> (numbers, extra), lengths, positions)
    "digits": (4400, lambda R: ([digit_number_tagged(R)], ""), ["short", "medium", "long"], ["start", "middle", "end"]),
    "digits_minpair": (500, minimal_pair, ["short", "medium"], ["middle"]),
    "alnum": (700, lambda R: ([alnum(R)], "Say the code letter by letter, digit by digit, exactly as given."), ["short", "medium"], ["middle", "end"]),
    "codes": (700, lambda R: ([code3(R)], ""), ["short", "medium"], ["start", "middle", "end"]),
    "ordinary": (3600, lambda R: ([ordinary3(R)] + ([ordinary3(R)] if R.random() < 0.25 else []), ""), ["short", "medium", "long"], ["start", "middle", "end"]),
    "passage": (1300, lambda R: ([R.choices([digit_number_tagged, ordinary3, alnum], weights=[3, 6, 1])[0](R) for _ in range(R.choice([3, 3, 4]))],
                                 "Spread the numbers through the passage: one near the start, one in the middle, one near the end."), ["passage"], ["spread"]),
    "filler": (500, filler_nums, ["short", "medium"], ["middle", "end"]),
    "word_repeats": (600, word_repeat, ["short", "medium", "long"], ["start", "middle", "end"]),
    "soundalike": (600, lambda R: ([], "Use NO numbers. Naturally use the word '%s' in its normal meaning%s." % (
        R.choices(SOUNDALIKE, weights=[16, 14, 16, 10, 8, 6, 5, 4, 3])[0], R.choice(["", ", and also the word '%s'" % R.choice(SOUNDALIKE)]))), ["short", "medium"], ["none"]),
    "no_number": (300, lambda R: ([], "Use NO numbers at all."), ["short", "medium", "long"], ["none"]),
    "repeat_word_control": (250, lambda R: ([], "Use NO numbers. Include the repeated words '%s' naturally, back to back, exactly once." % R.choice(REPEAT_WORDS)),
                            ["short", "medium"], ["none"]),
    "short_id": (700, lambda R: ([short_id(R)], ""), ["short", "medium"], ["start", "middle", "end"]),
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--scale", type=float, default=1.0); ap.add_argument("--seed", type=int, default=3024)
    ap.add_argument("--root", default=".", help="data directory (briefs/ and spec_<tag>.json are written here)")
    a = ap.parse_args(); os.chdir(a.root); R = random.Random(a.seed); out = []
    names = FIRST + [f"{f} {l}" for f, l in zip(R.sample(FIRST, len(FIRST)), R.sample(LAST * 2, len(FIRST)))]
    for cell, (cnt, build, lengths, positions) in SPEC.items():
        for _ in range(round(cnt * a.scale)):
            nums, extra = build(R)
            L = R.choice(lengths); pos = R.choice(positions)
            b = dict(cell=cell, setting=R.choice(SETTINGS), register=R.choice(REGISTERS), form=R.choice(FORMS), length=f"{L} ({LENGTHS[L]})",
                     position={"start": POS["start"], "end": POS["end"], "middle": POS["middle"], "spread": "spread through the passage", "none": "no numbers"}[pos],
                     pos_key=pos, numbers=nums, extra=extra, disfluent=cell == "filler" or R.random() < 0.10,
                     names_nearby=False, name=R.choice(names) if R.random() < 0.35 else "", place=R.choice(PLACES) if R.random() < 0.2 else "")
            b["id"] = "b3_" + hashlib.sha1(json.dumps(b, sort_keys=True).encode() + os.urandom(4)).hexdigest()[:10]
            out.append(b)
    for b in out:  # repeat tags from the actual digits/characters (builders' intent can differ from the value)
        for n in b["numbers"]:
            v = str(n.get("value") or "")
            if n.get("rk") in ("minpair", "cross_clause", "word", "teen_ty") or not v: continue
            runs = [len(m.group()) for m in re.finditer(r"(.)\1+", v)]
            n["rk"] = "none" if not runs else "triple" if max(runs) >= 3 else "multi" if len(runs) > 1 else "pair"
            n["repeat"] = bool(runs)
        b["require_repeat_word"] = b["cell"] == "repeat_word_control"
    R.shuffle(out); os.makedirs("briefs", exist_ok=True)
    open(f"briefs/{a.tag}.jsonl", "w").write("".join(json.dumps(b, ensure_ascii=False) + "\n" for b in out))
    json.dump({c: v[0] for c, v in SPEC.items()}, open(f"spec_{a.tag}.json", "w"), indent=1)
    nums = [n for b in out for n in b["numbers"]]
    dig = [n for n in nums if re.fullmatch(r"\d+", str(n.get("value") or "")) and len(str(n["value"])) >= 5]
    print(len(out), "briefs", dict(collections.Counter(b["cell"] for b in out)))
    print("positions", dict(collections.Counter(b["pos_key"] for b in out)))
    print("digit strings", len(dig), "repeat-free share", round(sum(n["rk"] == "none" for n in dig) / max(len(dig), 1), 3), "repeat kinds", dict(collections.Counter(n["rk"] for n in dig)))
    print("phone grouping", dict(collections.Counter(n["groups"] for n in nums if n["kind"].startswith("phone"))))
    print("lengths", dict(collections.Counter(len(n["value"]) for n in dig)))
    print("kinds", dict(collections.Counter(n["kind"] for n in nums).most_common(40)))


if __name__ == "__main__": main()
