"""Spoken readings of numbers and symbols (English).

Each `kind` function takes a `random.Random` and returns the words a speaker would say. The
training target is exactly that text, so no written form is ever produced. Spelling follows
Parakeet's own word output: tens and ones separated by a space ("twenty seven"), no hyphens,
lowercase "am"/"pm" as separate words.

>>> cardinal(0), cardinal(15), cardinal(40), cardinal(57)
('zero', 'fifteen', 'forty', 'fifty seven')
>>> cardinal(312), cardinal(312, use_and=True)
('three hundred twelve', 'three hundred and twelve')
>>> cardinal(2857)
'two thousand eight hundred fifty seven'
>>> cardinal(1_200_000_000)
'one billion two hundred million'
>>> ordinal(1), ordinal(21), ordinal(40), ordinal(112)
('first', 'twenty first', 'fortieth', 'one hundred twelfth')
>>> year(1998), year(2008), year(2026), year(1900)
('nineteen ninety eight', 'two thousand eight', 'twenty twenty six', 'nineteen hundred')
>>> pair(1305), pair(4090), pair(1200)
('thirteen oh five', 'forty ninety', 'twelve hundred')
>>> digits("4150", oh=True)
'four one five oh'
"""

import random
import re

ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
SCALES = [(10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand")]
ORD_IRREG = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
             "nine": "ninth", "twelve": "twelfth"}


def _under_1000(n: int, use_and: bool) -> list[str]:
    w = []
    if n >= 100:
        w += [ONES[n // 100], "hundred"]
        n %= 100
        if n and use_and:
            w.append("and")
    if n >= 20:
        w.append(TENS[n // 10])
        if n % 10:
            w.append(ONES[n % 10])
    elif n or not w:
        w.append(ONES[n])
    return w


def cardinal(n: int, use_and: bool = False) -> str:
    if n < 0:
        return "minus " + cardinal(-n, use_and)
    if n < 1000:
        return " ".join(_under_1000(n, use_and))
    w = []
    for v, name in SCALES:
        if n >= v:
            w += [cardinal(n // v, use_and), name]
            n %= v
    if n:
        if use_and and n < 100:
            w.append("and")
        w += _under_1000(n, use_and)
    return " ".join(w)


def ordinal(n: int) -> str:
    words = cardinal(n).split()
    last = words[-1]
    if last in ORD_IRREG:
        words[-1] = ORD_IRREG[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def digits(s: str, oh: bool = False) -> str:
    return " ".join(("oh" if oh and c == "0" else ONES[int(c)]) for c in s if c.isdigit())


def pair(n: int) -> str:
    """Four-digit numbers read as two pairs: 1305 'thirteen oh five', 1200 'twelve hundred'."""
    hi, lo = divmod(n, 100)
    if lo == 0:
        return cardinal(hi) + " hundred"
    if lo < 10:
        return f"{cardinal(hi)} oh {ONES[lo]}"
    return f"{cardinal(hi)} {cardinal(lo)}"


def year(y: int) -> str:
    if 2000 <= y <= 2009:
        return cardinal(y)
    if y % 100 == 0 and y < 2000:
        return cardinal(y // 100) + " hundred"
    return pair(y)


def _pick(r: random.Random, options: list[tuple[float, str]]) -> str:
    total = sum(w for w, _ in options)
    x = r.random() * total
    for w, s in options:
        x -= w
        if x <= 0:
            return s
    return options[-1][1]


def _maybe_a(words: str, r: random.Random, p: float = 0.3) -> str:
    """'one hundred ...' -> 'a hundred ...' some of the time."""
    for s in ("one hundred", "one thousand", "one million", "one billion"):
        if words.startswith(s) and r.random() < p:
            return "a" + words[3:]
    return words


# --- kinds ----------------------------------------------------------------------------------


def k_int_small(r):
    return cardinal(r.randint(0, 20))


def k_int_tens(r):
    return cardinal(r.randint(21, 99))


def k_int_hundreds(r):
    n = r.randint(1, 9) * 100 if r.random() < 0.3 else r.randint(100, 999)
    opts = [(0.5, cardinal(n)), (0.3, cardinal(n, use_and=True))]
    if n % 100 >= 10:
        opts.append((0.2, f"{ONES[n // 100]} {cardinal(n % 100)}"))
    return _maybe_a(_pick(r, opts), r)


def k_int_4digit(r):
    n = r.randint(10, 99) * 100 if r.random() < 0.3 else r.randint(1000, 9999)
    opts = [(0.4, cardinal(n)), (0.15, cardinal(n, use_and=True))]
    if n % 100 == 0 and n % 1000 != 0:
        opts.append((0.45, pair(n)))
    elif r.random() < 0.3:
        opts.append((0.2, pair(n)))
    return _maybe_a(_pick(r, opts), r)


def k_year(r):
    y = r.randint(1990, 2030) if r.random() < 0.8 else r.randint(1900, 2039)
    opts = [(0.85, year(y))]
    if y >= 2010:
        opts.append((0.15, cardinal(y)))
    if 2000 <= y <= 2009:
        opts.append((0.3, cardinal(y, use_and=True)))
    return _pick(r, opts)


def k_big(r):
    scale = r.choice(["thousand", "million", "million", "billion", "billion", "trillion"])
    m = r.randint(1, 999) if r.random() < 0.7 else r.randint(1, 9)
    if r.random() < 0.3:
        s = f"{cardinal(m)} point {digits(str(r.randint(1, 9)))} {scale}"
    else:
        s = f"{cardinal(m)} {scale}"
    return _maybe_a(s, r, 0.2)


def _decimal(r, max_int=999):
    i = r.randint(0, max_int) if r.random() < 0.7 else r.randint(0, 9)
    frac = str(r.randint(1, 999)).rstrip("0") or "5"
    frac = frac[: r.choice([1, 1, 2, 2, 3])]
    # Fractions with a leading zero ("point oh five", "point zero zero one"): none were generated before, and
    # the model then wrote "point zero five" for a spoken "point oh five" (adversarial test 2026-10-09).
    if r.random() < 0.3:
        frac = "0" * r.choice([1, 1, 2]) + frac[:2]
    tail = f"point {digits(frac, oh=r.random() < 0.4)}"
    if i == 0:
        # As said (owner 2026-10-09): a bare "point nine" is as common as "zero point nine".
        return _pick(r, [(0.4, f"zero {tail}"), (0.45, tail), (0.15, f"oh {tail}")])
    return f"{cardinal(i)} {tail}"


def k_decimal(r):
    return _decimal(r)


CUR = [("dollar", "dollars", "cent", "cents"), ("euro", "euros", "cent", "cents"),
       ("pound", "pounds", "penny", "pence"),
       # v10 (2026-10-09): real calls leak "EUR"/"EU" before euro amounts and synthetic euro rows leaked 14 of 20;
       # money was 80% dollars. More currencies, and dollars down to 55%.
       ("yen", "yen", "sen", "sen"), ("yuan", "yuan", "fen", "fen"), ("rupee", "rupees", "paisa", "paise"),
       ("franc", "francs", "centime", "centimes"), ("krona", "kronor", "ore", "ore"),
       ("zloty", "zloty", "grosz", "groszy"), ("peso", "pesos", "centavo", "centavos")]


def k_money(r):
    one, many, sub1, subs = CUR[0] if r.random() < 0.55 else (CUR[1] if r.random() < 0.5 else r.choice(CUR[2:]))
    if r.random() < 0.3:
        scale = r.choice(["thousand", "million", "billion"])
        amt = f"{cardinal(r.randint(1, 999))} point {digits(str(r.randint(1, 9)))}" if r.random() < 0.4 \
            else cardinal(r.randint(1, 999))
        return f"{amt} {scale} {many}"
    d = r.randint(1, 9999) if r.random() < 0.7 else r.randint(1, 99)
    unit = one if d == 1 else many
    dw = "a" if d == 1 and r.random() < 0.5 else cardinal(d)
    if r.random() < 0.3:
        c = r.randint(1, 99)
        cw = cardinal(c)
        return _pick(r, [(0.5, f"{dw} {unit} and {cw} {sub1 if c == 1 else subs}"),
                         (0.3, f"{cardinal(d)} {cw}"), (0.2, f"{dw} {unit} {cw}")])
    if d < 100 and r.random() < 0.15:
        return f"{cardinal(d)} {sub1 if d == 1 else subs}"
    if many == "dollars" and r.random() < 0.05:
        return f"{cardinal(d)} bucks"
    return f"{dw} {unit}"


def k_percent(r):
    if r.random() < 0.3:
        n = _decimal(r, 99)
    elif r.random() < 0.1:
        n = f"{cardinal(r.randint(1, 20))} and a half"
    else:
        n = cardinal(r.randint(0, 100))
    return f"{n} percent"


def k_basis_points(r):
    return f"{cardinal(r.choice([5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 200]))} basis points"


def k_clock(r):
    h, m = r.randint(1, 12), r.randint(0, 59)
    suffix = _pick(r, [(0.5, ""), (0.25, " PM"), (0.25, " AM")])
    if m == 0:
        base = _pick(r, [(0.6, f"{cardinal(h)} o'clock"), (0.4, cardinal(h))])
    elif m < 10:
        base = f"{cardinal(h)} oh {ONES[m]}"
    else:
        base = f"{cardinal(h)} {cardinal(m)}"
    roll = r.random()
    if roll < 0.08:
        return f"a quarter past {cardinal(h)}"
    if roll < 0.14:
        return f"half past {cardinal(h)}"
    if roll < 0.2:
        return f"{cardinal(r.randint(13, 23))} {cardinal(r.choice([15, 30, 45]))}"
    return base + (suffix if "o'clock" not in base else "")


def k_phone(r):
    oh = r.random() < 0.5
    if r.random() < 0.2:
        area = _pick(r, [(0.7, "eight hundred"), (0.3, digits("800", oh))])
    else:
        area = digits(str(r.randint(201, 989)), oh)
    a, b = digits(str(r.randint(200, 999)), oh), digits(f"{r.randint(0, 9999):04d}", oh)
    sep = ", " if r.random() < 0.3 else " "
    return f"{area}{sep}{a}{sep}{b}"


def k_code(r):
    s = str(r.randint(1000, 999999))
    if r.random() < 0.6 or len(s) % 2:
        return digits(s, oh=r.random() < 0.4)
    return " ".join(cardinal(int(s[i:i + 2])) if s[i] != "0" else f"oh {ONES[int(s[i + 1])]}"
                    for i in range(0, len(s), 2))


LETTERS = "A B C D E F G H J K M N P Q R S T V W X Z".split()


def k_alnum(r):
    pre = "".join(r.choice(LETTERS) for _ in range(r.choice([1, 1, 2, 3])))
    n = r.randint(1, 9999)
    num = _pick(r, [(0.4, digits(str(n), oh=r.random() < 0.5)), (0.4, cardinal(n) if n < 100 else pair(n) if n >= 1000 else cardinal(n)),
                    (0.2, digits(str(n)))])
    if r.random() < 0.15:
        num += f" dash {cardinal(r.randint(1, 9))}"
    return f"{pre} {num}"


def _octet(r, n):
    if n < 100:
        return cardinal(n)
    opts = [(0.3, cardinal(n)), (0.3, digits(str(n)))]
    if n % 100 >= 10:
        opts.append((0.4, f"{ONES[n // 100]} {cardinal(n % 100)}"))
    return _pick(r, opts)


def k_ip(r):
    quads = r.choice([[127, 0, 0, 1], [10, 0, 0, r.randint(1, 254)], [192, 168, r.randint(0, 9), r.randint(1, 254)],
                      [r.randint(1, 223), r.randint(0, 255), r.randint(0, 255), r.randint(1, 254)]])
    sep = _pick(r, [(0.6, " point "), (0.4, " dot ")])
    return sep.join(_octet(r, q) for q in quads)


def k_version(r):
    parts = [r.randint(0, 20), r.randint(0, 15)] + ([r.randint(0, 12)] if r.random() < 0.6 else [])
    sep = _pick(r, [(0.7, " point "), (0.3, " dot ")])
    s = sep.join(cardinal(p) for p in parts)
    return f"version {s}" if r.random() < 0.3 else s


def k_port(r):
    p = r.choice([80, 443, 3000, 5000, 5173, 8000, 8080, 8443, 9000, 5432, 6379, 27017])
    opts = [(0.4, cardinal(p)), (0.2, digits(str(p)))]
    if p >= 1000:
        opts.append((0.4, pair(p)))
    return _pick(r, opts)


MONTHS = ("January February March April May June July August September October November "
          "December").split()


def k_date(r):
    mo, d = r.choice(MONTHS), r.randint(1, 28)
    y = year(r.randint(1995, 2030))
    return _pick(r, [(0.35, f"{mo} {ordinal(d)}"), (0.25, f"{mo} {ordinal(d)} {y}"),
                     (0.15, f"the {ordinal(d)} of {mo}"), (0.1, f"{mo} {cardinal(d)}"),
                     (0.1, f"{mo} of {y}"), (0.05, f"{mo} {cardinal(d)}, {y}")])


def k_ordinal(r):
    return ordinal(r.randint(1, 100) if r.random() < 0.9 else r.randint(101, 1000))


def k_range(r):
    a = r.randint(1, 90)
    b = a + r.randint(1, 30)
    return _pick(r, [(0.4, f"{cardinal(a)} to {cardinal(b)}"), (0.3, f"between {cardinal(a)} and {cardinal(b)}"),
                     (0.15, f"{cardinal(a)} through {cardinal(b)}"), (0.15, f"{cardinal(a)} or {cardinal(a + 1)}")])


FRAC = [("half", "halves"), ("third", "thirds"), ("quarter", "quarters"), ("fifth", "fifths"),
        ("eighth", "eighths"), ("tenth", "tenths")]


def k_fraction(r):
    one, many = r.choice(FRAC)
    num = r.randint(1, 3)
    f = f"{'one' if r.random() < 0.5 else 'a'} {one}" if num == 1 else f"{cardinal(num)} {many}"
    if r.random() < 0.4:
        return f"{cardinal(r.randint(1, 12))} and {f if num > 1 else 'a ' + one}"
    return f


UNITS = ["milliseconds", "seconds", "minutes", "hours", "days", "weeks", "megabytes", "gigabytes",
         "terabytes", "gigs", "megs", "kilobytes", "kilometers", "miles", "meters", "feet", "inches",
         "pounds", "kilograms", "degrees", "hertz", "kilohertz", "watts", "volts", "frames per second",
         "requests per second", "tokens", "K", "x"]


def k_measure(r):
    u = r.choice(UNITS)
    n = _decimal(r, 99) if r.random() < 0.2 else cardinal(r.choice([r.randint(1, 99), r.randint(100, 999),
                                                                   r.randint(1, 20) * 100]))
    if r.random() < 0.1:
        return f"{cardinal(r.randint(1, 9))} hours and {cardinal(r.randint(1, 59))} minutes"
    return f"{n} {u}"


def k_negative(r):
    sign = _pick(r, [(0.7, 'minus'), (0.3, 'negative')])
    if r.random() < 0.35:  # "minus point two five": a signed decimal, as said
        return f"{sign} {_decimal(r, max_int=9)}"
    return f"{sign} {cardinal(r.randint(1, 99))}"


def k_score(r):
    a, b = r.randint(0, 12), r.randint(0, 12)
    return _pick(r, [(0.7, f"{cardinal(a)} to {cardinal(b)}"), (0.2, f"{cardinal(a)} {cardinal(b)}"),
                     (0.1, f"{cardinal(a)} nil")])


def k_identifier(r):
    return r.choice(["COVID nineteen", "Windows eleven", "iPhone sixteen", "Python three", "H two hundred",
                     "A one hundred", "GPT five", "Series B", "Phase three", "Section two thirty",
                     "Title nine", "Route sixty six", "Highway one oh one", "Room four oh two",
                     "Gate B twelve", "Flight two eighteen", "Interstate five", "Chapter eleven",
                     "five G", "four G", "four K", "ten K", "ten Q", "eight K", "B to B", "B to C", "D to C", "MP three",
                     "PD one", "W two", "four oh one K", "Tele two", "Web three", "Mark two", "Gen Z",
                     "three D", "USB C", "WiFi six", "H two O", "CO two", "Formula one", "Catch twenty two"])


def k_quarter(r):
    q = r.choice(["one", "two", "three", "four"])
    year_tail = _pick(r, [(0.5, ""), (0.3, " " + year(r.randint(2018, 2030))), (0.2, f" fiscal {year(r.randint(2018, 2030))}")])
    return _pick(r, [(0.45, f"Q {q}{year_tail}"), (0.15, f"H {r.choice(['one', 'two'])}{year_tail}"),
                     (0.2, f"the {ordinal(['one', 'two', 'three', 'four'].index(q) + 1)} quarter{year_tail}"),
                     (0.2, f"FY {cardinal(r.randint(18, 30))}")])


AND_ACRONYMS = ["Q and A", "M and A", "R and D", "P and L", "AT and T", "S and P", "B and B", "H and R",
                "G and A", "SG and A", "D and I", "T and E", "rock and roll", "B and Q",
                # v10: forms real earnings calls use, written in mixed forms by v9pr30 (pinned-57, E22)
                "O and M", "IT and S", "E and P", "P and C", "L and D", "C and I", "A and R"]


def k_and_acronym(r):
    a = r.choice(AND_ACRONYMS)
    return f"S and P five hundred" if a == "S and P" and r.random() < 0.5 else a


TITLES = [("mister", "Mister"), ("missus", "Missus"), ("ms", "Ms"), ("doctor", "Doctor"), ("professor", "Professor"),
          ("saint", "Saint"), ("senator", "Senator"), ("governor", "Governor"), ("captain", "Captain"),
          ("lieutenant", "Lieutenant"), ("general", "General"), ("reverend", "Reverend"), ("judge", "Judge"),
          ("president", "President"), ("sergeant", "Sergeant")]
SURNAMES = ["Smith", "Patel", "Nguyen", "Garcia", "Okafor", "Kowalski", "Chen", "Johnson", "Rossi", "Haddad",
            "Murphy", "Tanaka", "Silva", "Novak", "Brown", "Cohen", "Ali", "Fischer", "Moreau", "Lindqvist",
            "Freight", "Brady", "Clement", "Francisco", "Wutkling", "Osei", "Ramirez", "Kaur", "Dubois", "Yilmaz",
            "Sato", "Hughes", "Mbeki", "Andersson", "O'Brien", "Kim", "Romano", "Petrov", "Lambert", "Ito"]
# v10: real speech puts a first name between title and surname ("Mister Brendan Freight"); the synthetic rows only
# had surnames and titles transferred 61% synthetic vs 4% real (Opus mining 2026-10-09).
FIRST_NAMES = ["Brendan", "Todd", "Maureen", "Ryan", "Aisha", "Wei", "Priya", "Carlos", "Fatima", "Jonas", "Elena",
               "Kwame", "Sofia", "Hiroshi", "Grace", "Omar", "Ingrid", "Marcus", "Leila", "Patrick"]


def k_title_name(r):
    t = r.choice(TITLES)[1]
    if t == "Saint":
        return r.choice(["Saint Louis", "Saint Paul", "Saint Patrick", "Saint Petersburg", "Saint Lucia"])
    name = r.choice(SURNAMES)
    if r.random() < 0.45:
        name = f"{r.choice(FIRST_NAMES)} {name}"
    return f"{t} {name}" + (_pick(r, [(0.85, ""), (0.1, " Junior"), (0.05, " Senior")]))


def k_roman(r):
    return _pick(r, [(0.3, f"World War {r.choice(['one', 'two'])}"), (0.2, f"{r.choice(['Henry', 'Louis', 'Elizabeth', 'George', 'Charles'])} the {ordinal(r.randint(1, 16))}"),
                     (0.2, f"Phase {cardinal(r.randint(1, 4))}"), (0.1, f"Super Bowl {cardinal(r.randint(40, 60))}"),
                     (0.1, f"Part {cardinal(r.randint(1, 5))}"), (0.1, f"{r.choice(['Rocky', 'Star Wars Episode', 'Final Fantasy', 'Grand Theft Auto'])} {cardinal(r.randint(2, 9))}")])


ACRONYMS = ["US", "UK", "EU", "UN", "CDC", "FBI", "CEO", "CFO", "COO", "CTO", "VP", "HR", "IT", "PR", "AI", "API",
            "SDK", "UI", "UX", "PDF", "URL", "HTML", "CSS", "SQL", "AWS", "GPU", "CPU", "RAM", "SSD", "TV", "DVD",
            "EBIT", "EBITDA", "EPS", "ARR", "MRR", "KPI", "ROI", "OKR", "SaaS", "B2B", "IPO", "LLC", "IRS", "SEC",
            "FDA", "NASA", "NATO", "GAAP", "ESG", "CapEx", "OpEx", "YoY", "ETA", "FAQ", "PhD", "MBA", "LLM", "GPT"]


def k_acronym(r):
    a = r.choice([x for x in ACRONYMS if not re.search(r"\d", x)])
    return a


def k_spelled(r):
    word = r.choice(["Smith", "Nguyen", "Kowalski", "Haddad", "Lindqvist", "Okafor", "Moreau", "Rossi", "Fischer",
                     "Tanaka", "Siobhan", "Aoife", "Jaqueline"])
    return " ".join(word.upper())


DICTATION = ["comma", "period", "full stop", "colon", "semicolon", "question mark", "exclamation point", "dash",
             "hyphen", "underscore", "open paren", "close paren", "quote", "unquote", "new line", "new paragraph",
             "dot py", "dot json", "dot com", "dot txt", "dot swift", "snake case", "camel case", "dash dash force",
             "slash slash", "backslash", "at sign", "hashtag", "plus", "equals", "less than", "greater than",
             "asterisk", "ampersand", "percent sign", "dollar sign", "tilde", "pipe"]


def k_dictation(r):
    return r.choice(DICTATION)



def k_slash(r):
    a, b = r.choice([("and", "or"), ("input", "output"), ("read", "write"), ("client", "server"),
                     ("pass", "fail"), ("on", "off"), ("he", "she"), ("start", "stop")])
    return _pick(r, [(0.5, f"{a} slash {b}"), (0.3, f"{a} {b}"), (0.2, f"{a} or {b}")])


def k_url(r):
    host = r.choice(["github", "example", "docs", "status", "support", "api", "app", "blog"])
    tld = r.choice(["com", "org", "io", "app", "dev", "net"])
    path = r.choice(["", " slash docs", " slash feedback", " slash login", " slash api slash v two",
                     " slash settings"])
    return f"{host} dot {tld}{path}"


def k_email(r):
    user = r.choice(["support", "john dot smith", "sales", "hello", "maria", "dev dot team", "billing"])
    return f"{user} at {r.choice(['example', 'acme', 'gmail', 'outlook', 'company'])} dot {r.choice(['com', 'org', 'io'])}"


def k_multiplier(r):
    return f"{cardinal(r.randint(2, 20))} {_pick(r, [(0.5, 'x'), (0.5, 'times')])}"


def k_thousands_k(r):
    return f"{cardinal(r.randint(1, 999))} {_pick(r, [(0.6, 'K'), (0.4, 'grand')])}"


KINDS = {name[2:]: fn for name, fn in globals().items() if name.startswith("k_")}


LETTER_NAMES = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                        "ay bee see dee ee eff gee aitch eye jay kay el em en oh pee cue ar ess tee you vee "
                        "double-you ex why zee".split()))


# Acronyms said as words get a respelling for the synthesizer; other capitals are read as letters.
SPOKEN = {"EBIT": "ee bit", "EBITDA": "ee bit dah", "YoY": "why oh why", "PhD": "pee aitch dee", "CapEx": "cap ex",
          "OpEx": "op ex", "SaaS": "sass", "GAAP": "gap", "AT": "ay tee", "B2B": "bee two bee", "Ms": "Miz",
          "WiFi": "why fye"}


def tts_form(slot: str) -> str:
    """What the synthesizer is given for a slot. The target keeps `slot` unchanged; this only
    steers pronunciation (single capitals as letter names, am/pm as letters, x as 'ex')."""
    s = re.sub(r"\b(" + "|".join(SPOKEN) + r")\b", lambda m: SPOKEN[m.group(1)], slot)
    s = re.sub(r"\b([A-Z]{2,5})\b", lambda m: m.group(1) if m.group(1) in ("NASA", "NATO") else " ".join(m.group(1)), s)
    s = re.sub(r"\b([A-Z])\b", lambda m: LETTER_NAMES[m.group(1)], s)
    s = re.sub(r"\bx$", "ex", s)
    return s
