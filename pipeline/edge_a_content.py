"""Edge-case pack, tier A (financial, colloquial, disfluency) for en / es / fr: the constructions, their value universes,
and the carrier frames. Everything is written in strict spoken form (the label is what the speaker says: no digit, no
hyphen, letters as letters, letter + number as "B two B" / "Q three"). A construction is a family of edge phrases
("N basis points", "a billion N", "twenty something"); its values are the keys the readout splits (V_a / V_b) and the
construction itself is what C_a / C_b hold out. Rows that depend on an open label decision are tagged `ambig:<key>`:

  slang_value       a grand / two K / a buck fifty / ten mil: words, or a number?
  half_seven        UK 'half seven' = 7:30 (tier B, not used here)
  nine_eleven       date 9/11, emergency number or a time
  elided_hundreds   three fifty / one forty / eighty two fifty (350 / 3.50 / 3:50)
  disfluency_verbatim  both numbers of a self-correction and the filler are kept in the label
  bips_bil_mil_k    labelled as pronounced (bips, bil, mil, K), not normalised to basis points / billion
  tickers_indices   letters as letters, index names as words
  indian_grouping   (tier B, not used here)
  billon_lucas_palos (tier B, not used here)
  regional_numerals (tier B, not used here)
  paired_years      Q three twenty twenty six: a year said as two pairs
  scores            'three two', 'ten nil': two numbers, no 'to' in the label

No sentence is taken from any evaluation set: the frames are written from general knowledge.
"""
from dataclasses import dataclass, field
from typing import Callable

import edge_a_words as W
import readings as R


@dataclass
class Piece:
    label: str                      # the edge phrase as spoken
    caption: str                    # the same phrase as it would be written (digits), for the gate classes only
    nums: list = field(default_factory=list)   # [(kind, written text, reading words)] reading is a substring of label
    ambig: tuple = ()


@dataclass
class Cons:
    id: str
    lang: str
    family: str                     # fin | col | dis
    cat: str                        # frame category (FRAMES[lang][cat]) unless value_cores is given
    values: list                    # value keys (str); the readout splits these
    make: Callable                  # make(key, rng) -> Piece
    ambig: tuple = ()               # tags every row of this construction carries
    split_values: bool = True       # False: the keys are surface forms, no V_a / V_b split (all in V_a)
    value_cores: dict = field(default_factory=dict)   # key -> cores, when each value needs its own frames
    slots: int = 1                  # 2: the frames hold {X} and {Y}
    weight: float = 1.0
    key_ambig: Callable = None      # key -> extra ambig tags that depend on the value (a year said as pairs, elided hundreds)


def _c(lang):
    return lambda n: W.card(n, lang)


def _fmt(x: float) -> tuple:
    """(int part, fraction digits or '') of a decimal written with at most 3 places."""
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    a, _, b = s.partition(".")
    return int(a), b


# ================================================================================================ ENGLISH
def build_en() -> list:
    L = "en"
    c = _c(L)
    out: list = []

    def add(**kw):
        out.append(Cons(lang=L, **kw))

    def hundred(n, rng, p=0.3):
        return "a hundred" if n == 100 and rng.random() < p else c(n)

    # ---- financial
    bps = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 65, 70, 75, 80, 90, 100, 110, 120, 125, 130, 140, 150, 160, 175,
           180, 190, 200, 225, 240, 250, 275, 300, 325, 350, 375, 400, 450, 500, 600, 700, 750, 800, 900]
    def bps_make(k, rng):
        w = hundred(int(k), rng)
        return Piece(f"{w} basis points", f"{k} basis points", [("int", k, w)])
    add(id="en.fin.bps", family="fin", cat="bps", values=[str(n) for n in bps], weight=1.6, make=bps_make)
    add(id="en.fin.bips", family="fin", cat="bps", ambig=("bips_bil_mil_k",),
        values=[str(n) for n in (5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250, 300, 350, 400, 500)],
        make=lambda k, rng: Piece(f"{c(int(k))} bips", f"{k} bps", [("int", k, c(int(k)))]))

    def ellip(k, rng):  # k = "b1.2" / "m2.4"
        u = {"b": "billion", "m": "million"}[k[0]]
        a, d = k[1:].split(".")
        a = int(a)
        head = "a" if (a == 1 and rng.random() < 0.7) else c(a)
        lab = f"{head} {u} {c(int(d))}"
        return Piece(lab, f"${a}.{d} {u}", [("cur", f"${a}.{d} {u}", lab)])
    add(id="en.fin.scale_ellipsis", family="fin", cat="amt", weight=1.2,
        values=[f"{u}{a}.{d}" for u in "bm" for a in range(1, 10) for d in range(1, 10)], make=ellip)

    def bilmil(k, rng):
        if k.startswith("bil:"):
            a, d = k[4:].split(".")
            lab = f"{c(int(a))} point {c(int(d))} bil"
            return Piece(lab, f"${a}.{d}B", [("cur", f"${a}.{d}B", lab)])
        lab = f"{c(int(k[4:]))} mil"
        return Piece(lab, f"${k[4:]}M", [("cur", f"${k[4:]}M", lab)])
    add(id="en.fin.bil_mil", family="fin", cat="amt", ambig=("bips_bil_mil_k",),
        values=[f"bil:{a}.{d}" for a in range(1, 10) for d in range(1, 10)] +
               [f"mil:{n}" for n in (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200)], make=bilmil)

    def rng_bn(k, rng):  # k = "3-3.2" (billion)
        a, b = k.split("-")
        wb = (lambda x: f"{c(int(x.split('.')[0]))} point {c(int(x.split('.')[1]))}" if "." in x else c(int(x)))
        lo, hi = wb(a), wb(b)
        form = rng.choice(["{lo} to {hi} billion", "between {lo} and {hi} billion", "{lo} to {hi} billion dollars"])
        lab = form.format(lo=lo, hi=hi)
        return Piece(lab, f"${a}-{b} billion", [("int", a, lo), ("dec" if "." in b else "int", b, hi)])
    pairs = [f"{a}-{a + s:g}" for a in range(1, 21) for s in (0.2, 0.5, 1, 2)]
    add(id="en.fin.range_bn", family="fin", cat="rng", values=pairs, make=rng_bn)

    def rng_pct(k, rng):
        a, b = k.split("-")
        wb = (lambda x: f"{c(int(x.split('.')[0]))} point {c(int(x.split('.')[1]))}" if "." in x else c(int(x)))
        lo, hi = wb(a), wb(b)
        form = rng.choice(["{lo} to {hi} percent", "between {lo} and {hi} percent"])
        lab = form.format(lo=lo, hi=hi)
        return Piece(lab, f"{a}-{b}%", [("pct", f"{a}%", lo), ("pct", f"{b}%", hi)])
    pairs = [f"{a}-{a + s:g}" for a in range(1, 16) for s in (0.5, 1, 2, 3)]
    add(id="en.fin.range_pct", family="fin", cat="rngp", values=pairs, make=rng_pct)

    def eps(k, rng):  # k = "1.05"
        d, cc = k.split(".")
        d = int(d)
        if d == 0:
            amt = f"{c(int(cc))} cents"
            cap = f"{cc} cents"
        elif d == 1:
            amt = ("a dollar oh " + c(int(cc))) if int(cc) < 10 else f"a dollar {c(int(cc))}"
            cap = f"${k}"
        else:
            amt = (f"{c(d)} dollars oh {c(int(cc))}" if int(cc) < 10 else
                   rng.choice([f"{c(d)} dollars {c(int(cc))}", f"{c(d)} dollars and {c(int(cc))} cents"]))
            cap = f"${k}"
        pre = rng.choice(["E P S of ", "earnings of ", "", "profit of "])
        post = rng.choice([" a share", " per share", ""]) if pre != "E P S of " else ""
        lab = f"{pre}{amt}{post}"
        return Piece(lab, f"{pre}{cap}{post}", [("cur", cap, amt)])
    add(id="en.fin.eps_dollar", family="fin", cat="eps", weight=1.3,
        values=[f"{d}.{cc:02d}" for d in range(0, 5) for cc in (5, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95)
                if not (d == 0 and cc < 10)], make=eps)

    def neg(k, rng):
        a, b = k.split(".")
        w = f"{c(int(a)) if int(a) else 'zero'} point {c(int(b))}"
        lab = f"{rng.choice(['negative', 'minus'])} {w} percent"
        return Piece(lab, f"-{k}%", [("pct", f"-{k}%", lab)])
    add(id="en.fin.neg_pct", family="fin", cat="neg", values=[f"{a}.{b}" for a in range(0, 10) for b in range(1, 10)], make=neg)

    def mult(k, rng):
        den = rng.choice(["earnings", "forward earnings", "sales", "free cash flow", "book value", "revenue", "cash flow"])
        n = float(k)
        w = c(int(n)) if n == int(n) else f"{c(int(n))} point {c(int(round((n - int(n)) * 10)))}"
        unit = rng.choice(["x", "times"]) if n == int(n) else "times"
        lab = f"{w} {unit} {den}"
        return Piece(lab, f"{k}x {den}", [("int" if n == int(n) else "dec", k, w)])
    add(id="en.fin.multiple", family="fin", cat="mult",
        values=[str(n) for n in range(4, 41)] + [f"{n}.5" for n in (6, 8, 9, 11, 13, 15, 17, 22)], make=mult)

    def yoy(k, rng):
        suf = rng.choice(["year on year", "year over year", "Y over Y", "Q over Q", "quarter on quarter", "sequentially",
                          "from a year ago", "annualized"])
        if "." in k:
            a, b = k.split(".")
            w = f"{c(int(a)) if int(a) else 'zero'} point {c(int(b))}"
        else:
            w = c(int(k))
        lab = f"{w} percent {suf}"
        return Piece(lab, f"{k}% {suf}", [("pct", f"{k}%", f"{w} percent")])
    add(id="en.fin.yoy_pct", family="fin", cat="yoy", weight=1.3,
        values=[str(n) for n in range(1, 61)] + [f"{a}.{b}" for a in range(1, 20) for b in (2, 5, 8)], make=yoy)

    def qcode(k, rng):
        q, _, y = k.partition(":")
        w = f"Q {c(int(q))}"
        if y:
            yw = R.year(int(y))
            return Piece(f"{w} {yw}", f"Q{q} {y}", [("code", f"Q{q}", w), ("int", y, yw)])
        return Piece(w, f"Q{q}", [("code", f"Q{q}", w)])
    add(id="en.fin.q_code", family="fin", cat="qcode", key_ambig=lambda k: ("paired_years",) if ":" in k else (), split_values=True,
        values=[str(q) for q in (1, 2, 3, 4)] + [f"{q}:{y}" for q in (1, 2, 3, 4) for y in (2024, 2025, 2026, 2027)], make=qcode)
    # the 'paired_years' tag only for the keys with a year: handled in the planner (ambig_if_key)

    sec = {"10K": "ten K", "10Q": "ten Q", "8K": "eight K", "13F": "thirteen F", "S1": "S one", "20F": "twenty F",
           "6K": "six K", "F4": "form four"}
    add(id="en.fin.sec_form", family="fin", cat="sec", values=list(sec),
        make=lambda k, rng: Piece(sec[k], k, [("code", k, sec[k])]))

    names = {"sp500": ("the S and P five hundred", "the S&P 500", "500", "five hundred"),
             "r2000": ("the Russell two thousand", "the Russell 2000", "2000", "two thousand"),
             "ndx": ("the Nasdaq one hundred", "the Nasdaq 100", "100", "one hundred"),
             "dj30": ("the Dow Jones thirty", "the Dow Jones 30", "30", "thirty"),
             "ftse": ("the FTSE one hundred", "the FTSE 100", "100", "one hundred"),
             "dax": ("the Dax forty", "the Dax 40", "40", "forty"),
             "cac": ("the Cac forty", "the CAC 40", "40", "forty"),
             "nik": ("the Nikkei two twenty five", "the Nikkei 225", "225", "two twenty five")}
    add(id="en.fin.index_name", family="fin", cat="idxn", ambig=("tickers_indices",), split_values=False, values=list(names),
        make=lambda k, rng: Piece(names[k][0], names[k][1], [("int", names[k][2], names[k][3])]))

    def level(k, rng):
        t, _, v = k.partition(":")
        name = {"dow": "the Dow", "spx": "the S and P", "nas": "the Nasdaq"}[t]
        w = c(int(v))
        lab = f"{name} at {w}"
        return Piece(lab, f"{name} at {v}", [("int", v, w)])
    add(id="en.fin.index_level", family="fin", cat="lvl", ambig=("tickers_indices",),
        values=[f"dow:{v}" for v in range(30000, 45001, 400)] + [f"spx:{v}" for v in range(4000, 6501, 50)] +
               [f"nas:{v}" for v in range(13000, 20001, 250)], make=level)

    def yfx(k, rng):
        t, _, v = k.partition(":")
        if t == "y10":
            a, _, b = v.partition(".")
            w = f"{c(int(a))} point {W.frac(b, L)}" if b else c(int(a))
            lab = f"the ten year yield at {w} percent"
            return Piece(lab, f"10 year yield {v}%", [("pct", f"{v}%", f"{w} percent")])
        if t == "eurusd":
            a, b = v.split(".")
            w = f"{c(int(a))} point {W.frac(b, L)}"
            lab = f"the euro dollar at {w}"
            return Piece(lab, f"EUR/USD {v}", [("dec", v, w)])
        if t == "cable":
            a, b = v.split(".")
            w = f"{c(int(a))} point {W.frac(b, L)}"
            lab = f"cable at {w}"
            return Piece(lab, f"GBP/USD {v}", [("dec", v, w)])
        if t == "gold":
            n = int(v)
            w = f"{c(n // 100)} hundred" if n % 100 == 0 and n >= 1100 else (f"{c(n // 100)} {c(n % 100)}" if n % 100 else c(n))
            lab = f"gold at {w}"
            return Piece(lab, f"gold at {v}", [("int", v, w)])
        if t == "oil":
            d, cc = v.split(".")
            w = f"{c(int(d))} {c(int(cc))}" if int(cc) else c(int(d))
            return Piece(f"oil at {w}", f"oil at ${v}", [("cur", f"${v}", w)])
        w = f"one {c(int(v) - 100)}"  # usdjpy 152 -> "one fifty two"
        return Piece(f"dollar yen at {w}", f"USD/JPY {v}", [("int", v, w)])
    add(id="en.fin.yield_fx", family="fin", cat="lvl",
        values=[f"y10:{v / 100:.2f}".rstrip("0").rstrip(".") for v in range(300, 551, 5)] + [f"eurusd:1.{d:03d}" for d in range(30, 200, 7)] +
               [f"cable:1.{d:02d}" for d in range(20, 40)] + [f"gold:{v}" for v in range(1800, 3201, 50)] +
               [f"oil:{v}.{p}" for v in range(55, 96) for p in ("50",)] + [f"jpy:{v}" for v in range(130, 160)],
        ambig=(), make=yfx,
        key_ambig=lambda k: ("elided_hundreds",) if (k.startswith("jpy:") or k.startswith("oil:") or (k.startswith("gold:") and int(k[5:]) % 100)) else ())

    ms = (12, 17, 24, 36, 42, 45, 58, 64, 72, 85, 96, 110, 120, 135, 150, 175, 240, 360, 450, 500, 640, 720, 850)

    def money(k, rng):
        n, u = k.split(":")
        w = c(int(n))
        tail = rng.choice([" dollars", ""])
        lab = f"{w} {u}{tail}"
        return Piece(lab, f"${n} {u}", [("cur", f"${n} {u}", f"{w} {u}")])
    add(id="en.fin.money_scale", family="fin", cat="amt",
        values=[f"{n}:{u}" for n in ms for u in ("million", "billion")], make=money)

    # ---- colloquial
    cu = {"ch": "a couple hundred", "cth": "a couple thousand", "coh": "a couple of hundred", "cm": "a couple million",
          "fh": "a few hundred", "fth": "a few thousand", "cd": "a couple dozen", "fm": "a few million"}
    add(id="en.col.couple", family="col", cat="cnt", split_values=False, values=list(cu),
        make=lambda k, rng: Piece(cu[k], cu[k], [("edge", cu[k], cu[k])]))

    def something(k, rng):
        n, f = k.split(":")
        w = c(int(n))
        lab = {"s": f"{w} something", "o": f"{w} odd"}[f]
        return Piece(lab, lab, [("edge", lab, lab)])
    add(id="en.col.something_odd", family="col", cat="som",
        values=[f"{n}:s" for n in (20, 30, 40, 50, 60)] + [f"{n}:o" for n in (20, 30, 40, 50, 60, 100, 200, 300)], make=something)

    def grand(k, rng):
        n = int(k)
        lab = "a grand" if n == 1 else f"{c(n)} grand"
        return Piece(lab, lab, [("edge", lab, lab)])
    add(id="en.col.grand", family="col", cat="mon", ambig=("slang_value",),
        values=[str(n) for n in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100)], make=grand)

    def ksuf(k, rng):
        lab = f"{hundred(int(k), rng, 0.4)} K"
        return Piece(lab, f"{k}K", [("edge", f"{k}K", lab)])
    add(id="en.col.k_suffix", family="col", cat="mon", ambig=("slang_value", "bips_bil_mil_k"),
        values=[str(n) for n in (2, 3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 80, 100, 120, 150, 200)], make=ksuf)

    bk = {"a buck": "a buck", "3b": "three bucks", "5b": "five bucks", "10b": "ten bucks", "20b": "twenty bucks",
          "50b": "fifty bucks", "100b": "a hundred bucks", "bf": "a buck fifty", "tbf": "two bucks fifty", "b25": "a buck twenty five",
          "b75": "a buck seventy five", "3b50": "three bucks fifty", "b10": "a buck ten", "4b": "four bucks", "15b": "fifteen bucks"}
    add(id="en.col.buck", family="col", cat="mon", ambig=("slang_value",), values=list(bk),
        make=lambda k, rng: Piece(bk[k], bk[k], [("edge", bk[k], bk[k])]))

    dec_en = {"90": "the nineties", "80": "the eighties", "60": "the sixties", "70": "the seventies", "50": "the fifties",
              "00": "the early two thousands", "90l": "the late nineties", "80e": "the early eighties", "70m": "the mid seventies",
              "60l": "the late sixties", "90e": "the early nineties", "20": "the twenties", "30": "the thirties", "40": "the forties"}
    add(id="en.col.decade", family="col", cat="dec", values=list(dec_en),
        make=lambda k, rng: Piece(dec_en[k], dec_en[k], [("decade", dec_en[k], dec_en[k])]))

    add(id="en.col.nine_eleven", family="col", cat="n11", ambig=("nine_eleven",), split_values=False, values=["911d", "911c"],
        make=lambda k, rng: Piece({"911d": "nine eleven", "911c": "nine one one"}[k], "9/11", [("edge", "9/11", {"911d": "nine eleven", "911c": "nine one one"}[k])]),
        value_cores={"911d": ["everything changed after {X}", "she still remembers where she was on {X}",
                              "the commission that investigated {X} published its report", "airport security was never the same after {X}",
                              "he joined the army right after {X}", "a whole generation grew up in the shadow of {X}"],
                     "911c": ["if there is an emergency, call {X}", "the dispatcher who answered the {X} call stayed on the line",
                              "she dialed {X} as soon as the car stopped", "do not hang up when you reach {X}",
                              "the {X} operator sent an ambulance within minutes", "he had to call {X} twice before anyone came"]})

    idi = {"247": ("twenty four seven", ["the store is open {X}", "our support team works {X}", "the system runs {X} without anyone watching"]),
           "5050": ("fifty fifty", ["the odds are about {X}", "they split the profit {X}", "it was a {X} chance, nothing more"]),
           "95": ("nine to five", ["I have a regular {X} job now", "nobody works {X} anymore", "she left the {X} world behind"]),
           "4x4": ("four by four", ["he drives a {X} through the mountains", "we rented a {X} for the trip", "the old {X} still starts every morning"]),
           "2f1": ("two for one", ["the pizza place has a {X} deal on Tuesdays", "everyone loves a {X} offer", "there was a {X} sale at the hardware store"]),
           "1o1": ("one on one", ["we had a {X} meeting on Monday", "she prefers {X} coaching", "the manager scheduled a {X} with each person"]),
           "360": ("three sixty", ["the team ran a {X} review of the project", "they asked for {X} feedback", "the camera gives you a {X} view"])}
    add(id="en.col.idiom_num", family="col", cat="idi", split_values=False, values=list(idi),
        make=lambda k, rng: Piece(idi[k][0], idi[k][0], [("edge", idi[k][0], idi[k][0])]),
        value_cores={k: v[1] for k, v in idi.items()})

    sc = {"3-2": "three two", "10-0": "ten nil", "40-L": "forty love", "2-0": "two nil", "1-1": "one all", "21-14": "twenty one fourteen",
          "6-4": "six four", "4-1": "four one", "0-0": "nil nil", "5-3": "five three", "3-0": "three nil", "24-17": "twenty four seventeen",
          "7-6": "seven six", "2-1": "two one", "98-95": "ninety eight ninety five"}
    add(id="en.col.score", family="col", cat="score", ambig=("scores",), values=list(sc),
        make=lambda k, rng: Piece(sc[k], k, [("edge", k, sc[k])]))

    el = {"350": "three fifty", "140": "one forty", "275": "two seventy five", "420": "four twenty", "550": "five fifty",
          "125": "one twenty five", "865": "eight sixty five", "395": "three ninety five", "225": "two twenty five",
          "175": "one seventy five", "650": "six fifty", "475": "four seventy five"}
    add(id="en.col.elided_hundreds", family="col", cat="eli", ambig=("elided_hundreds",), values=list(el),
        make=lambda k, rng: Piece(el[k], k, [("int", k, el[k])]))

    qt = {"dz": ("a dozen", ["{X} people were waiting outside", "she bought {X} eggs", "I have asked {X} times"]),
          "dzs": ("dozens of", ["{X} people signed the petition", "there are {X} options on the menu", "{X} flights were cancelled"]),
          "dd": ("double digits", ["inflation is finally out of {X}", "the team has scored in {X} in every game", "growth stayed in {X} for three years"]),
          "td": ("triple digits", ["temperatures hit {X} all week", "the stock is back in {X}", "the heat index reached {X} on Sunday"]),
          "hth": ("hundreds of thousands", ["{X} of people lost power", "the campaign reached {X} of voters", "{X} of tourists visit every year"]),
          "ho": ("a hundred and one", ["there are {X} ways to cook rice", "we had {X} things to do before the trip", "the list had {X} names on it"]),
          "tho": ("a thousand and one", ["he told me {X} reasons not to go", "the shop sells {X} little gadgets", "there are {X} questions left to answer"]),
          "mil": ("millions of", ["{X} people watched the final", "{X} dollars were spent on the campaign", "{X} songs are uploaded every week"])}
    add(id="en.col.quantifier", family="col", cat="qty", split_values=False, values=list(qt),
        make=lambda k, rng: Piece(qt[k][0], qt[k][0], [("edge", qt[k][0], qt[k][0])]),
        value_cores={k: v[1] for k, v in qt.items()})

    # ---- disfluency / self-correction (verbatim; both numbers kept)
    A = ("disfluency_verbatim",)
    up = [(20, 30), (15, 25), (40, 50), (100, 150), (200, 300), (5, 10), (60, 70), (12, 15), (80, 90), (30, 40), (25, 35), (18, 20)]

    def uhpair(k, rng):
        a, b = map(int, k.split("-"))
        f = rng.choice(["uh", "um", "uh"])   # "er" is read as "twentyer" (bulk pilot: 0 of 8 accepted)
        lab = f"{c(a)} {f} {c(b)}"
        return Piece(lab, f"{a}-{b}", [("int", str(a), c(a)), ("int", str(b), c(b))], A)
    add(id="en.dis.uh_pair", family="dis", cat="dis", ambig=A, values=[f"{a}-{b}" for a, b in up], make=uhpair)

    fixes = [(15, 50), (13, 30), (16, 60), (14, 40), (17, 70), (18, 80), (19, 90), (22, 12), (30, 13), (60, 16), (25, 52), (45, 54)]

    def fix(k, rng):
        a, b = map(int, k.split("-"))
        f = rng.choice(["sorry", "I mean", "no wait", "or rather", "sorry"])
        lab = f"{c(a)} {f} {c(b)}"
        return Piece(lab, f"{a}-{b}", [("int", str(a), c(a)), ("int", str(b), c(b))], A)
    add(id="en.dis.self_fix", family="dis", cat="dis", ambig=A, values=[f"{a}-{b}" for a, b in fixes], make=fix)

    def umbefore(k, rng):
        n = int(k)
        f = rng.choice(["um", "uh"])
        lab = f"about {f} {c(n)} percent"
        return Piece(lab, f"about {n}%", [("pct", f"{n}%", f"{c(n)} percent")], A)
    add(id="en.dis.filler_before", family="dis", cat="disp", ambig=A, values=[str(n) for n in (5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 75, 80)], make=umbefore)

    def hundred_uh(k, rng):
        h, t = map(int, k.split("-"))
        lab = f"{c(h)} hundred and uh {c(t)} million"
        return Piece(lab, f"{h}{t:02d} million", [("int", f"{h}{t:02d}", f"{c(h)} hundred and uh {c(t)}")], A)
    add(id="en.dis.number_pause", family="dis", cat="disa", ambig=A, values=[f"{h}-{t}" for h in (2, 3, 4, 5, 6, 7, 8) for t in (20, 40, 50, 60, 75, 80)], make=hundred_uh)
    return out


# ================================================================================================ SPANISH
def build_es() -> list:
    L = "es"
    c = _c(L)
    out: list = []

    def add(**kw):
        out.append(Cons(lang=L, **kw))

    ok = lambda ns: [n for n in ns if n % 10 != 1]  # a value ending in 1 takes the apocope ("veintiún"): left out

    def dec_es(k):
        a, _, b = k.partition(".")
        whole = W.DIGIT['es'][0] if int(a) == 0 else c(int(a))
        return f"{whole} coma {W.frac(b, L)}" if b else whole

    bps = ok([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 65, 70, 75, 80, 90, 100, 110, 120, 125, 130, 140, 150, 160, 175,
              180, 190, 200, 225, 240, 250, 275, 300, 325, 350, 375, 400, 450, 500, 600, 700, 750, 800, 900])
    add(id="es.fin.bps", family="fin", cat="bps", values=[str(n) for n in bps], weight=1.6,
        make=lambda k, rng: Piece(f"{c(int(k))} puntos {rng.choice(['básicos', 'básicos', 'base'])}", f"{k} pb", [("int", k, c(int(k)))]))

    def big(k, rng):  # k = "3200" (millones) or "m1.2" -> "mil doscientos millones"
        n = int(k)
        w = c(n)
        tail = rng.choice([" de euros", " de dólares", " de euros"])
        lab = f"{w} millones{tail}"
        return Piece(lab, f"{n} millones{tail.replace(' de ', ' ')}", [("int", k, w)])
    add(id="es.fin.mil_millones", family="fin", cat="amt", weight=1.6,
        values=[str(n) for n in ok(list(range(1000, 6001, 100)) + [8000, 9000, 10000, 12000, 15000, 20000, 25000, 30000, 38000, 45000, 50000])
                if n % 1000 or True], make=big)

    def money(k, rng):
        n, u = k.split(":")
        w = c(int(n))
        un = {"m": "millones", "mm": "millones"}[u]
        tail = rng.choice([" de euros", " de dólares"])
        lab = f"{w} {un}{tail}"
        return Piece(lab, f"{n} millones{tail}", [("int", n, w)])
    add(id="es.fin.money_scale", family="fin", cat="amt",
        values=[f"{n}:m" for n in ok([12, 17, 24, 36, 42, 45, 58, 64, 72, 85, 96, 110, 120, 135, 150, 175, 240, 360, 450, 500, 640, 720, 850])], make=money)

    def rng_amt(k, rng):  # "3000-3200" in millones
        a, b = k.split("-")
        lo, hi = c(int(a)), c(int(b))
        form = rng.choice(["de entre {lo} y {hi} millones de euros", "de entre {lo} y {hi} millones de euros", "de {lo} a {hi} millones de dólares"])
        lab = form.format(lo=lo, hi=hi)
        return Piece(lab, f"{a}-{b} millones", [("int", a, lo), ("int", b, hi)])
    prs = [f"{a}-{a + s}" for a in range(1000, 6001, 400) for s in (100, 200, 400, 500, 1000)]
    prs = [p for p in prs if int(p.split("-")[0]) % 10 != 1 and int(p.split("-")[1]) % 10 != 1]
    add(id="es.fin.range_amt", family="fin", cat="rng", values=prs, make=rng_amt)

    def rng_pct(k, rng):
        a, b = k.split("-")
        wb = lambda x: dec_es(x) if "." in x else c(int(x))
        lo, hi = wb(a), wb(b)
        lab = rng.choice(["de entre {lo} y {hi} por ciento", "de {lo} a {hi} por ciento"]).format(lo=lo, hi=hi)
        return Piece(lab, f"{a}-{b}%", [("pct", f"{a}%", lo), ("pct", f"{b}%", hi)])
    prs = [f"{a}-{a + s:g}" for a in range(2, 16) for s in (0.5, 1, 2, 3) if a % 10 != 1 and int(a + s) % 10 != 1]
    add(id="es.fin.range_pct", family="fin", cat="rngp", values=prs, make=rng_pct)

    def eps(k, rng):
        d, cc = k.split(".")
        d, cc = int(d), int(cc)
        if d == 0:
            amt = f"{c(cc)} céntimos"
            cap = f"{cc} céntimos"
        else:
            eur = "un euro" if d == 1 else f"{c(d)} euros"
            amt = f"{eur} con {c(cc)}"
            cap = f"{k} euros"
        pre = rng.choice(["un beneficio por acción de ", "unos beneficios de ", "unas ganancias por acción de "])
        lab = f"{pre}{amt}"
        return Piece(lab, f"{pre}{cap}", [("cur", cap, amt)])
    add(id="es.fin.eps_euro", family="fin", cat="eps", weight=1.2,
        values=[f"{d}.{cc:02d}" for d in range(0, 5) for cc in (5, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95)
                if not (d == 0 and cc < 10) and (cc % 10 != 1)], make=eps)

    def neg(k, rng):
        w = dec_es(k)
        lab = f"menos {w} por ciento"
        return Piece(lab, f"-{k}%", [("pct", f"-{k}%", lab)])
    add(id="es.fin.neg_pct", family="fin", cat="neg", values=[f"{a}.{b}" for a in range(0, 10) for b in range(1, 10)], make=neg)

    def mult(k, rng):
        den = rng.choice(["beneficios", "los beneficios", "ventas", "el flujo de caja", "el valor contable", "los ingresos"])
        n = float(k)
        w = c(int(n)) if n == int(n) else f"{c(int(n))} coma {c(int(round((n - int(n)) * 10)))}"
        lab = f"{w} veces {den}"
        return Piece(lab, f"{k}x {den}", [("int" if n == int(n) else "dec", k, w)])
    add(id="es.fin.multiple", family="fin", cat="mult",
        values=[str(n) for n in ok(range(4, 41))] + [f"{n}.5" for n in (6, 8, 9, 12, 13, 15, 17, 22)], make=mult)

    def yoy(k, rng):
        suf = rng.choice(["interanual", "en tasa interanual", "respecto al año anterior", "trimestre a trimestre", "intertrimestral",
                          "frente al año pasado", "anualizado"])
        w = dec_es(k) if "." in k else c(int(k))
        lab = f"un {w} por ciento {suf}"
        return Piece(lab, f"{k}% {suf}", [("pct", f"{k}%", f"{w} por ciento")])
    add(id="es.fin.yoy_pct", family="fin", cat="yoy", weight=1.3,
        values=[str(n) for n in ok(range(1, 61))] + [f"{a}.{b}" for a in range(1, 20) for b in (2, 5, 8) if a % 10 != 1], make=yoy)

    def qcode(k, rng):
        w = f"Q {c(int(k))}"
        return Piece(w, f"Q{k}", [("code", f"Q{k}", w)])
    add(id="es.fin.q_code", family="fin", cat="qcode", values=["1", "2", "3", "4"], make=qcode)

    fixed = {"ibex": ("el Ibex treinta y cinco", "el Ibex 35", "35", "treinta y cinco"),
             "euro50": ("el Eurostoxx cincuenta", "el Eurostoxx 50", "50", "cincuenta"),
             "dax": ("el Dax cuarenta", "el Dax 40", "40", "cuarenta"),
             "sp500": ("el S y P quinientos", "el S&P 500", "500", "quinientos"),
             "cac": ("el Cac cuarenta", "el CAC 40", "40", "cuarenta"),
             "ftse": ("el Footsie cien", "el FTSE 100", "100", "cien")}
    add(id="es.fin.index_name", family="fin", cat="idxn", ambig=("tickers_indices",), split_values=False, values=list(fixed),
        make=lambda k, rng: Piece(fixed[k][0], fixed[k][1], [("int", fixed[k][2], fixed[k][3])]))

    def level(k, rng):
        t, _, v = k.partition(":")
        name = {"dow": "el Dow Jones", "spx": "el S y P", "nas": "el Nasdaq", "ibexl": "el Ibex"}[t]
        w = c(int(v))
        return Piece(f"{name} en {w} puntos", f"{name} en {v} puntos", [("int", v, w)])
    add(id="es.fin.index_level", family="fin", cat="lvl", ambig=("tickers_indices",),
        values=[f"dow:{v}" for v in ok(range(30000, 45001, 400))] + [f"spx:{v}" for v in ok(range(4000, 6501, 50))] +
               [f"ibexl:{v}" for v in ok(range(9000, 12501, 50))], make=level)

    def yfx(k, rng):
        t, _, v = k.partition(":")
        if t == "y10":
            w = dec_es(v)
            return Piece(f"el bono a diez años en {w} por ciento", f"bono 10 años {v}%", [("pct", f"{v}%", f"{w} por ciento")])
        if t == "eurusd":
            a, b = v.split(".")
            w = f"{c(int(a))} coma {W.frac(b, L)}"
            return Piece(f"el euro dólar en {w}", f"EUR/USD {v}", [("dec", v, w)])
        if t == "gold":
            w = c(int(v))
            return Piece(f"el oro a {w} dólares la onza", f"oro {v} dólares", [("cur", f"${v}", w)])
        if t == "oil":
            w = c(int(v))
            return Piece(f"el petróleo a {w} dólares el barril", f"petróleo {v} dólares", [("cur", f"${v}", w)])
        w = c(int(v))
        return Piece(f"el dólar yen en {w}", f"USD/JPY {v}", [("int", v, w)])
    add(id="es.fin.yield_fx", family="fin", cat="lvl",
        values=[f"y10:{v / 100:.2f}".rstrip("0").rstrip(".") for v in range(300, 551, 5)] + [f"eurusd:1.{d:03d}" for d in range(30, 200, 7)] +
               [f"gold:{v}" for v in ok(range(1800, 3201, 50))] + [f"oil:{v}" for v in ok(range(55, 96))] + [f"jpy:{v}" for v in ok(range(130, 160))],
        make=yfx)

    # ---- colloquial
    pc = {"pc": "un par de cientos de", "pm": "un par de miles de", "um": "unos mil", "ud": "unos doscientos", "ct": "cientos de",
          "mi": "miles de", "de": "decenas de", "cn": "centenares de", "mm": "un par de millones de", "utr": "unos trescientos",
          "uc": "unos quinientos", "uo": "unos ochocientos"}
    add(id="es.col.par_de", family="col", cat="cnt", split_values=False, values=list(pc),
        make=lambda k, rng: Piece(pc[k], pc[k], [("edge", pc[k], pc[k])]))

    def pico(k, rng):
        n, f = k.split(":")
        w = c(int(n))
        lab = {"p": f"{w} y pico", "t": f"{w} y tantos", "o": f"{w} y pocos"}[f] if f != "v" else f"{w}pico"
        if f == "v":
            lab = {"20": "veintipico", "30": "treintaipico"}.get(n, f"{w} y pico")
        return Piece(lab, lab, [("edge", lab, lab)])
    add(id="es.col.pico_tantos", family="col", cat="pic",
        values=[f"{n}:p" for n in (20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 1000)] + [f"{n}:t" for n in (30, 40, 50, 60, 70, 80)] +
               [f"{n}:o" for n in (30, 40, 50, 60)] + ["20:v"], make=pico)

    dec = {"90": "los noventa", "80": "los ochenta", "60": "los sesenta", "70": "los setenta", "50": "los cincuenta",
           "90a": "los años noventa", "80a": "los años ochenta", "60a": "los años sesenta", "20": "los veinte", "30": "los treinta"}
    add(id="es.col.decada", family="col", cat="dec", values=list(dec),
        make=lambda k, rng: Piece(dec[k], dec[k], [("decade", dec[k], dec[k])]))

    idi = {"c100": ("cien por cien", ["estoy {X} seguro de que vendrá", "el material es {X} natural", "lo que dijo es {X} cierto"]),
           "5050": ("cincuenta cincuenta", ["las posibilidades están {X}", "repartieron las ganancias {X}", "es un {X}, nada más"]),
           "247": ("veinticuatro siete", ["la tienda abre {X}", "nuestro equipo de soporte trabaja {X}", "el sistema funciona {X} sin que nadie lo vigile"]),
           "11": ("uno a uno", ["revisamos las cuentas {X}", "el entrenador habla con los jugadores {X}", "repasaron los errores {X}"]),
           "32": ("tres por dos", ["en el súper hay una oferta de {X} en yogures", "a todos les gusta el {X}", "la tienda tenía un {X} en camisetas"]),
           "1010": ("diez de diez", ["la película fue un {X}", "el servicio merece un {X}", "le pusieron un {X} a la comida"])}
    add(id="es.col.modismo_num", family="col", cat="idi", split_values=False, values=list(idi),
        make=lambda k, rng: Piece(idi[k][0], idi[k][0], [("edge", idi[k][0], idi[k][0])]), value_cores={k: v[1] for k, v in idi.items()})

    sc = {"3-2": "tres a dos", "0-0": "cero a cero", "2-1": "dos a uno", "4-0": "cuatro a cero", "1-1": "uno a uno", "5-3": "cinco a tres",
          "3-0": "tres a cero", "2-2": "dos a dos", "1-0": "uno a cero", "4-2": "cuatro a dos", "6-1": "seis a uno", "2-0": "dos a cero"}
    add(id="es.col.marcador", family="col", cat="score", ambig=("scores",), values=list(sc),
        make=lambda k, rng: Piece(sc[k], k, [("edge", k, sc[k])]))

    qt = {"doc": "una docena de", "dec": "una decena de", "vei": "una veintena de", "tre": "una treintena de", "cua": "una cuarentena de",
          "cen": "un centenar de", "mil": "un millar de", "med": "media docena de", "cin": "una cincuentena de", "sese": "una sesentena de"}
    add(id="es.col.cuantificador", family="col", cat="cnt", values=list(qt), split_values=False,
        make=lambda k, rng: Piece(qt[k], qt[k], [("edge", qt[k], qt[k])]))

    # ---- disfluency
    A = ("disfluency_verbatim",)

    def uh(k, rng):
        a, b = map(int, k.split("-"))
        f = "este"   # "eh" and "mmm" are not voiced by the engines (pilot: judges write "y" / "de" / nothing)
        lab = f"{c(a)} {f} {c(b)}"
        return Piece(lab, f"{a}-{b}", [("int", str(a), c(a)), ("int", str(b), c(b))], A)
    pr = [(20, 30), (15, 25), (40, 50), (100, 150), (200, 300), (5, 10), (60, 70), (12, 15), (80, 90), (30, 40), (25, 35), (18, 20)]
    add(id="es.dis.eh_pair", family="dis", cat="dis", ambig=A, values=[f"{a}-{b}" for a, b in pr], make=uh)

    def fix(k, rng):
        a, b = map(int, k.split("-"))
        f = rng.choice(["perdón", "o sea", "no espera", "mejor dicho", "perdón"])
        lab = f"{c(a)} {f} {c(b)}"
        return Piece(lab, f"{a}-{b}", [("int", str(a), c(a)), ("int", str(b), c(b))], A)
    fx = [(15, 50), (13, 30), (16, 60), (14, 40), (17, 70), (18, 80), (19, 90), (22, 12), (30, 13), (60, 16), (25, 52), (45, 54)]
    add(id="es.dis.autocorreccion", family="dis", cat="dis", ambig=A, values=[f"{a}-{b}" for a, b in fx], make=fix)

    def um(k, rng):
        n = int(k)
        f = "este"
        lab = f"alrededor de {f} {c(n)} por ciento"
        return Piece(lab, f"alrededor de {n}%", [("pct", f"{n}%", f"{c(n)} por ciento")], A)
    add(id="es.dis.relleno", family="dis", cat="disp", ambig=A, values=[str(n) for n in (5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 75, 80)], make=um)
    return out


# ================================================================================================ FRENCH
def build_fr() -> list:
    L = "fr"
    c = _c(L)
    out: list = []

    def add(**kw):
        out.append(Cons(lang=L, **kw))

    def dec_fr(k):
        a, _, b = k.partition(".")
        whole = W.DIGIT['fr'][0] if int(a) == 0 else c(int(a))
        return f"{whole} virgule {W.frac(b, L)}" if b else whole

    bps = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 65, 70, 75, 80, 90, 100, 110, 120, 125, 130, 140, 150, 160, 175,
           180, 190, 200, 225, 240, 250, 275, 300, 325, 350, 375, 400, 450, 500, 600, 700, 750, 800, 900]
    add(id="fr.fin.bps", family="fin", cat="bps", values=[str(n) for n in bps], weight=1.6,
        make=lambda k, rng: Piece(f"{c(int(k))} points de base", f"{k} pb", [("int", k, c(int(k)))]))

    def ellip(k, rng):  # k = "b1.2" / "m2.4"
        u = {"b": ("milliard", "milliards"), "m": ("million", "millions")}[k[0]]
        a, d = k[1:].split(".")
        a = int(a)
        head = "un" if a == 1 else c(a)
        lab = f"{head} {u[0] if a == 1 else u[1]} {c(int(d))} d'euros"
        return Piece(lab, f"{a}.{d} {u[1]} d'euros", [("cur", f"{a}.{d} {u[1]}", lab)])
    add(id="fr.fin.echelle_ellipse", family="fin", cat="amt", weight=1.2,
        values=[f"{u}{a}.{d}" for u in "bm" for a in range(1, 10) for d in range(2, 10)], make=ellip)

    def big(k, rng):  # "3:200" -> trois milliards deux cents millions
        mi, mm = k.split(":")
        lab = f"{c(int(mi))} milliards {c(int(mm))} millions d'euros"
        return Piece(lab, f"{mi}.{mm} milliards", [("int", mi, c(int(mi))), ("int", mm, c(int(mm)))])
    add(id="fr.fin.milliards_millions", family="fin", cat="amt",
        values=[f"{a}:{b}" for a in range(2, 20) for b in (100, 200, 300, 400, 500, 600, 700, 800, 900)], make=big)

    def money(k, rng):
        n, u = k.split(":")
        w = c(int(n))
        un = {"m": "millions", "b": "milliards"}[u]
        tail = rng.choice(["d'euros", "de dollars", "d'euros"])
        lab = f"{w} {un} {tail}"
        return Piece(lab, f"{n} {un} {tail}", [("int", n, w)])
    ms = [12, 17, 24, 36, 42, 45, 58, 64, 72, 85, 96, 110, 120, 135, 150, 175, 240, 360, 450, 500, 640, 720, 850]
    add(id="fr.fin.money_scale", family="fin", cat="amt", values=[f"{n}:m" for n in ms] + [f"{n}:b" for n in ms if n < 200], make=money)

    def rng_amt(k, rng):
        a, b = k.split("-")
        wb = lambda x: dec_fr(x) if "." in x else c(int(x))
        lo, hi = wb(a), wb(b)
        lab = rng.choice(["entre {lo} et {hi} milliards d'euros", "entre {lo} et {hi} milliards de dollars", "de {lo} à {hi} milliards d'euros"]).format(lo=lo, hi=hi)
        return Piece(lab, f"{a}-{b} milliards", [("int", a, lo), ("dec" if "." in b else "int", b, hi)])
    add(id="fr.fin.fourchette_md", family="fin", cat="rng", values=[f"{a}-{a + s:g}" for a in range(1, 21) for s in (0.2, 0.5, 1, 2)], make=rng_amt)

    def rng_pct(k, rng):
        a, b = k.split("-")
        wb = lambda x: dec_fr(x) if "." in x else c(int(x))
        lo, hi = wb(a), wb(b)
        lab = rng.choice(["entre {lo} et {hi} pour cent", "de {lo} à {hi} pour cent"]).format(lo=lo, hi=hi)
        return Piece(lab, f"{a}-{b}%", [("pct", f"{a}%", lo), ("pct", f"{b}%", hi)])
    add(id="fr.fin.fourchette_pct", family="fin", cat="rngp", values=[f"{a}-{a + s:g}" for a in range(1, 16) for s in (0.5, 1, 2, 3)], make=rng_pct)

    def eps(k, rng):
        d, cc = k.split(".")
        d, cc = int(d), int(cc)
        if d == 0:
            amt, cap = f"{c(cc)} centimes", f"{cc} centimes"
        elif d == 1:
            amt, cap = f"un euro {c(cc)}", f"{k} euros"
        else:
            amt, cap = f"{c(d)} euros {c(cc)}", f"{k} euros"
        pre = rng.choice(["un bénéfice par action ", "un bénéfice "])
        lab = pre + ("d'" if amt.startswith("un ") else "de ") + amt
        return Piece(lab, f"{pre}{cap}", [("cur", cap, amt)])
    add(id="fr.fin.bpa_euros", family="fin", cat="eps", weight=1.2,
        values=[f"{d}.{cc:02d}" for d in range(0, 5) for cc in (10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95) if not (d == 0 and cc < 10)],
        make=eps)

    def neg(k, rng):
        w = dec_fr(k)
        lab = f"moins {w} pour cent"
        return Piece(lab, f"-{k}%", [("pct", f"-{k}%", lab)])
    add(id="fr.fin.neg_pct", family="fin", cat="neg", values=[f"{a}.{b}" for a in range(0, 10) for b in range(1, 10)], make=neg)

    def mult(k, rng):
        den = rng.choice(["les bénéfices", "les bénéfices attendus", "le chiffre d'affaires", "les flux de trésorerie", "la valeur comptable"])
        n = float(k)
        w = c(int(n)) if n == int(n) else f"{c(int(n))} virgule {c(int(round((n - int(n)) * 10)))}"
        lab = f"{w} fois {den}"
        return Piece(lab, f"{k}x {den}", [("int" if n == int(n) else "dec", k, w)])
    add(id="fr.fin.multiple", family="fin", cat="mult", values=[str(n) for n in range(4, 41)] + [f"{n}.5" for n in (6, 8, 9, 12, 13, 15, 17, 22)], make=mult)

    def yoy(k, rng):
        suf = rng.choice(["sur un an", "en glissement annuel", "par rapport à l'an dernier", "d'un trimestre à l'autre", "en rythme annualisé",
                          "sur douze mois"])
        w = dec_fr(k) if "." in k else c(int(k))
        lab = f"{w} pour cent {suf}"
        return Piece(lab, f"{k}% {suf}", [("pct", f"{k}%", f"{w} pour cent")])
    add(id="fr.fin.yoy_pct", family="fin", cat="yoy", weight=1.3,
        values=[str(n) for n in range(1, 61)] + [f"{a}.{b}" for a in range(1, 20) for b in (2, 5, 8)], make=yoy)

    def qcode(k, rng):
        w = f"T {c(int(k))}"
        return Piece(w, f"T{k}", [("code", f"T{k}", w)])
    add(id="fr.fin.t_code", family="fin", cat="qcode", values=["1", "2", "3", "4"], make=qcode)

    fixed = {"cac": ("le CAC quarante", "le CAC 40", "40", "quarante"), "euro50": ("l'Euro Stoxx cinquante", "l'Euro Stoxx 50", "50", "cinquante"),
             "dax": ("le Dax quarante", "le Dax 40", "40", "quarante"), "sp500": ("le S et P cinq cents", "le S&P 500", "500", "cinq cents"),
             "ftse": ("le Footsie cent", "le FTSE 100", "100", "cent")}
    add(id="fr.fin.indice_nom", family="fin", cat="idxn", ambig=("tickers_indices",), split_values=False, values=list(fixed),
        make=lambda k, rng: Piece(fixed[k][0], fixed[k][1], [("int", fixed[k][2], fixed[k][3])]))

    def level(k, rng):
        t, _, v = k.partition(":")
        name = {"dow": "le Dow Jones", "spx": "le S et P", "nas": "le Nasdaq", "cacl": "le CAC"}[t]
        w = c(int(v))
        return Piece(f"{name} à {w} points", f"{name} à {v} points", [("int", v, w)])
    add(id="fr.fin.indice_niveau", family="fin", cat="lvl", ambig=("tickers_indices",),
        values=[f"dow:{v}" for v in range(30000, 39001, 400)] + [f"spx:{v}" for v in range(4000, 6501, 50)] +
               [f"cacl:{v}" for v in range(6500, 8501, 50)], make=level)

    def yfx(k, rng):
        t, _, v = k.partition(":")
        if t == "y10":
            w = dec_fr(v)
            return Piece(f"le taux à dix ans à {w} pour cent", f"taux 10 ans {v}%", [("pct", f"{v}%", f"{w} pour cent")])
        if t == "eurusd":
            a, b = v.split(".")
            w = f"{c(int(a))} virgule {W.frac(b, L)}"
            return Piece(f"l'euro dollar à {w}", f"EUR/USD {v}", [("dec", v, w)])
        if t == "gold":
            w = c(int(v))
            return Piece(f"l'or à {w} dollars l'once", f"or {v} dollars", [("cur", f"${v}", w)])
        if t == "oil":
            w = c(int(v))
            return Piece(f"le baril à {w} dollars", f"baril {v} dollars", [("cur", f"${v}", w)])
        w = c(int(v))
        return Piece(f"le dollar yen à {w}", f"USD/JPY {v}", [("int", v, w)])
    add(id="fr.fin.taux_change", family="fin", cat="lvl",
        values=[f"y10:{v / 100:.2f}".rstrip("0").rstrip(".") for v in range(300, 551, 5)] + [f"eurusd:1.{d:03d}" for d in range(30, 200, 7)] +
               [f"gold:{v}" for v in range(1800, 3201, 50)] + [f"oil:{v}" for v in range(55, 96)] + [f"jpy:{v}" for v in range(130, 160)], make=yfx)

    # ---- colloquial
    dz = {"dz": "une dizaine de", "qz": "une quinzaine de", "vg": "une vingtaine de", "tr": "une trentaine de", "qr": "une quarantaine de",
          "cq": "une cinquantaine de", "sx": "une soixantaine de", "ct": "une centaine de", "ml": "un millier de", "dzn": "une douzaine de",
          "dct": "des centaines de", "dml": "des milliers de"}
    add(id="fr.col.dizaine", family="col", cat="cnt", split_values=False, values=list(dz),
        make=lambda k, rng: Piece(dz[k], dz[k], [("edge", dz[k], dz[k])]))
    qq = {"qc": "quelques centaines de", "qm": "quelques milliers de", "qd": "quelques dizaines de", "pm": "un petit millier de",
          "pc": "plusieurs centaines de", "pml": "plusieurs milliers de", "ac": "à peine une centaine de", "ad": "à peine une dizaine de"}
    add(id="fr.col.quelques", family="col", cat="cnt", split_values=False, values=list(qq),
        make=lambda k, rng: Piece(qq[k], qq[k], [("edge", qq[k], qq[k])]))

    def equ(k, rng):
        n = int(k)
        lab = f"{c(n)} et quelques"
        return Piece(lab, lab, [("edge", lab, lab)])
    add(id="fr.col.et_quelques", family="col", cat="eq", values=[str(n) for n in (20, 30, 40, 50, 60, 100, 200, 1000)], make=equ)

    dec = {"80": "les années quatre vingt", "90": "les années quatre vingt dix", "60": "les années soixante", "00": "les années deux mille",
           "50": "les années cinquante", "70": "les années soixante dix", "30": "les années trente", "20": "les années vingt", "10": "les années dix"}
    add(id="fr.col.annees", family="col", cat="dec", values=list(dec),
        make=lambda k, rng: Piece(dec[k], dec[k], [("decade", dec[k], dec[k])]))

    idi = {"2424": ("vingt quatre heures sur vingt quatre", ["la boutique est ouverte {X}", "notre équipe d'assistance travaille {X}", "le système tourne {X} sans surveillance"]),
           "77": ("sept jours sur sept", ["le magasin ouvre {X}", "les urgences fonctionnent {X}", "il s'occupe de ses chevaux {X}"]),
           "100": ("cent pour cent", ["je suis {X} sûr qu'il viendra", "le produit est {X} naturel", "ce qu'elle a dit est {X} vrai"]),
           "5050": ("cinquante cinquante", ["les chances sont {X}", "ils se sont partagé les gains {X}", "c'est un {X}, rien de plus"]),
           "11": ("un contre un", ["le coach discute {X} avec chaque joueur", "ils ont repris les erreurs {X}", "la réunion se fait {X}"]),
           "22": ("deux par deux", ["les enfants sont entrés {X}", "ils ont rangé les chaises {X}", "les élèves marchent {X} dans la rue"])}
    add(id="fr.col.locution_num", family="col", cat="idi", split_values=False, values=list(idi),
        make=lambda k, rng: Piece(idi[k][0], idi[k][0], [("edge", idi[k][0], idi[k][0])]), value_cores={k: v[1] for k, v in idi.items()})

    sc = {"3-2": "trois à deux", "0-0": "zéro à zéro", "1-1": "un partout", "2-2": "deux partout", "4-0": "quatre à zéro", "2-1": "deux à un",
          "15-0": "quinze zéro", "30-15": "trente quinze", "40-30": "quarante trente", "5-3": "cinq à trois", "1-0": "un à zéro", "3-1": "trois à un"}
    add(id="fr.col.score", family="col", cat="score", ambig=("scores",), values=list(sc),
        make=lambda k, rng: Piece(sc[k], k, [("edge", k, sc[k])]))

    # ---- disfluency
    A = ("disfluency_verbatim",)

    def uh(k, rng):
        a, b = map(int, k.split("-"))
        f = rng.choice(["euh", "euh", "heu", "hum"])
        lab = f"{c(a)} {f} {c(b)}"
        return Piece(lab, f"{a}-{b}", [("int", str(a), c(a)), ("int", str(b), c(b))], A)
    pr = [(20, 30), (15, 25), (40, 50), (100, 150), (200, 300), (5, 10), (60, 70), (12, 15), (80, 90), (30, 40), (25, 35), (18, 20)]
    add(id="fr.dis.euh_pair", family="dis", cat="dis", ambig=A, values=[f"{a}-{b}" for a, b in pr], make=uh)

    def fix(k, rng):
        a, b = map(int, k.split("-"))
        f = rng.choice(["pardon", "enfin", "non attends", "je veux dire", "pardon"])
        lab = f"{c(a)} {f} {c(b)}"
        return Piece(lab, f"{a}-{b}", [("int", str(a), c(a)), ("int", str(b), c(b))], A)
    fx = [(15, 50), (13, 30), (16, 60), (14, 40), (17, 70), (18, 80), (19, 90), (22, 12), (30, 13), (60, 16), (25, 52), (45, 54)]
    add(id="fr.dis.autocorrection", family="dis", cat="dis", ambig=A, values=[f"{a}-{b}" for a, b in fx], make=fix)

    def um(k, rng):
        n = int(k)
        f = rng.choice(["euh", "hum"])
        lab = f"environ {f} {c(n)} pour cent"
        return Piece(lab, f"environ {n}%", [("pct", f"{n}%", f"{c(n)} pour cent")], A)
    add(id="fr.dis.remplissage", family="dis", cat="disp", ambig=A, values=[str(n) for n in (5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 75, 80)], make=um)
    return out


BUILD = {"en": build_en, "es": build_es, "fr": build_fr}
