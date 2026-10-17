"""judge_plro.py value rules (runs the script on a temp pass): python3 test_judge_plro.py"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(lang, rows, ext):
    name = f"test_{uuid.uuid4().hex[:8]}"
    root = Path(tempfile.mkdtemp(prefix="judge_plro_test_"))
    w = root / lang / name
    w.mkdir(parents=True)
    try:
        (w / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        (w / "ext.jsonl").write_text("".join(json.dumps({"id": i, "text": t}, ensure_ascii=False) + "\n" for i, t in ext.items()))
        subprocess.run([sys.executable, str(HERE / "judge_plro.py"), lang, name], check=True, capture_output=True,
                       env=dict(os.environ, SPOKEN_FORM_JUDGE_ROOT=str(root)))
        return {json.loads(l)["id"]: json.loads(l)["reason"] for l in open(w / "jr/judge_rows.jsonl")}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def row(i, text, spans):
    return {"id": i, "text": text, "spans": spans}


def test_uk_values():
    rows = [
        row("big", "живе приблизно п'ять тисяч п'ятсот людей", [{"text": "5 500", "kind": "int", "reading": "п'ять тисяч п'ятсот"}]),
        row("bigm", "відвідали шість мільйонів туристів", [{"text": "6 000 000", "kind": "int", "reading": "шість мільйонів"}]),
        row("bigx", "живе приблизно п'ять тисяч п'ятсот людей", [{"text": "5 500", "kind": "int", "reading": "п'ять тисяч п'ятсот"}]),
        row("date", "підписали вісімнадцятого квітня дві тисячі тринадцятого року", [{"text": "18.04.2013", "kind": "date", "reading": "вісімнадцятого квітня дві тисячі тринадцятого року"}]),
        row("datex", "підписали вісімнадцятого квітня дві тисячі тринадцятого року", [{"text": "18.04.2013", "kind": "date", "reading": "вісімнадцятого квітня дві тисячі тринадцятого року"}]),
        row("ph", "за номером нуль дев'яносто п'ять вісімсот нуль шість вісімнадцять",
            [{"text": "095", "kind": "code", "reading": "нуль дев'яносто п'ять"}, {"text": "800-06-18", "kind": "chain", "reading": "вісімсот нуль шість вісімнадцять"}]),
        row("phx", "за номером нуль дев'яносто п'ять вісімсот нуль шість вісімнадцять",
            [{"text": "095", "kind": "code", "reading": "нуль дев'яносто п'ять"}, {"text": "800-06-18", "kind": "chain", "reading": "вісімсот нуль шість вісімнадцять"}]),
        row("far", "зовсім інші слова тут і там п'ять", [{"text": "7", "kind": "int", "reading": "сім"}]),
    ]
    ext = {"big": "живе приблизно 5500 людей", "bigm": "відвідали 6 мільйонів туристів", "bigx": "живе приблизно 5600 людей",
           "date": "підписали 18 квітня 2013 року", "datex": "підписали 18 травня 2013 року",
           "ph": "за номером 0958 000618", "phx": "за номером 0958 000619", "far": "ну так ось воно і є 5"}
    got = run("uk", rows, ext)
    # red-on-base: before, big / ph were span_missing, and datex / far were kept without a value check (slice WER > 0.25)
    assert got == {"big": "kept", "bigm": "kept", "bigx": "span_missing", "date": "kept", "datex": "span_missing",
                   "ph": "kept", "phx": "span_missing", "far": "span_missing"}, got


def test_uk_said_forms():
    rows = [
        row("pr", "коштує вісімнадцять євро п'ятдесят центів", [{"text": "18,50 €", "kind": "cur", "reading": "вісімнадцять євро п'ятдесят центів"}]),
        row("prx", "коштує вісімнадцять євро п'ятдесят центів", [{"text": "18,50 €", "kind": "cur", "reading": "вісімнадцять євро п'ятдесят центів"}]),
        row("sc", "становить мільйон двісті тисяч гривень", [{"text": "1 200 000", "kind": "int", "reading": "один мільйон двісті тисяч"}]),
        row("cl", "зустрінемося о вісімнадцятій нуль п'ять", [{"text": "18:05", "kind": "clock", "reading": "вісімнадцятій нуль п'ять"}]),
        row("ph", "за номером нуль дев'яносто сім вісімсот дев'яносто двадцять дев'ять сорок вісім",
            [{"text": "097", "kind": "code", "reading": "нуль дев'яносто сім"}, {"text": "890-29-48", "kind": "chain", "reading": "вісімсот дев'яносто двадцять дев'ять сорок вісім"}]),
    ]
    ext = {"pr": "коштує 18 євро 50 центів", "prx": "коштує 18 євро 40 центів", "sc": "становить 1 мільйон 200 тисяч гривень",
           "cl": "зустрінемося о 18,05", "ph": "за номером 097 800 90 29 48"}
    # red-on-base: pr / sc / cl / ph were span_missing before
    assert run("uk", rows, ext) == {"pr": "kept", "prx": "span_missing", "sc": "kept", "cl": "kept", "ph": "kept"}


def test_review_negatives():
    """No wrong value passes through an alternative (review 2026-10-17)."""
    ph = [{"text": "20", "kind": "code", "reading": "двадцять"}]
    rows = [
        # a 2-digit code inside a year: below the 7-digit floor, and not a phone sequence
        row("code20", "код двадцять у звіті", ph),
        # digits of a phone spread over two separate numbers of the transcript (not one sequence)
        row("phsplit", "номер нуль дев'яносто п'ять вісімсот нуль шість вісімнадцять",
            [{"text": "095", "kind": "code", "reading": "нуль дев'яносто п'ять"}, {"text": "800-06-18", "kind": "chain", "reading": "вісімсот нуль шість вісімнадцять"}]),
        # wrong month name; the right month number only as a stray digit elsewhere
        row("dmon", "підписали вісімнадцятого квітня дві тисячі тринадцятого року", [{"text": "18.04.2013", "kind": "date", "reading": "вісімнадцятого квітня дві тисячі тринадцятого року"}]),
        row("dm0", "підписали вісімнадцятого року", [{"text": "18.00.2013", "kind": "date", "reading": "вісімнадцятого"}]),
        row("dm13", "підписали вісімнадцятого року", [{"text": "18.13.2013", "kind": "date", "reading": "вісімнадцятого"}]),
        # cents present only as coincidental numbers
        row("cents", "коштує вісімнадцять євро п'ятдесят центів", [{"text": "18,50 €", "kind": "cur", "reading": "вісімнадцять євро п'ятдесят центів"}]),
        # a price heard as a clock, and a clock heard as a price
        row("prclk", "коштує вісімнадцять гривень п'ять копійок", [{"text": "18,05 грн", "kind": "cur", "reading": "вісімнадцять гривень п'ять копійок"}]),
        row("clkpr", "о вісімнадцятій нуль п'ять", [{"text": "18:05", "kind": "clock", "reading": "вісімнадцятій нуль п'ять"}]),
        # partial scale sum, and a scale stem inside another word
        row("part", "становить мільйон двісті тисяч триста гривень", [{"text": "1 200 000", "kind": "int", "reading": "один мільйон двісті тисяч"}]),
        row("stem", "два тисячоліття тому", [{"text": "2 000", "kind": "int", "reading": "дві тисячі"}]),
    ]
    ext = {"code20": "код у звіті за 2023", "phsplit": "номер 095 800 у 2023 році 06 18",
           "dmon": "підписали 18 травня 2013 року і 4 дні тому", "dm0": "підписали 18 грудня 2013", "dm13": "підписали 18 13 2013",
           "cents": "коштує 18 євро і ще 50 центів потім", "prclk": "коштує 18:05", "clkpr": "о 18,05 гривень",
           "part": "становить 1 мільйон 200 тисяч 300 гривень", "stem": "2 тисячоліття тому"}
    got = run("uk", rows, ext)
    # red-on-base: the previous judge kept every one of these (scattered numbers, a December read for month 0, the
    # blind x1000 scale, a clock read as cents and back)
    assert all(v == "span_missing" for v in got.values()), got


def test_clock_adjacent_forms_and_real_phone_groups():
    cl = lambda i, t: row(i, "о годині", [{"text": t, "kind": "clock", "reading": "x y"}])  # noqa: E731
    rows = [cl("a", "14:15"), cl("b", "9:06"), cl("c", "15:20"), cl("d", "14:15")]
    ext = {"a": "о 14 і 15 біля входу", "b": "було 9 0 6 коли", "c": "було 15-20 коли", "d": "о 14 біля входу і 15 хвилин"}
    assert run("uk", rows, ext) == {"a": "kept", "b": "kept", "c": "kept", "d": "span_missing"}
    # a real (tightened) row: the same digits in other groups is another reading, not a pass
    real = lambda i: dict(row(i, "regulamentului șaisprezece nouăzeci și opt două mii cinci în",  # noqa: E731
                              [{"text": "16-98-2005", "kind": "phone", "reading": "șaisprezece nouăzeci și opt două mii cinci"}]), cer_ext=0.0)
    got = run("ro", [real("r1"), real("r2")], {"r1": "regulamentului 1698-2005 în", "r2": "regulamentului 16 98 2005 în"})
    assert got == {"r1": "span_missing", "r2": "kept"}, got


def test_tightened_row_slice_rule_unchanged():
    # control: a tightened row (cer_ext present) with a slice that does not line up is still not value-checked
    r = dict(row("t", "zupełnie inne słowa tutaj siedem", [{"text": "7", "kind": "int", "reading": "siedem"}]), cer_ext=0.02)
    assert run("pl", [r], {"t": "no to jest coś 5"}) == {"t": "kept"}


def test_pl_sv_month_names_and_cents_gap():
    # 2026-10-17: Polish synthetic dates were rejected for want of month names; "zł i 90 groszy" / "kronor och 99 öre"
    rows = [row("d", "osiemnastego maja dwa tysiące siódmego", [{"text": "18.5.2007", "kind": "date", "reading": "osiemnastego maja dwa tysiące siódmego"}]),
            row("c", "dziewięćdziesiąt sześć złotych i dziewięćdziesiąt groszy", [{"text": "96,90 zł", "kind": "cur", "reading": "dziewięćdziesiąt sześć złotych i dziewięćdziesiąt groszy"}]),
            row("w", "osiemnastego maja", [{"text": "18.6.2007", "kind": "date", "reading": "osiemnastego czerwca dwa tysiące siódmego"}])]
    got = run("pl", rows, {"d": "spotkanie 18 maja 2007 aby", "c": "kosztował 96 zł i 90 groszy", "w": "spotkanie 18 maja 2007"})
    assert got == {"d": "kept", "c": "kept", "w": "span_missing"}, got   # w: the month heard is not the caption's
    ro = [row("r", "pe optsprezece martie două mii douăzeci și unu", [{"text": "18.3.2021,", "kind": "date", "reading": "optsprezece martie două mii douăzeci și unu"}])]
    assert run("ro", ro, {"r": "comunicată pe 18 martie 2021 după"}) == {"r": "kept"}   # trailing comma in the span text
    ro2 = [row("k", "la ora zece și cincisprezece", [{"text": "10:15", "kind": "clock", "reading": "zece și cincisprezece"}]),
           row("v", "versiunea treisprezece virgulă cinci", [{"text": "13.5,", "kind": "dec", "reading": "treisprezece virgulă cinci"}])]
    assert run("ro", ro2, {"k": "ne revedem la ora 10 și 15 pentru", "v": "versiunea 13,5 stabilește"}) == {"k": "kept", "v": "kept"}
    sv = [row("s", "sjuttiotvå kronor och nittionio öre", [{"text": "72,99 kr", "kind": "cur", "reading": "sjuttiotvå kronor och nittionio öre"}])]
    assert run("sv", sv, {"s": "det kostar 72 kronor och 99 öre"}) == {"s": "kept"}


def test_long_number_langs_month_names_and_cents():
    # 2026-10-17 long-number sets: dated "15. maaliskuuta" / "15. ožujka", Hungarian month-first dates, and cents after
    # "a" / "és" / "ja" / "i" with the minor unit (red-on-base: every kept row below was span_missing before)
    def d(i, t, rd):
        return row(i, rd, [{"text": t, "kind": "date", "reading": rd}])

    def c(i, t, rd):
        return row(i, rd, [{"text": t, "kind": "cur", "reading": rd}])
    cases = {
        "sv": ([d("ok", "15.3.2041", "femtonde mars tjugohundrafyrtioett"), d("bad", "15.4.2041", "femtonde april tjugohundrafyrtioett")],
               {"ok": "mötet den 15 mars 2041 i", "bad": "mötet den 15 mars 2041 i"}),
        "cs": ([d("ok", "3.11.2035", "třetího listopadu dva tisíce třicet pět"), c("p", "72,99 Kč", "sedmdesát dva korun a devadesát devět haléřů")],
               {"ok": "konference 3. listopadu 2035 v", "p": "stojí 72 korun a 99 haléřů"}),
        "sk": ([d("ok", "1.5.1998", "prvého mája tisícdeväťstodeväťdesiatosem")], {"ok": "od 1. mája 1998"}),
        "hu": ([d("ok", "18.3.2021", "kétezer-huszonegy március tizennyolcadikán"), c("p", "72,50 €", "hetvenkét euró és ötven cent"),
                d("bad", "18.3.2021", "kétezer-huszonegy március tizennyolcadikán")],
               {"ok": "2021. március 18-án", "p": "ára 72 euró és 50 cent", "bad": "2021. április 18-án"}),
        "fi": ([d("ok", "15.3.2050", "viidentenätoista maaliskuuta kaksituhattaviisikymmentä"), c("p", "12,40 €", "kaksitoista euroa ja neljäkymmentä senttiä")],
               {"ok": "kokous 15. maaliskuuta 2050 alkaa", "p": "maksaa 12 euroa ja 40 senttiä"}),
        "hr": ([d("ok", "20.11.2030", "dvadesetog studenoga dvije tisuće tridesete"), c("p", "15,20 €", "petnaest eura i dvadeset centi")],
               {"ok": "do 20. studenoga 2030", "p": "košta 15 eura i 20 centi"}),
        "bg": ([d("ok", "5.7.2044", "пети юли две хиляди четиридесет и четвърта")], {"ok": "на 5 юли 2044 г"}),
    }
    for lang, (rows, ext) in cases.items():
        got = run(lang, rows, ext)
        want = {r["id"]: ("span_missing" if r["id"] == "bad" else "kept") for r in rows}
        assert got == want, (lang, got)


def test_long_number_scale_words():
    # 2026-10-17: a space-grouped big value said with a scale word ("600 000" heard "600 tisíc", "23 000 000 000" heard
    # "23 miliárd"), and Romanian "lei și 99 de bani" (red-on-base: the kept rows below were span_missing before)
    def b(i, t, rd):
        return row(i, rd, [{"text": t, "kind": "int", "reading": rd}])
    sk = [b("k", "600 000", "šesťsto tisíc"), b("g", "23 000 000 000", "dvadsaťtri miliárd"), b("x", "600 000", "šesťsto tisíc"),
          b("m", "3 500 000", "tri milióny päťsto tisíc")]
    got = run("sk", sk, {"k": "ďalších 600 tisíc eur", "g": "dosiahla 23 miliárd eur", "x": "ďalších 700 tisíc eur",
                         "m": "spolu 3 milióny 500 tisíc eur"})
    assert got == {"k": "kept", "g": "kept", "x": "span_missing", "m": "kept"}, got
    pl = [b("p", "817 000", "osiemset siedemnaście tysięcy"), b("q", "386 002 000", "trzysta osiemdziesiąt sześć milionów dwa tysiące")]
    assert run("pl", pl, {"p": "poddano 817 tysięcy kobiet", "q": "aż 386,2 mln osób"}) == {"p": "kept", "q": "span_missing"}
    for lang, rd, et in (("sv", "tolv miljoner", "investerar 12 miljoner kronor"), ("hu", "hárommillió-ötszázezer", "összesen 3 millió 500 ezer euró"),
                         ("fi", "neljätuhatta", "noin 4 tuhatta ihmistä"), ("bg", "седем хиляди", "около 7 хиляди души")):
        t = {"sv": "12 000 000", "hu": "3 500 000", "fi": "4 000", "bg": "7 000"}[lang]
        assert run(lang, [b("s", t, rd)], {"s": et}) == {"s": "kept"}, lang
    ro = [row("c", "nouăzeci și patru de lei și cincizeci de bani", [{"text": "94,50 lei", "kind": "cur", "reading": "nouăzeci și patru de lei și cincizeci de bani"}]),
          row("cx", "nouăzeci și patru de lei și cincizeci de bani", [{"text": "94,50 lei", "kind": "cur", "reading": "nouăzeci și patru de lei și cincizeci de bani"}])]
    assert run("ro", ro, {"c": "acum 94 lei și 50 de bani", "cx": "acum 94 lei și 60 de bani"}) == {"c": "kept", "cx": "span_missing"}


def test_long_number_ranges():
    # 2026-10-17: big-number ranges heard with scale words (red-on-base: every kept row below was span_missing before)
    def r(i, t, rd):
        return row(i, rd, [{"text": t, "kind": "range", "reading": rd}])
    sk = [r("a", "420 000 000 000–630 000 000 000", "štyristodvadsať miliárd až šesťstotridsať miliárd"),
          r("b", "8 030 000–12 000 000", "osem miliónov tridsaťtisíc až dvanásť miliónov"),
          r("x", "420 000 000 000–630 000 000 000", "štyristodvadsať miliárd až šesťstotridsať miliárd"),
          r("y", "8 030 000–12 000 000", "osem miliónov tridsaťtisíc až dvanásť miliónov")]
    got = run("sk", sk, {"a": "medzi 420 miliárd až 630 miliárd eur", "b": "na 8 miliónov 30 tisíc až 12 miliónov eur",
                         "x": "medzi 420 miliárd až 640 miliárd eur", "y": "na 8 miliónov až 12 miliónov eur"})
    assert got == {"a": "kept", "b": "kept", "x": "span_missing", "y": "span_missing"}, got
    hu = [r("s", "300 000–750 000", "háromszázezer és hétszázötvenezer"), r("t", "300 000–750 000", "háromszázezer és hétszázötvenezer")]
    got = run("hu", hu, {"s": "ez az összeg 300 és 750 ezer euró között", "t": "ez az összeg 350 és 750 ezer euró között"})
    assert got == {"s": "kept", "t": "span_missing"}, got


if __name__ == "__main__":
    test_long_number_ranges()
    test_long_number_scale_words()
    test_long_number_langs_month_names_and_cents()
    test_uk_values()
    test_uk_said_forms()
    test_review_negatives()
    test_clock_adjacent_forms_and_real_phone_groups()
    test_tightened_row_slice_rule_unchanged()
    test_pl_sv_month_names_and_cents_gap()
    print("ok")
