#!/usr/bin/env python3
"""copycount_gate: repeat-aware gate for synthetic speech (strict COPY-COUNT probe). Pure functions + a fake-scorer selftest; the CTC log-probs come from
any CTC scorer you give it (a callable nll(text) -> nats, e.g. tools/labels/w2v_torch.losses_fn on the clip's log-probs).
WHY: a gate that keeps a clip on character error rate and 'reading heard' lets a collapsed copy through ('two two' voiced as 'two' is ~3 characters = ~5 % CER and every
reading probe still passes); TTS engines do collapse repeated groups. Single-word deletion probes usually skip short words and nothing tests an ADDED copy.
PER repeat slot, for every UNIT of every designed run (units: (w0, w1, copies, digit_word_index) in chunk word indices, with an explicit unit kind 'mult' = modifier + digit
word ('double five') or 'whole' = every other copy: a single word, a 2-3 word period ('four two'), a pair word ('thirty two')):
  1. variants of the chunk text T: MINUS = one copy removed (mult: 'double five' -> 'five', 'triple' -> 'double', 'quadruple' -> 'triple'; whole: the unit deleted, only if its run keeps another unit),
     PLUS = one copy added (mult: 'double' -> 'triple', 'triple' -> 'quadruple', 'quadruple' -> 'quintuple'; whole: the unit duplicated, e.g. 'four two' -> 'four two four two').
  2. NLL(variant) - NLL(T) in CTC nats must be >= FLOOR (3.0) for every variant, else FAIL (copy_dropped_suspect / copy_added_suspect). Between the floor and the CALIBRATED margin
     (calibrate_margin(): a low percentile of the same contrast on ordinary chunks of ONE engine; saved as {"contrasts": {"minus": {"margin": m}, "plus": {"margin": p}}}
     and loaded with load_margins(path)) the slot is UNCERTAIN: it is kept and routed to a human listening queue.
  3. each copy word's forced-alignment duration >= 0.12 s and, between two adjacent plain copy words without a planned pause, the energy envelope dips >= DIP_DB (onset_between).
  4. filler slots: the filler word must be voiced (variant without the filler must lose by >= FILLER_MIN) - informational flag `filler_unvoiced`.
  5. A slot that fails 2 / 3 is a `collapsed` suspect. Policy: NEVER silently drop it and NEVER relabel it to the fewer-copies reading (that would keep only the repeats the engine
     renders cleanly and make the data easy): re-voice it (different speed / punctuated rendering / other engine) and, if the last round still fails, keep it as a FLAGGED stratum
     that is scored apart; cap exclusions (for example 10 % of repeat slots).
  6. LISTENERS ARE ASYMMETRIC: many ASR systems dedupe repeats, so 'fewer copies heard' never disputes a slot and an ASR system is never a voter on a copy count; only 'more copies or
     different digits than the script' is a dispute. Disputes and CTC-uncertain slots go to a human listening page; unheard disputes stay as a scored HARD stratum, never dropped.
Selftest: python3 -I copycount_gate.py   (fake scorer: audio = the string actually voiced)"""
import difflib, json, os, re, sys

FLOOR = 3.0
MIN_COPY_DUR = 0.12
DIP_DB = 3.0
FILLER_MIN = 2.0
MULT_UP = {"double": "triple", "triple": "quadruple", "quadruple": "quintuple"}
MULT_DOWN = {"quintuple": "quadruple", "quadruple": "triple", "triple": "double"}
MULT_ALL = set(MULT_UP) | set(MULT_DOWN) | {"double"}


def load_margins(path=None):
    """calibrated margins from a json file {"contrasts": {"minus": {"margin": m}, "plus": {"margin": p}}} (write it from calibrate_margin()); without a file,
    or when it cannot be read, the floor: nothing is then reported as uncertain. `provisional` stays True until you calibrate on your own engine."""
    if not path:
        return {"minus": FLOOR, "plus": FLOOR, "provisional": True}
    try:
        with open(path, encoding="utf-8") as f:
            c = json.load(f)["contrasts"]
        return {"minus": float(c["minus"]["margin"]), "plus": float(c["plus"]["margin"]), "provisional": False}
    except (OSError, ValueError, KeyError, TypeError):
        return {"minus": FLOOR, "plus": FLOOR, "provisional": True}


def _bare(w):
    return w.rstrip(",").lower()


def variants(words, units, ukinds=None, run_ids=None):
    """words: chunk word list (tts form); units: [(w0, w1, copies, digit_word, ...)] in CHUNK word indices; ukinds: 'mult' | 'whole' per unit (detected from the first word when absent);
    run_ids: run index per unit. -> list of dict(kind 'minus'|'plus', unit, words)"""
    out = []
    n_in_run = {}
    for i, u in enumerate(units):
        r = run_ids[i] if run_ids else 0
        n_in_run[r] = n_in_run.get(r, 0) + 1
    for ui, u in enumerate(units):
        w0, w1 = u[0], u[1]
        sl = words[w0:w1]
        kind = ukinds[ui] if ukinds else ("mult" if (len(sl) == 2 and _bare(sl[0]) in MULT_ALL) else "whole")
        if kind == "mult":
            m, d = _bare(sl[0]), sl[1]
            minus = [d] if m == "double" else ([MULT_DOWN[m], d] if m in MULT_DOWN else None)
            plus = [MULT_UP[m], d] if m in MULT_UP else None
        else:
            minus = [] if n_in_run[run_ids[ui] if run_ids else 0] >= 2 else None  # deleting the only unit of a run would delete the digit itself
            plus = sl + sl
        for k, new in (("minus", minus), ("plus", plus)):
            if new is not None:
                out.append({"kind": k, "unit": ui, "words": words[:w0] + new + words[w1:]})
    return out


def probe(nll, words, units, ukinds=None, run_ids=None, margins=None, fillers=()):
    """nll: callable(text) -> nats. -> dict(ok, why, uncertain, margin_minus, margin_plus, worst_unit, filler_unvoiced)"""
    margins = margins or load_margins()
    base = nll(" ".join(words))
    mm = mp = None
    worst = {}
    for v in variants(words, units, ukinds, run_ids):
        d = nll(" ".join(v["words"]).strip()) - base
        if v["kind"] == "minus":
            if mm is None or d < mm:
                mm, worst["minus"] = d, v["unit"]
        else:
            if mp is None or d < mp:
                mp, worst["plus"] = d, v["unit"]
    why = ""
    if mm is not None and mm < FLOOR:
        why = "copy_dropped_suspect"
    elif mp is not None and mp < FLOOR:
        why = "copy_added_suspect"
    unc = (not why) and ((mm is not None and mm < margins["minus"]) or (mp is not None and mp < margins["plus"]))
    fu = False
    for wi in fillers:  # a filler word that is not voiced: deleting it costs (almost) nothing
        if nll(" ".join(words[:wi] + words[wi + 1:])) - base < FILLER_MIN:
            fu = True
    return {"ok": not why, "why": why, "uncertain": bool(unc), "margin_minus": None if mm is None else round(mm, 2), "margin_plus": None if mp is None else round(mp, 2), "worst_unit": worst,
            "filler_unvoiced": fu, "margins_used": {"minus": margins["minus"], "plus": margins["plus"], "provisional": margins.get("provisional", True)}}


def copy_word_indices(units, ukinds=None, words=None):
    """words that carry a copy and must last >= MIN_COPY_DUR: the digit word of a mult unit, every word of a whole unit"""
    out = []
    for i, u in enumerate(units):
        w0, w1, c, dw = u[0], u[1], u[2], u[3]
        k = ukinds[i] if ukinds else ("mult" if (w1 - w0 == 2 and words and _bare(words[w0]) in MULT_ALL) else "whole")
        out += [dw] if k == "mult" else list(range(w0, w1))
    return out


def copy_durations_ok(word_times, idx, min_dur=MIN_COPY_DUR):
    return all(word_times[i] is not None and (word_times[i][1] - word_times[i][0]) >= min_dur for i in idx)


def onset_between(x, sr, word_times, i, j, dip_db=DIP_DB):
    """energy dip between copy words i and j (plain runs): the lowest 20 ms RMS between the two word centres is >= dip_db below the lower of the two word peaks"""
    import numpy as np

    def env(a, b):
        seg = x[int(a * sr):int(b * sr)]
        n = int(0.02 * sr)
        if len(seg) < n:
            return np.array([0.0])
        return np.sqrt((seg[: len(seg) // n * n].reshape(-1, n) ** 2).mean(1) + 1e-12)
    pi, pj = env(*word_times[i]).max(), env(*word_times[j]).max()
    gap = env((word_times[i][0] + word_times[i][1]) / 2, (word_times[j][0] + word_times[j][1]) / 2).min()
    return bool(20 * np.log10(min(pi, pj) / (gap + 1e-9)) >= dip_db)


def adjacent_plain_pairs(units, ukinds, run_ids, pause_after=()):
    """[(digit_word_i, digit_word_j)] for consecutive 1-word units of one run; pause_after = set of (run_id, unit_index) whose following gap carries a planned pause (the stitch adds that silence: skipped)"""
    out = []
    for k in range(len(units) - 1):
        same = (run_ids[k] == run_ids[k + 1]) if run_ids else True
        one = all((units[x][1] - units[x][0]) == 1 for x in (k, k + 1))
        if same and one and (ukinds is None or (ukinds[k] == "whole" and ukinds[k + 1] == "whole")) and (run_ids[k] if run_ids else 0, k) not in set(pause_after):
            out.append((units[k][3], units[k + 1][3]))
    return out


def calibrate_margin(contrasts, floor=FLOOR, pct=5):
    """contrasts: NLL(false variant) - NLL(true text) over ordinary chunks of ONE engine. -> (margin, separable)"""
    xs = sorted(contrasts)
    if not xs:
        return floor, False
    m = xs[min(len(xs) - 1, int(len(xs) * pct / 100))]
    return max(floor, m), m >= floor


def _selftest():
    def make(voiced):
        v = voiced.split()

        def nll(text):
            t = text.split()
            sm = difflib.SequenceMatcher(None, v, t, autojunk=False)
            return 10.0 + 4.0 * sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
        return nll
    M = {"minus": 8.0, "plus": 20.0, "provisional": True}
    F = {"minus": 3.0, "plus": 3.0}
    words = ["the", "code", "is", "four", "four", "seven"]
    units = [(3, 4, 1, 3), (4, 5, 1, 4)]
    ok = probe(make("the code is four four seven"), words, units, margins=F)
    assert ok["ok"] and not ok["uncertain"], ok
    unc = probe(make("the code is four four seven"), words, units, margins=M)  # 4 nats per word edit < the margins 8.0 / 20.0 -> uncertain (listening queue), not failed
    assert unc["ok"] and unc["uncertain"], unc
    assert probe(make("the code is four seven"), words, units, margins=M)["why"] == "copy_dropped_suspect"
    assert probe(make("the code is four four four seven"), words, units, margins=M)["why"] == "copy_added_suspect"
    w2 = ["pin", "double", "five", "nine"]
    u2 = [(1, 3, 2, 2)]
    assert probe(make("pin double five nine"), w2, u2, ["mult"], [0], F)["ok"]
    assert probe(make("pin five nine"), w2, u2, ["mult"], [0], M)["why"] == "copy_dropped_suspect"
    assert not probe(make("pin triple five nine"), w2, u2, ["mult"], [0], M)["ok"]
    w3 = ["pin", "quadruple", "five", "nine"]
    assert any(v["kind"] == "plus" and v["words"][1] == "quintuple" for v in variants(w3, [(1, 3, 4, 2)], ["mult"], [0]))
    for txt, us in (("pin is four two four two nine", [(2, 4, 1, 3), (4, 6, 1, 5)]), ("id is thirty two thirty two", [(2, 4, 1, 3), (4, 6, 1, 5)]), ("pin is four two one four two one nine", [(2, 5, 1, 4), (5, 8, 1, 7)])):
        ws = txt.split()
        vs = variants(ws, us, ["whole", "whole"], [0, 0])
        assert len(vs) == 4 and {v["kind"] for v in vs} == {"minus", "plus"}, (txt, vs)
        assert probe(make(txt), ws, us, ["whole", "whole"], [0, 0], F)["ok"]
        collapsed = " ".join(ws[:us[0][0]] + ws[us[0][0]:us[0][1]] + ws[us[1][1]:])
        assert not probe(make(collapsed), ws, us, ["whole", "whole"], [0, 0], M)["ok"], txt
        added = " ".join(ws[:us[1][1]] + ws[us[0][0]:us[0][1]] + ws[us[1][1]:])
        assert not probe(make(added), ws, us, ["whole", "whole"], [0, 0], M)["ok"], txt
    d1 = "pin is four two four two nine".split()
    assert len(variants(d1, [(2, 4, 1, 3), (4, 6, 1, 5)])) == 4  # a 2-word period unit is NOT a multiplier (auto-detection)
    fw = "two uh two".split()
    fu = [(0, 1, 1, 0), (2, 3, 1, 2)]
    assert not probe(make("two uh two"), fw, fu, fillers=[1], margins=F)["filler_unvoiced"]
    assert probe(make("two two"), fw, fu, fillers=[1], margins=F)["filler_unvoiced"]
    assert adjacent_plain_pairs([(0, 1, 1, 0), (1, 2, 1, 1), (2, 3, 1, 2)], ["whole"] * 3, [0, 0, 0], pause_after={(0, 0)}) == [(1, 2)]
    m, sep = calibrate_margin([5, 6, 7, 8, 9, 10, 4, 6, 7, 8, 9, 10, 11, 12, 5, 6, 7, 8, 9, 10])
    assert sep and m >= 3
    print("copycount_gate selftest ok: collapsed/added caught for plain, mult (double..quintuple), whole multi-word units; provisional margins -> uncertain, floor -> fail; filler check; pause-gap skip; calibration", m)


def main(argv=None):
    """selftest (no arguments) or `calibrate FILE [--pct 5]`: FILE = json list of NLL(false variant) - NLL(true text) contrasts from ordinary chunks of one engine"""
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        _selftest()
        return 0
    if argv[0] == "calibrate" and len(argv) >= 2:
        pct = float(argv[argv.index("--pct") + 1]) if "--pct" in argv else 5
        with open(argv[1], encoding="utf-8") as f:
            m, sep = calibrate_margin(json.load(f), pct=pct)
        print(json.dumps({"margin": round(m, 3), "separable": sep}))
        return 0
    print(main.__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
