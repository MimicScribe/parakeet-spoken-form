import sys, os, json, random, tempfile, io, contextlib
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import variety as V

R = random.Random(11)
def _word(i):
    c, v = "bdfgklmnprstv", "aeiou"
    return c[i % 13] + v[(i // 13) % 5] + c[(i // 65) % 13] + v[(i // 845) % 5] + c[(i * 7) % 13] + "n"


VOCAB = [_word(i) for i in range(700)]
NUM = ["four fifteen", "twelve thousand", "nine nine one", "two hundred", "twenty twenty", "seven point five"]


def sentence(i, length=None, q=False, opening=None):
    n = length or R.choice([4, 6, 9, 12, 16, 20, 24])
    ws = [R.choice(VOCAB) for _ in range(n)]
    if opening:
        ws[:3] = opening
    ws.insert(R.randrange(len(ws) + 1), R.choice(NUM))
    return " ".join(ws) + ("?" if q else ".")


def good_rows(n=400):
    rows = []
    for i in range(n):
        rows.append(dict(said=sentence(i, q=(i % 12 == 0)), domain="d%d" % (i % 30), clip_pos="clip_start" if i % 5 == 0 else "interior",
                         slots=[dict(value=str(100000 + i * 7), cell="open")] + ([dict(value=str(500000 + i * 7), cell="open")] if i % 5 == 1 else [])))
    return rows


def names(gates):
    return {n.split()[0]: ok for n, ok, _ in gates}


def test_a_varied_pack_passes_the_text_gates():
    g = V.evaluate(good_rows())
    assert all(ok for _, ok, _ in g), [x for x in g if not x[1]]
    assert {"T1", "T2", "T3", "T8", "T9", "N", "V1"} <= set(names(g))


def test_template_openings_and_low_variety_fail():
    rows = [dict(said=sentence(i, opening=["the", "total", "is"], q=(i % 12 == 0))) for i in range(400)]
    assert names(V.evaluate(rows))["T3"] is False
    same = [dict(said="the meter reads %s today." % R.choice(NUM)) for _ in range(400)]
    g = names(V.evaluate(same))
    assert g["T1"] is False and g["T2"] is False and g["T8"] is False and g["T9"] is False


def test_caps_and_shares_from_config():
    rows = good_rows()
    cfg = {"caps": {"domain": 0.005}, "shares": [{"field": "clip_pos", "value": "clip_start", "min": 0.15, "max": 0.25}]}
    g = V.evaluate(rows, config=cfg)
    byname = {n: ok for n, ok, _ in g}
    assert byname["C domain: max value share <= 0% [n]"] is False          # 30 values -> 3.3 % each
    assert byname["S clip_pos=clip_start share in [0.15, 0.25] [n]"] is True
    assert V.evaluate(rows, config={"caps": {"domain": 0.10}})[-1][1] is True


def test_sampling_noise_relaxes_small_packs_only():
    assert V.noise(100) > V.noise(1000) > 0
    rows = good_rows(300)
    for r in rows[:300]:
        r["domain"] = "d0" if R.random() < 0.04 else r["domain"]
    cfg = {"caps": {"domain": 0.06}}
    assert [ok for n, ok, _ in V.evaluate(rows, config=cfg) if n.startswith("C domain")] == [True]       # n < 1000: margin applied
    big = [dict(r) for r in good_rows(1000)]
    for r in big:
        r["domain"] = "d0" if R.random() < 0.075 else r["domain"]
    assert [ok for n, ok, _ in V.evaluate(big, config=cfg) if n.startswith("C domain")] == [False]       # n >= 1000: strict


def test_slot_value_repeats_and_closed_cells():
    rows = good_rows()
    rows[1]["slots"][0]["value"] = rows[0]["slots"][0]["value"]
    assert names(V.evaluate(rows))["V1"] is True                                            # small packs tolerate 3 open-space repeats
    for k in range(2, 9):
        rows[k]["slots"][0]["value"] = rows[0]["slots"][0]["value"]
    assert names(V.evaluate(rows))["V1"] is False
    cfgs = {"slots": {"closed_cells": ["open"], "max_repeat": 10}}
    assert names(V.evaluate(rows, config=cfgs))["V1"] is True                              # closed-space cells may repeat up to the cap


def test_audio_plan_gates():
    n = 200
    audio = [dict(tempo=1.0 if i % 2 else 0.97, voice="v%d" % (i % 40), gender="f" if i % 2 else "m", engine="e1" if i % 10 else "e2") for i in range(n)]
    cfg = {"audio": {"tempo": [0.95, 1.05], "tempo_exact_min": 0.45, "gender_band": [0.47, 0.53], "max_rows_per_voice": 6, "shares": [{"field": "engine", "value": "e2", "min": 0.05, "max": 0.15}]}}
    g = V.evaluate(good_rows(n), audio, cfg)
    assert all(ok for nme, ok, _ in g if nme.startswith("A")), [x for x in g if x[0].startswith("A") and not x[1]]
    audio[0]["tempo"] = 1.3
    assert [ok for nme, ok, _ in V.evaluate(good_rows(n), audio, cfg) if nme.startswith("A1")] == [False]
    audio2 = [dict(a, voice="v0") for a in audio]
    assert [ok for nme, ok, _ in V.evaluate(good_rows(n), audio2, cfg) if nme.startswith("A3")] == [False]


def test_cli_exit_code_and_report():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "rows.jsonl"); open(p, "w").write("".join(json.dumps(r) + "\n" for r in good_rows()))
    out = os.path.join(d, "r.txt")
    with contextlib.redirect_stdout(io.StringIO()):
        assert V.main(["--rows", p, "--out", out]) == 0
    assert "gates pass" in open(out).read()
    bad = os.path.join(d, "bad.jsonl"); open(bad, "w").write("".join(json.dumps(dict(said="the meter reads four fifteen today.")) + "\n" for _ in range(50)))
    with contextlib.redirect_stdout(io.StringIO()):
        assert V.main(["--rows", bad]) == 1


if __name__ == "__main__":
    import tinytest; tinytest.run(globals())
