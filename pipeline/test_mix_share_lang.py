"""Per-language synthetic share caps of the wave-1 mix (no Modal): python3 test_mix_share_lang.py"""
import inspect

import mix_build as MB

# en / es / fr (decision 2026-10-17): every class may be up to half synthetic
for lang in ("en", "es", "fr"):
    for kind in ("int", "year", "unit", "roman", "frac", "amp", "clock", "cur"):
        assert MB.tts_share(lang, kind) == 0.5, (lang, kind, MB.tts_share(lang, kind))
# every other language keeps the caps: 30%, and 50% only for clock / price / phone / date
for lang in ("de", "it", "nl", "pt", "uk", "ru"):
    assert MB.tts_share(lang, "unit") == 0.30, lang
    assert MB.tts_share(lang, "int") == 0.30, lang
    for kind in ("clock", "cur", "phone", "date"):
        assert MB.tts_share(lang, kind) == 0.5, (lang, kind)
# the build's per-class cap reads the per-language share (not the language-blind table)
src = inspect.getsource(MB.build4)
assert "select_synthetic(" in src and "TTS_SHARE_KIND.get(k, TTS_SHARE)" not in src
import census_common as CC  # noqa: E402
assert all(MB.tts_share(l, k) == CC.syn_limit(l, k) for l in ("en", "de", "uk") for k in ("unit", "clock", "year"))
print("ok")

# every English part the v4 mix lists has a number share (else mix_share counts it as 0% numbers)
try:
    import modal_app as MA
    en = [p for p, _, _ in MA.retrain3_mix_v4() if p.startswith(("d1/en4_ok/", "mix3b/d1", "mix3b/d10n"))
          and p not in MA.EN_FREE_PARTS]  # the number-free parts (d1p_v / d10p_v) count 0% by design
    assert en and all(p in MB.NUMBER_PART for p in en), [p for p in en if p not in MB.NUMBER_PART]
    print("ok en number parts", len(en))
except ImportError:  # modal_app needs the modal package; the table entries are checked by name instead
    import re
    src = open("modal_app.py").read()
    en = set(re.findall(r'"(d1/en4_ok/[\w.]+\.jsonl|mix3b/d1\w*_ycap\.jsonl|mix3b/d10n_ycap\.jsonl)"', src))
    en = {p for p in en if not p.split("/")[-1].startswith("heldout_")}  # eval-only sets (probe_seam), never mix parts
    assert len(en) >= 10 and all(p in MB.NUMBER_PART for p in en), sorted(p for p in en if p not in MB.NUMBER_PART)
    print("ok en number parts (from modal_app source)", len(en))
