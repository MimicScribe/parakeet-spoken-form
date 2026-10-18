"""Rehearsal anchor rows (2026-10-18): real public speech held to STOCK's own written output, for languages that are
trained on spoken-form number rows but have no anchor rows. CPU only; nothing here trains. Writes only new paths
(/vol/ml/anchor/<tag>/<lang>/), and refuses to overwrite.

Per language, from the D2 harvest (harvest.py: VoxPopuli / YODAS-Granary / FLEURS-train, decoded once with stock;
label = stock's text; audit.jsonl holds every decision):
  plain.jsonl : number-free rows (no digit or symbol in caption or stock; no number word of the language's lexicon),
                ~plain_h hours. Rows the D2 harvest saved but the mix build did not list, then FLEURS-train rows the
                harvest audited as plain (cut from the FLEURS parquet; label = the audited stock text).
  digit.jsonl : digit-form rows (the harvest's "digit" kind: stock wrote the digits the caption has), ~digit_h hours.
Both exclude the build's held-out ids and held-out days / videos, anything the dedupe finds duplicating a protected
row (held-out rows, synthetic dev rows), any row whose text is a spoken-form target in the build, and rows already
listed in the build's d2p / fleurs_train_multi parts.

    modal run anchor_rows.py::build --langs ru,uk,pl,ro,sv,cs,hu,fi,sk,bg,hr --tag ancRH1 --mix-dir mix4final8
"""

import json
import os
import random
import re

import modal

VOL = "/vol"
HERE = os.path.dirname(os.path.abspath(__file__))
vol = modal.Volume.from_name("parakeet-spoken-form")
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg", "libsndfile1")
         .pip_install("pyarrow", "huggingface_hub", "hf_transfer", "soundfile", "librosa", "numpy")
         .add_local_python_source("harvest", "ml_balance", "census_common", "ml_p3_core", "ml_cands", "ml_slav")
         .add_local_file(os.path.join(HERE, "ml_grammar_table.json"), "/root/ml_grammar_table.json")
         .add_local_file(os.path.join(HERE, "ml_multi_lex.json"), "/root/ml_multi_lex.json"))
app = modal.App("parakeet-anchor-rows")

TARGET_PARTS = ("p3", "win", "tts", "seam", "seam_long", "win15")  # build4's spoken-form target texts
FLEURS_CAP = 0.6  # at most this share of a part's hours from FLEURS (the harvest's per-source cap)
UNIT_CAP = 0.08  # at most this share of a part's hours from one held-out unit (parliament day / video)


def _rows(path):
    if not os.path.exists(path):
        return []
    return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]


def _h(rows):
    return sum(r["duration"] for r in rows) / 3600


def _fold(w):
    import unicodedata

    return "".join(c for c in unicodedata.normalize("NFKD", w.lower()) if unicodedata.category(c) != "Mn")


def number_word_set(lang: str) -> set:
    """Written number words of a language (ml_multi_lex.json cardinals + ordinals, case-folded), for the languages the
    ml_balance lexicon does not cover; empty for the others."""
    p = "/root/ml_multi_lex.json"
    if not os.path.exists(p):
        p = os.path.join(HERE, "ml_multi_lex.json")
    lex = json.load(open(p)).get(lang)
    if not lex:
        return set()
    out = set()
    for k in ("cardinal", "ordinal"):
        for forms in lex.get(k, {}).values():
            for f in forms:
                out |= {_fold(t) for t in re.split(r"[\s\-]+", f) if len(t) >= 3}
    return out


def has_number_word(text: str, words: set) -> bool:
    return bool({_fold(t) for t in re.findall(r"[^\W\d_]+", text)} & words)


def pick(rows, hours, rng, unit_of, src_cap=None):
    """Random rows up to `hours`; no source above src_cap of the hours (when several sources), no unit above UNIT_CAP."""
    rows = list(rows)
    rng.shuffle(rows)
    out, h, by_src, by_unit = [], 0.0, {}, {}
    multi = len({r.get("src") for r in rows}) > 1
    for r in rows:
        if h >= hours * 3600:
            break
        s, u = r.get("src") or "?", unit_of(r["id"])
        if multi and src_cap and by_src.get(s, 0.0) + r["duration"] > src_cap * hours * 3600:
            continue
        if u and by_unit.get(u, 0.0) + r["duration"] > UNIT_CAP * hours * 3600:
            continue
        out.append(r)
        h += r["duration"]
        by_src[s] = by_src.get(s, 0.0) + r["duration"]
        if u:
            by_unit[u] = by_unit.get(u, 0.0) + r["duration"]
    return out


@app.function(image=image, volumes={VOL: vol}, secrets=[modal.Secret.from_name("hf")], cpu=2, memory=16384,
              timeout=3 * 3600, max_containers=11)
def build_lang(lang: str, tag: str, mix_dir: str = "mix4final8", plain_h: float = 2.0, digit_h: float = 1.0,
               seed: int = 71) -> dict:
    import soundfile as sf

    import census_common as CC
    import harvest
    import ml_balance as B

    vol.reload()
    out = f"{VOL}/ml/anchor/{tag}/{lang}"
    assert not os.path.exists(out), f"{out} exists: new tags only"
    rng = random.Random(f"{seed}-{lang}")
    md = f"{VOL}/{mix_dir}"
    held, units = CC.read_held_list(f"{VOL}/ml/p3v2/heldout_{lang}.txt")
    ho = _rows(f"{md}/heldout_{lang}.jsonl")
    held |= {x["id"] for x in ho}
    units |= {CC.held_unit(x["id"]) for x in ho} - {None}
    prot = ho + _rows(f"{md}/tts_dev_{lang}.jsonl") + _rows(f"{md}/tts_dev_other_{lang}.jsonl")
    targets = {x["text"] for k in TARGET_PARTS for x in _rows(f"{md}/{k}_{lang}.jsonl")}
    inmix = [x for p in ("d2p.jsonl", "fleurs_train_multi.jsonl") for x in _rows(f"{md}/{p}") if (x.get("lang") or "")[:2] == lang]
    mix_ids = {x["id"] for x in inmix}
    mix_text = {x["text"] for x in inmix}
    audit = {r["id"]: r for r in _rows(f"{VOL}/d2/{lang}/audit.jsonl")}
    nw = number_word_set(lang)
    lex = B.number_lexicon(lang)
    unit_of = CC.held_unit
    rep = {"lang": lang, "held_ids": len(held), "held_units": len(units), "targets": len(targets)}

    def clean(rows, kind):
        """The shared exclusions; returns (kept rows with caption, counts)."""
        c = {"in": len(rows)}
        rows = [r for r in rows if r["id"] not in mix_ids and r["text"] not in mix_text]
        c["not_in_mix"] = len(rows)
        rows = [r for r in rows if r["id"] not in held and unit_of(r["id"]) not in units]
        c["not_heldout"] = len(rows)
        rows = [r for r in rows if r["text"] not in targets]
        c["not_target"] = len(rows)
        if kind == "plain":
            rows = [r for r in rows if B.number_free(r["text"], lang, lex) and not has_number_word(r["text"], nw)]
            c["number_free"] = len(rows)
        rows = [dict(r, caption=r.get("caption") or audit.get(r["id"], {}).get("cap") or r["text"]) for r in rows]
        rows, drep = B.dedupe(rows, prot, part_of=lambda x: kind)
        c["after_dedupe"] = len(rows)
        c["dedupe"] = dict(drep["counts"])
        return rows, c

    os.makedirs(f"{out}/wav", exist_ok=True)
    # ---- digit rows: the harvest's saved digit rows (stock wrote the caption's digits; label = stock)
    drows, c = clean(_rows(f"{VOL}/d2/{lang}/digit.jsonl"), "digit")
    dsel = pick(drows, digit_h, rng, unit_of, FLEURS_CAP)
    rep["digit"] = dict(c, picked=len(dsel), h=round(_h(dsel), 3), pool_h=round(_h(drows), 3),
                        src={s: round(_h([r for r in dsel if r.get("src") == s]), 3) for s in {r.get("src") for r in dsel}})
    # ---- plain rows: saved harvest rows the build did not list, then FLEURS-train rows audited as plain
    saved, c1 = clean(_rows(f"{VOL}/d2/{lang}/plain.jsonl"), "plain")
    psel = pick(saved, plain_h, rng, unit_of, FLEURS_CAP)
    need = plain_h - _h(psel)
    fl_st = {}
    if need > 0.02:
        done = {r["id"] for r in psel}
        saved_ids = {x["id"] for x in _rows(f"{VOL}/d2/{lang}/plain.jsonl")}
        cand = [{"id": i, "text": a["stock"], "caption": a["cap"], "src": "fleurs", "duration": 9.0}
                for i, a in audit.items() if a.get("kept") == "plain" and a.get("src") == "fleurs"
                and i not in saved_ids and i not in done]
        cand, c2 = clean(cand, "plain")
        rng.shuffle(cand)
        # FLEURS may fill what the harvest rows leave, up to the hours target (the part's source cap is for the mix of
        # sources; a language whose saved rows are thin is topped up from FLEURS alone)
        chosen = {r["id"]: r for r in cand[: int(need * 3600 / 8.0 * 1.3) + 20]}
        have = {}
        fl_path = harvest.sources(lang)[0]
        assert fl_path[0] == "fleurs", fl_path
        for r in harvest.rows_of(*fl_path):
            if r["id"] in chosen:
                have[r["id"]] = r
        fl, tot = [], 0.0
        for i in chosen:
            if i not in have:
                continue
            if tot >= need * 3600:
                break
            r = have[i]
            wav = f"{out}/wav/{i}.wav"
            sf.write(wav, r["x"], 16000, subtype="PCM_16")
            fl.append(dict(chosen[i], audio_filepath=wav, duration=round(r["dur"], 3), lang=lang, origin="anchor_fleurs"))
            tot += r["dur"]
        psel += fl
        fl_st = dict(c2, asked=len(chosen), found=len(have), picked=len(fl), h=round(_h(fl), 3))
    rep["plain"] = dict(c1, picked=len(psel), h=round(_h(psel), 3), pool_h=round(_h(saved), 3), fleurs=fl_st,
                        src={s: round(_h([r for r in psel if r.get("src") == s]), 3) for s in {r.get("src") for r in psel}})
    for name, rows in (("plain", psel), ("digit", dsel)):
        with open(f"{out}/{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                assert os.path.exists(r["audio_filepath"]), r
                r = {k: v for k, v in r.items() if k not in ("controls",)}
                r.setdefault("lang", lang)
                r["anchor"] = name
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(rep, open(f"{out}/report.json", "w"), indent=1, ensure_ascii=False)
    vol.commit()
    return rep


@app.local_entrypoint()
def build(langs: str = "ru,uk,pl,ro,sv,cs,hu,fi,sk,bg,hr", tag: str = "ancRH1", mix_dir: str = "mix4final8",
          plain_h: float = 2.0, digit_h: float = 1.0):
    for r in build_lang.starmap([(l, tag, mix_dir, plain_h, digit_h) for l in langs.split(",") if l],
                                return_exceptions=True):
        print(json.dumps(r, ensure_ascii=False) if isinstance(r, dict) else f"FAIL {r!r}"[:600], flush=True)
