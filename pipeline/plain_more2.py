"""More number-free real speech (2026-10-18 top-up), by the D2p rule (harvest: no digit in caption or stock, CER <= 0.15,
text LID = the language; label = STOCK's text). Writes only new paths under /vol/ml/plain/<lang>/; never overwrites.

recover : rows the plain_more pass (modal_app.plain_more_lang, tag "more") already decoded and passed, but never saved
          because the per-source 60% cap of its 10 h quota stopped it (audit_more.jsonl kept == "plain", id not in
          plain_more.jsonl). Re-reads the same VoxPopuli files and cuts those rows: CPU only, no new decode.
              modal run plain_more2.py::recover --langs de,es,fr,it --tag morerec
gv      : Portuguese (and any VoxPopuli-unlabelled language) parliament speech: NVIDIA Granary's VoxPopuli
          pseudo-labels (CC-BY-4.0; audio = VoxPopuli unlabelled, CC0), number-free captions only (no digit, no number
          word, ml_balance.number_free), segments 3-20 s, at most PER_EVENT per sitting, spread over `years`; cut from
          the session tars (as ml_p3.granvox_cut_lang), then decoded with stock and kept by the D2p rule.
              modal run plain_more2.py::gv --lang pt --years 2016,2017,2018 --hours 45 --tag gvplain1
Rows carry src "vox" (the build's 60% source cap counts them with the other parliament rows), origin "d2p_rec" /
"gv_plain", number_free and controls (ml_balance.lookalikes) as plain_recover does.
"""

import json
import os
import re

import modal

VOL = "/vol"
vol = modal.Volume.from_name("parakeet-spoken-form")
app = modal.App("parakeet-plain2")
HERE = os.path.dirname(os.path.abspath(__file__))
cpu_image = (modal.Image.debian_slim(python_version="3.11")
             .apt_install("ffmpeg", "libsndfile1")
             .pip_install("pyarrow", "huggingface_hub", "hf_transfer", "soundfile", "librosa", "numpy")
             .add_local_python_source("harvest", "ml_balance", "census_common", "ml_p3_core", "ml_cands", "ml_slav")
             .add_local_file(os.path.join(HERE, "ml_grammar_table.json"), "/root/ml_grammar_table.json"))
# modal_app.nemo_image (the stock decode environment), with the number-free check's modules; a module named modal_*
# is not shipped to the container, so the image is restated here
gpu_image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
             .apt_install("ffmpeg", "sox", "libsndfile1", "git", "wget", "build-essential")
             .uv_pip_install("torch==2.8.0", "torchaudio==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
             .uv_pip_install("Cython", "packaging")
             .uv_pip_install("nemo_toolkit[asr,cu12]==2.7.3", "huggingface_hub", "safetensors", "jiwer", "pyarrow",
                             "soundfile", "hf_transfer")
             .apt_install("unzip")
             .uv_pip_install("langid", "transformers")
             .env({"HF_HOME": f"{VOL}/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
             .add_local_python_source("common", "ft_eval", "harvest", "ml_balance", "ml_cands", "ml_p3_core", "ml_slav",
                                      "census_common")
             .add_local_file(os.path.join(HERE, "ml_grammar_table.json"), "/root/ml_grammar_table.json"))
ULTRA_REPO, ULTRA_FILE = "ncannings/parakeet-ultra-nemo", "ultra.nemo"  # = modal_app.ULTRA_*
ULTRA_REV = "68d18c58f3d5012f885f4e01039684cd398aff19"


def ultra_path() -> str:
    """modal_app.ultra_path: the stock Ultra checkpoint."""
    from huggingface_hub import hf_hub_download

    dst = f"{VOL}/ultra.nemo"
    if not os.path.exists(dst):
        return hf_hub_download(ULTRA_REPO, ULTRA_FILE, revision=ULTRA_REV)
    return dst


PER_EVENT = 30  # rows per parliament sitting (speaker / topic variety)
MIN_S, MAX_S = 3.0, 20.0


def _refuse(path: str):
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists: give the pass its own --tag")


@app.function(image=cpu_image, volumes={VOL: vol}, secrets=[modal.Secret.from_name("hf")], cpu=2, memory=8192,
              timeout=4 * 3600, max_containers=6)
def recover_lang(lang: str, tag: str = "morerec") -> dict:
    import soundfile as sf

    import harvest
    import ml_balance as B

    out = f"{VOL}/ml/plain/{lang}"
    _refuse(f"{out}/plain_{tag}.jsonl")
    saved = {json.loads(l)["id"] for l in open(f"{out}/plain_more.jsonl")}
    for p in os.listdir(out):  # never cut a row another plain pass already holds
        if p.startswith("plain_") and p.endswith(".jsonl"):
            saved |= {json.loads(l)["id"] for l in open(f"{out}/{p}")}
    want = {}
    for line in open(f"{out}/audit_more.jsonl"):
        r = json.loads(line)
        if r.get("kept") == "plain" and r["id"] not in saved and r.get("src") == "vox":
            want[r["id"]] = r
    wdir = f"{out}/wav_{tag}"
    os.makedirs(wdir, exist_ok=True)
    first = set(harvest.sources(lang))  # the D2 harvest's files; plain_more read the next ones (its default extras)
    srcs = [s for s in harvest.sources(lang, yodas_shards=40 + 80, vox_files=4 + 6) if s not in first and s[0] == "vox"]
    st = {"lang": lang, "wanted": len(want), "cut": 0, "cut_s": 0.0, "number_free": 0, "files": 0}
    rows = []
    for kind, path in srcs:
        if not want:
            break
        st["files"] += 1
        for r in harvest.rows_of(kind, path):
            d = want.pop(r["id"], None)
            if d is None:
                continue
            wav = f"{wdir}/{r['id']}.wav"
            sf.write(wav, r["x"], 16000, subtype="PCM_16")
            nf = B.number_free(d["stock"], lang)
            rows.append({"audio_filepath": wav, "duration": round(r["dur"], 3), "text": d["stock"], "caption": d["cap"],
                         "lang": lang, "src": r["src"], "id": r["id"], "origin": "d2p_rec", "cer_stock": d["cer"],
                         "number_free": nf, "controls": B.lookalikes(d["stock"], lang) if nf else []})
            st["cut"] += 1
            st["cut_s"] += r["dur"]
            st["number_free"] += nf
            if not want:
                break
    with open(f"{out}/plain_{tag}.jsonl", "w") as f:
        f.write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    st["cut_h"] = round(st.pop("cut_s") / 3600, 2)
    st["not_found"] = len(want)
    json.dump(st, open(f"{out}/plain_{tag}_stats.json", "w"), indent=1)
    vol.commit()
    return st


@app.local_entrypoint()
def recover(langs: str = "de,es,fr,it", tag: str = "morerec"):
    for r in recover_lang.starmap([(l, tag) for l in langs.split(",")], return_exceptions=True):
        print(json.dumps(r) if isinstance(r, dict) else f"FAIL {r!r}"[:400])


# ------------------------------------------------------------------------------------------- Granary-VoxPopuli
@app.function(image=cpu_image, volumes={VOL: vol}, secrets=[modal.Secret.from_name("hf")], cpu=2, memory=16384,
              timeout=3 * 3600)
def gv_select(lang: str, years: str, hours: float, tag: str, seed: int = 1) -> dict:
    """Number-free Granary-VoxPopuli rows of `years`, 3-20 s, <= PER_EVENT per sitting, shuffled, up to `hours`."""
    import csv
    import gzip
    import random
    import urllib.request
    from collections import defaultdict

    from huggingface_hub import hf_hub_download

    import ml_balance as B

    out = f"{VOL}/ml/plain/{lang}"
    os.makedirs(out, exist_ok=True)
    _refuse(f"{out}/gv_select_{tag}.json")
    ys = set(years.split(","))
    local = hf_hub_download("nvidia/Granary", f"{lang}/voxpopuli/{lang}_asr.jsonl", repo_type="dataset",
                            local_dir="/tmp/granary")
    cand = defaultdict(list)
    st = {"rows": 0, "in_years": 0, "digit": 0, "not_number_free": 0}
    with open(local) as f:
        for line in f:
            st["rows"] += 1
            y = re.search(r'"utt_id": "(\d{4})', line)
            if not y or y.group(1) not in ys:
                continue
            r = json.loads(line)
            u = r["utt_id"]
            st["in_years"] += 1
            cap = (r.get("text") or "").strip()
            if not cap or re.search(r"\d", cap):
                st["digit"] += 1
                continue
            if not B.number_free(cap, lang):
                st["not_number_free"] += 1
                continue
            cand[u.rsplit("_", 1)[0]].append((u, cap))
    os.remove(local)
    seg_gz = "/tmp/unlabelled_v2.tsv.gz"
    urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/voxpopuli/annotations/unlabelled_v2.tsv.gz", seg_gz)
    dur = {}
    with gzip.open(seg_gz, "rt") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["event_id"] in cand:
                dur[f'{r["event_id"]}_{r["segment_no"]}'] = float(r["end"]) - float(r["start"])
    rng = random.Random(f"{seed}-{lang}")
    per_ev = {}
    for ev, us in cand.items():
        us = [x for x in us if MIN_S <= dur.get(x[0], 0) <= MAX_S]
        rng.shuffle(us)
        per_ev[ev] = us[:PER_EVENT]
    evs = sorted(per_ev)
    rng.shuffle(evs)
    pick, h = [], 0.0
    # round-robin over sittings so the hours spread over many of them
    i = 0
    while h < hours * 3600 and any(per_ev.values()):
        ev = evs[i % len(evs)]
        i += 1
        if per_ev[ev]:
            u, cap = per_ev[ev].pop()
            pick.append({"id": u, "caption": cap, "dur": dur[u]})
            h += dur[u]
    st.update({"events": len(cand), "picked": len(pick), "picked_h": round(h / 3600, 2),
               "events_used": len({p["id"].rsplit("_", 1)[0] for p in pick})})
    json.dump({"stats": st, "rows": pick}, open(f"{out}/gv_select_{tag}.json", "w"), ensure_ascii=False)
    vol.commit()
    return st


@app.function(image=cpu_image, volumes={VOL: vol}, cpu=4, memory=16384, timeout=6 * 3600, max_containers=12)
def gv_cut(lang: str, year: str, rows: list, tag: str) -> dict:
    """Cut the picked clips of one language-year from the VoxPopuli unlabelled session tars (ml_p3.granvox_cut_lang)."""
    import csv
    import gzip
    import subprocess
    import tarfile
    import urllib.error
    import urllib.request

    import numpy as np
    import soundfile as sf

    want = {r["id"]: r for r in rows}
    events = {u.rsplit("_", 1)[0] for u in want}
    seg_gz = "/tmp/unlabelled_v2.tsv.gz"
    urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/voxpopuli/annotations/unlabelled_v2.tsv.gz", seg_gz)
    segs = {}
    with gzip.open(seg_gz, "rt") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["event_id"] in events:
                segs[f'{r["event_id"]}_{r["segment_no"]}'] = (float(r["start"]), float(r["end"]))
    wdir = f"{VOL}/ml/plain/{lang}/wav_{tag}"
    os.makedirs(wdir, exist_ok=True)
    part = f"{VOL}/ml/plain/{lang}/gv_parts_{tag}/{year}.jsonl"
    if os.path.exists(part):  # finished on an earlier attempt
        return {"lang": lang, "year": year, "resumed": True, "rows": [json.loads(l) for l in open(part)]}
    out, st, done = [], {"lang": lang, "year": year, "wanted": len(want), "cut": 0, "retries": 0}, set()
    for tname in (f"{lang}_{year}", f"{lang}_{year}_2"):
        for _ in range(5):
            try:
                resp = urllib.request.urlopen(f"https://dl.fbaipublicfiles.com/voxpopuli/audios/{tname}.tar", timeout=300)
            except urllib.error.HTTPError as e:
                if e.code in (403, 404):
                    break
                st["retries"] += 1
                continue
            except OSError:
                st["retries"] += 1
                continue
            try:
                with tarfile.open(fileobj=resp, mode="r|") as tar:
                    for m in tar:
                        if not m.isfile():
                            continue
                        ev = os.path.basename(m.name).rsplit(".", 1)[0]
                        if ev not in events or ev in done:
                            continue
                        p = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", "16000",
                                            "-f", "s16le", "pipe:1"], input=tar.extractfile(m).read(), capture_output=True)
                        x = np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768
                        for u in [u for u in want if u.rsplit("_", 1)[0] == ev and u in segs]:
                            s0, e0 = segs[u]
                            clip = x[int(s0 * 16000): int(e0 * 16000)]
                            if len(clip) < MIN_S * 16000:
                                continue
                            w = f"{wdir}/{u}.wav"
                            sf.write(w, clip, 16000, subtype="PCM_16")
                            out.append({"audio_filepath": w, "duration": round(len(clip) / 16000, 3),
                                        "caption": want[u]["caption"], "lang": lang, "src": "vox", "id": u,
                                        "origin": "gv_plain"})
                            st["cut"] += 1
                        done.add(ev)
                break
            except (tarfile.ReadError, OSError, EOFError):
                st["retries"] += 1
                continue
        vol.commit()
    os.makedirs(os.path.dirname(part), exist_ok=True)
    with open(part, "w") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    vol.commit()
    return dict(st, rows=out)


@app.function(image=gpu_image, volumes={VOL: vol}, secrets=[modal.Secret.from_name("hf")], gpu="L4", timeout=4 * 3600)
def gv_verify(lang: str, tag: str) -> dict:
    """Stock decode of the cut clips; D2p rule (harvest._process): no digit in caption or stock, CER <= 0.15, LID ok."""
    import glob

    import langid

    import ft_eval
    import harvest
    import ml_balance as B

    out = f"{VOL}/ml/plain/{lang}"
    _refuse(f"{out}/plain_{tag}.jsonl")
    langid.set_languages(harvest.ALL + ["en"])
    rows = [json.loads(l) for p in sorted(glob.glob(f"{out}/gv_parts_{tag}/*.jsonl")) for l in open(p)]
    if not rows:  # a cut stopped early (budget) writes no part file: take the clips it wrote whose length checks out
        import soundfile as sf

        sel = json.load(open(f"{out}/gv_select_{tag}.json"))["rows"]
        for r in sel:
            w = f"{out}/wav_{tag}/{r['id']}.wav"
            try:
                d = sf.info(w).duration if os.path.exists(w) else None
            except RuntimeError:
                d = None
            if d is not None and abs(d - r["dur"]) <= 0.05:
                rows.append({"audio_filepath": w, "duration": round(d, 3), "caption": r["caption"], "lang": lang,
                             "src": "vox", "id": r["id"], "origin": "gv_plain"})
    m = ft_eval.load_model(VOL, "ultra", ultra_path())
    st = {"lang": lang, "rows": len(rows), "lid_reject": 0, "cer_reject": 0, "digit": 0, "kept": 0, "kept_s": 0.0,
          "number_free": 0}
    kept = []
    with open(f"{out}/audit_{tag}.jsonl", "w") as audit:
        for k in range(0, len(rows), 256):
            batch = rows[k:k + 256]
            hyps = [h["text"] for h in ft_eval.decode(m, [r["audio_filepath"] for r in batch], batch_size=32)]
            for r, h in zip(batch, hyps):
                lid = langid.classify(h)[0] if h.strip() else "?"
                cer = harvest._cer(r["caption"], h)
                why = ("lid" if lid != lang else "digit" if re.search(r"\d", h) or re.search(r"\d", r["caption"])
                       else "cer" if cer > 0.15 else None)
                audit.write(json.dumps({"id": r["id"], "cap": r["caption"], "stock": h, "lid": lid, "cer": round(cer, 3),
                                        "kept": None if why else "plain"}, ensure_ascii=False) + "\n")
                if why:
                    st[{"lid": "lid_reject", "digit": "digit", "cer": "cer_reject"}[why]] += 1
                    continue
                nf = B.number_free(h, lang)
                kept.append(dict(r, text=h, cer_stock=round(cer, 3), number_free=nf,
                                 controls=B.lookalikes(h, lang) if nf else []))
                st["kept"] += 1
                st["kept_s"] += r["duration"]
                st["number_free"] += nf
    with open(f"{out}/plain_{tag}.jsonl", "w") as f:
        f.write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in kept))
    st["kept_h"] = round(st.pop("kept_s") / 3600, 2)
    json.dump(st, open(f"{out}/plain_{tag}_stats.json", "w"), indent=1)
    vol.commit()
    return st


@app.local_entrypoint()
def gv(lang: str = "pt", years: str = "2016,2017,2018", hours: float = 45.0, tag: str = "gvplain1", stage: str = "all"):
    if stage in ("all", "select"):
        print(json.dumps(gv_select.remote(lang, years, hours, tag)))
    if stage in ("all", "cut"):
        sel = json.loads(b"".join(vol.read_file(f"ml/plain/{lang}/gv_select_{tag}.json")))["rows"]
        by = {}
        for r in sel:
            by.setdefault(r["id"][:4], []).append(r)
        for r in gv_cut.starmap([(lang, y, rs, tag) for y, rs in sorted(by.items())], return_exceptions=True):
            print(json.dumps({k: v for k, v in r.items() if k != "rows"}) if isinstance(r, dict) else f"FAIL {r!r}"[:400])
    if stage in ("all", "verify"):
        print(json.dumps(gv_verify.remote(lang, tag)))


# ------------------------------------------------------------------------------- false-conversion probe decodes
@app.function(image=gpu_image, volumes={VOL: vol}, secrets=[modal.Secret.from_name("hf")], gpu="L4", timeout=2 * 3600)
def numfree_decode(runs: str, manifest: str = "eval/numfree_heldout_v1/all.jsonl", out_name: str = "numfree_v1") -> dict:
    """Decode a number-free manifest with each run ("ultra" = stock, "<run>@<step>" = exp/<run>/step<step>.pt over the
    stock decoder + joint, as modal_app._load_runs) -> hyps/<run>/<out_name>.jsonl ({id, ref, hyp}), scored locally by
    eval/quality/false_conversion.py. Never overwrites a hyps file."""
    import torch

    import ft_eval

    vol.reload()
    m = ft_eval.load_model(VOL, "ultra", ultra_path())
    stock = {k: v.clone() for k, v in m.state_dict().items() if not k.startswith(("encoder.", "preprocessor."))}
    rows = [json.loads(l) for l in open(f"{VOL}/{manifest}") if l.strip()]
    st = {}
    for run in runs.split(","):
        dst = f"{VOL}/hyps/{run}/{out_name}.jsonl"
        _refuse(dst)
        if run == "ultra":
            n_enc = ft_eval.load_over_stock(m, stock, {})
        else:
            r, _, step = run.partition("@")
            n_enc = ft_eval.load_over_stock(m, stock, torch.load(f"{VOL}/exp/{r}/step{step}.pt", map_location="cpu"))
        assert n_enc == 0, f"{run}: encoder tensors (unfreeze_top): this probe decodes with the stock encoder only"
        hyps = ft_eval.decode(m, [x["audio_filepath"] for x in rows])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w") as f:
            for x, h in zip(rows, hyps):
                f.write(json.dumps({"id": x["id"], "lang": x["lang"], "ref": x["text"], "hyp": h["text"]},
                                   ensure_ascii=False) + "\n")
        st[run] = len(rows)
        vol.commit()
    return st


@app.local_entrypoint()
def numfree(runs: str = "ultra,w3soup_a0.5@0", manifest: str = "eval/numfree_heldout_v1/all.jsonl",
            out_name: str = "numfree_v1"):
    print(json.dumps(numfree_decode.remote(runs, manifest, out_name)))
