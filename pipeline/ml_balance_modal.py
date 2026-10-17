"""Modal jobs for the data-balance pieces (ml_balance). CPU only; nothing here trains or rebuilds the mix.

plain_recover : number-free rows the D2 harvest already decoded with STOCK (d2/<lang>/audit.jsonl kept == "plain": no
                digit in caption or stock, CER <= 0.15, text LID ok) but never saved because the 1.2 h quota or the
                60% source cap stopped it. Re-reads the same VoxPopuli / YODAS-Granary files (harvest.sources), cuts
                those rows and writes /vol/ml/plain/<lang>/plain_<tag>.jsonl + wav/ (label = the audited stock text,
                exactly the D2p rule; no GPU, no new decode). A dry run writes under .../dry/.

    modal run ml_balance_modal.py::plain_recover --langs de,es,fr,it,nl --tag rec
"""

import json
import os

import modal

VOL = "/vol"
vol = modal.Volume.from_name("parakeet-spoken-form")
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg", "libsndfile1")
         .pip_install("pyarrow", "huggingface_hub", "hf_transfer", "soundfile", "librosa")
         .add_local_python_source("harvest", "ml_balance", "ml_p3_core", "ml_cands", "ml_slav")
         .add_local_file("ml_grammar_table.json", "/root/ml_grammar_table.json"))
app = modal.App("parakeet-balance")


@app.function(image=image, volumes={VOL: vol}, secrets=[modal.Secret.from_name("hf")], cpu=2, memory=8192,
              timeout=4 * 3600, max_containers=6)
def plain_recover_lang(lang: str, tag: str = "rec", limit: int = 0, dry: bool = False) -> dict:
    import soundfile as sf

    import harvest
    import ml_balance as B

    want = {}
    for line in open(f"{VOL}/d2/{lang}/audit.jsonl"):
        r = json.loads(line)
        if r.get("kept") == "plain" and r.get("src") in ("vox", "yodas"):
            want[r["id"]] = r
    saved = {json.loads(l)["id"] for l in open(f"{VOL}/d2/{lang}/plain.jsonl")}
    want = {k: v for k, v in want.items() if k not in saved}
    out = f"{VOL}/ml/plain/{lang}" + ("/dry" if dry else "")
    os.makedirs(f"{out}/wav", exist_ok=True)
    st = {"lang": lang, "audited_plain_unsaved": len(want), "cut": 0, "cut_s": 0.0, "number_free": 0, "files": 0}
    rows = []
    for kind, path in harvest.sources(lang):
        if kind == "fleurs" or not want or (limit and st["cut"] >= limit):
            continue
        if not any(v["src"] == kind for v in want.values()):  # e.g. every audited YODAS plain row was saved already
            continue
        st["files"] += 1
        for r in harvest.rows_of(kind, path):
            d = want.pop(r["id"], None)
            if d is None:
                continue
            wav = f"{out}/wav/{r['id']}.wav"
            sf.write(wav, r["x"], 16000, subtype="PCM_16")
            nf = B.number_free(d["stock"], lang)
            rows.append({"audio_filepath": wav, "duration": round(r["dur"], 3), "text": d["stock"], "caption": d["cap"],
                         "lang": lang, "src": r["src"], "id": r["id"], "origin": "d2p_rec", "cer_stock": d["cer"],
                         "number_free": nf, "controls": B.lookalikes(d["stock"], lang) if nf else []})
            st["cut"] += 1
            st["cut_s"] += r["dur"]
            st["number_free"] += nf
            if limit and st["cut"] >= limit:
                break
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
def plain_recover(langs: str = "de,es,fr,it,nl", tag: str = "rec", limit: int = 0, dry: bool = False):
    for r in plain_recover_lang.starmap([(l, tag, limit, dry) for l in langs.split(",")], return_exceptions=True):
        print(json.dumps(r) if isinstance(r, dict) else f"FAIL {r!r}"[:400])
