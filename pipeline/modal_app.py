"""Modal entry points for the spoken-form fine-tune.

Run from the repo root, e.g.:
    modal run pipeline/modal_app.py::env_check
    modal run --detach pipeline/modal_app.py::killtest
"""

import os

import modal

CARRIERS_FILE = os.path.join(os.path.dirname(__file__), "carriers.txt")

app = modal.App("parakeet-spoken-form")
vol = modal.Volume.from_name("parakeet-spoken-form", create_if_missing=True, version=2)
VOL = "/vol"
ULTRA_REPO, ULTRA_FILE = "ncannings/parakeet-ultra-nemo", "ultra.nemo"
ULTRA_REV = "68d18c58f3d5012f885f4e01039684cd398aff19"

nemo_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("ffmpeg", "sox", "libsndfile1", "git", "wget", "build-essential")
    .uv_pip_install(
        "torch==2.8.0", "torchaudio==2.8.0", index_url="https://download.pytorch.org/whl/cu128"
    )
    .uv_pip_install("Cython", "packaging")
    .uv_pip_install(
        "nemo_toolkit[asr,cu12]==2.7.3",
        "huggingface_hub",
        "safetensors",
        "jiwer",
        "pyarrow",
        "soundfile",
        "hf_transfer",
    )
    .apt_install("unzip")
    .uv_pip_install("langid", "transformers")
    .env({"HF_HOME": f"{VOL}/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("common", "data_e22", "data_replay", "data_fleurs", "ft_train", "ft_eval", "rescore", "preflight",
                              "labeller", "readings", "label_gpu", "harvest", "beds", "veto",
                              "mix_build")
)

tts_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("espeak-ng", "libsndfile1")
    .uv_pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("kokoro==0.9.4", "misaki[en]>=0.9.4", "transformers>=4.45", "soundfile", "scipy",
                    "numpy", "huggingface_hub", "pip")
    .run_commands("python -m spacy download en_core_web_sm")
    # Per-container cache: parallel shards writing one shared cache read each other's partial files.
    .env({"HF_HOME": "/root/hf"})
    .add_local_python_source("common", "readings", "gen", "voice")
    .add_local_file(CARRIERS_FILE, "/root/carriers.txt")
    .add_local_file(os.path.join(os.path.dirname(__file__), "carriers_year.txt"), "/root/carriers_year.txt")
    .add_local_file(os.path.join(os.path.dirname(__file__), "carriers_year_b.txt"), "/root/carriers_year_b.txt")
    .add_local_file(os.path.join(os.path.dirname(__file__), "carriers_d9.txt"), "/root/carriers_d9.txt")
)


def ultra_path() -> str:
    import os
    from huggingface_hub import hf_hub_download

    dst = f"{VOL}/ultra.nemo"
    if not os.path.exists(dst):
        src = hf_hub_download(ULTRA_REPO, ULTRA_FILE, revision=ULTRA_REV)
        os.symlink(src, dst)
        vol.commit()
    return dst


@app.function(image=nemo_image, volumes={VOL: vol}, gpu="L4", timeout=1800)
def env_check():
    """Load Ultra, freeze the encoder, report trainable params and one decode."""
    import torch
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.losses.rnnt import RNNTLoss  # noqa: F401  (numba TDT kernels)

    m = nemo_asr.models.ASRModel.restore_from(ultra_path(), map_location="cpu")
    print("class", type(m).__name__, "torch", torch.__version__, "cuda", torch.cuda.is_available())
    m.encoder.freeze()
    groups = {}
    for n, p in m.named_parameters():
        if p.requires_grad:
            k = ".".join(n.split(".")[:2])
            groups[k] = groups.get(k, 0) + p.numel()
    print("trainable", sum(groups.values()), groups)
    print("train_ds.text_field", m.cfg.train_ds.get("text_field"))
    print("decoding", m.cfg.decoding.get("strategy"), m.cfg.decoding.greedy)


@app.function(image=nemo_image, volumes={VOL: vol}, cpu=4, memory=16384, timeout=3600)
def prep_e22():
    import data_e22

    data_e22.prepare(VOL)
    vol.commit()


@app.function(image=nemo_image, volumes={VOL: vol}, gpu="L4", timeout=3600)
def transcribe(run: str, manifest: str, out_name: str):
    """Decode a manifest with Ultra (run='ultra') or a trained run's checkpoint."""
    import ft_eval

    ft_eval.transcribe(VOL, run, manifest, out_name, ultra_path())
    vol.commit()


@app.function(image=nemo_image, volumes={VOL: vol}, cpu=8, memory=16384, timeout=3 * 3600)
def prep_replay():
    import data_replay

    data_replay.prepare(VOL)
    vol.commit()


@app.function(image=nemo_image, volumes={VOL: vol}, cpu=8, memory=32768, timeout=3 * 3600)
def prep_fleurs(hours_per_lang: float = 0.4):
    import data_fleurs

    data_fleurs.prepare(VOL, hours_per_lang)
    vol.commit()


@app.function(image=tts_image, volumes={VOL: vol}, gpu="L4", timeout=3600)
def voice_shard(rows: list, out_dir: str) -> list:
    import voice

    out = voice.voice_rows(rows, out_dir)
    vol.commit()
    return out


@app.function(image=tts_image, volumes={VOL: vol}, timeout=3 * 3600)
def make_tts(name: str, n_rows: int, seed: int = 1, shard: int = 200, multi_frac: float = 0.0, slots: str = "",
             sub_one: bool = False, control_frac: float = 0.1, carriers: str = "carriers.txt", speed: str = ""):
    """Generate rows, voice them in parallel shards, write train/dev manifests under /vol/tts/<name>.
    `slots` + `sub_one`: a targeted supplement (only those carriers; every scaled amount/percent below one)."""
    import json
    import os

    import gen
    import readings

    if sub_one:
        readings.SUB_ONE_P = 1.0
    rows = list(gen.rows(n_rows, seed, multi_frac=multi_frac, slots=slots, control_frac=control_frac,
                         carriers=carriers))
    if speed:  # "lo,hi": a speech-rate range for this set (v12 year supplement: fast speech)
        lo, hi = (float(x) for x in speed.split(","))
        for row in rows:
            row["speed_lo"], row["speed_hi"] = lo, hi
    shards = [rows[i:i + shard] for i in range(0, len(rows), shard)]
    out_dir = f"{VOL}/tts/{name}"
    os.makedirs(out_dir, exist_ok=True)
    done = []
    for part in voice_shard.map(shards, kwargs={"out_dir": f"{out_dir}/wav"}):
        done += part
    for split in ("train", "dev"):
        sel = [r for r in done if r["split"] == split]
        with open(f"{out_dir}/{split}.jsonl", "w") as f:
            for r in sel:
                f.write(json.dumps(r) + "\n")
        print(f"{name}/{split}: {len(sel)} clips, {sum(r['duration'] for r in sel) / 3600:.2f} h")
    vol.commit()


@app.function(image=nemo_image, volumes={VOL: vol}, gpu="A100-40GB", cpu=8, memory=32768,
              timeout=2 * 3600)
def killtest(run: str = "kill1", max_steps: int = 150, tts: str = "smoke", lr: float = 1e-4,
             lspc_h: float = 3.0, numwords_h: float = 0.7, warmup: int = 20):
    """Day-1 kill test: train briefly, then decode E22 number utterances, TTS dev and LS dev."""
    import ft_eval
    import ft_train

    vol.reload()
    m = ft_train.train(VOL, ultra_path(), run,
                       mix=[(f"tts/{tts}/train.jsonl", 0), ("replay/lspc_train.jsonl", lspc_h),
                            ("replay/lspc_train_numwords.jsonl", numwords_h)],
                       max_steps=max_steps, lr=lr, warmup=warmup)
    vol.commit()
    m = m.cuda().eval()
    ft_eval.transcribe_loaded(VOL, m, run, "e22/numbers.jsonl", "e22_numbers")
    ft_eval.transcribe_loaded(VOL, m, run, f"tts/{tts}/dev.jsonl", "tts_dev")
    ft_eval.transcribe_loaded(VOL, m, run, "replay/lspc_dev.jsonl", "lspc_dev", limit=600)
    vol.commit()


@app.function(image=nemo_image, volumes={VOL: vol}, gpu="L4", timeout=3600)
def baseline(tts: str = "smoke", run: str = "ultra", full: bool = False):
    """Stock Ultra on the same sets as the kill test (full=True: the run() sets)."""
    import ft_eval

    vol.reload()
    m = ft_eval.load_model(VOL, "ultra", ultra_path())
    if full:
        ft_eval.eval_sets(VOL, m, run, tts)
        vol.commit()
        return
    ft_eval.transcribe_loaded(VOL, m, run, "e22/numbers.jsonl", "e22_numbers")
    ft_eval.transcribe_loaded(VOL, m, run, f"tts/{tts}/dev.jsonl", "tts_dev")
    ft_eval.transcribe_loaded(VOL, m, run, "replay/lspc_dev.jsonl", "lspc_dev", limit=600)
    vol.commit()


@app.function(image=nemo_image, volumes={VOL: vol}, cpu=4, memory=16384, timeout=1800)
def export_weights(run: str = "ultra", step: int = 0):
    """Decoder + joint tensors (fp32, NeMo names) as safetensors, for swap_weights.py."""
    import os

    import torch
    from safetensors.torch import save_file

    vol.reload()
    if run == "ultra":
        sd = torch.load(ultra_state(), map_location="cpu")
    elif step:
        sd = torch.load(f"{VOL}/exp/{run}/step{step}.pt", map_location="cpu")
        run = f"{run}@{step}"
    else:
        sd = torch.load(f"{VOL}/exp/{run}/decoder_joint.pt", map_location="cpu")
    keep = {k: v.float().contiguous() for k, v in sd.items() if k.startswith(("decoder.", "joint."))}
    os.makedirs(f"{VOL}/export/{run}", exist_ok=True)
    save_file(keep, f"{VOL}/export/{run}/decoder_joint.safetensors")
    print(run, len(keep), "tensors", sum(v.numel() for v in keep.values()), "params")
    vol.commit()


@app.function(image=nemo_image, volumes={VOL: vol}, cpu=4, memory=16384, timeout=1800)
def soup(src: str, alphas: str, name: str):
    """Interpolate decoder + joint between stock Ultra and `src` ("<run>@<step>"): weights =
    stock + alpha * (fine-tuned - stock), one checkpoint per alpha at exp/<name>/step<100*alpha>.pt,
    so `eval_ckpts` and `export_weights --step` take them as they are."""
    import os

    import torch

    vol.reload()
    run, _, step = src.partition("@")
    ft = torch.load(f"{VOL}/exp/{run}/step{step}.pt", map_location="cpu")
    base = torch.load(ultra_state(), map_location="cpu")
    trained = {k for k in ft if k.startswith(("decoder.", "joint."))}
    assert trained == set(base), (sorted(trained ^ set(base))[:5])
    os.makedirs(f"{VOL}/exp/{name}", exist_ok=True)
    for a in (float(x) for x in alphas.split(",")):
        sd = {k: (base[k].float() + a * (v.float() - base[k].float())).to(v.dtype) if k in base else v
              for k, v in ft.items()}
        torch.save(sd, f"{VOL}/exp/{name}/step{round(100 * a)}.pt")
        print(f"{name}: alpha {a} -> step{round(100 * a)}.pt")
    vol.commit()


@app.function(image=nemo_image, volumes={VOL: vol}, cpu=4, memory=16384, timeout=1800)
def seedsoup(srcs: str, name: str, step: int = 1500, weights: str = ""):
    """Weighted average (uniform by default) of several checkpoints: "v10a,v10as2" (each at `step`) or
    "v10a@1000,v10a@1500"; saved as exp/<name>/step<step>.pt so `eval_ckpts` and `export_weights` take it."""
    import os

    import torch

    vol.reload()
    items = [x.partition("@") for x in srcs.split(",")]
    sds = [torch.load(f"{VOL}/exp/{r}/step{s or step}.pt", map_location="cpu") for r, _, s in items]
    assert all(set(sd) == set(sds[0]) for sd in sds), "checkpoints differ in keys"
    w = [float(x) for x in weights.split(",")] if weights else [1.0 / len(sds)] * len(sds)
    assert len(w) == len(sds) and abs(sum(w) - 1) < 1e-6, w
    out = {k: sum(wi * sd[k].float() for wi, sd in zip(w, sds)).to(sds[0][k].dtype)
           if sds[0][k].is_floating_point() else sds[0][k] for k in sds[0]}
    os.makedirs(f"{VOL}/exp/{name}", exist_ok=True)
    torch.save(out, f"{VOL}/exp/{name}/step{step}.pt")
    print(f"{name}: {srcs} weights {w} -> step{step}.pt")
    vol.commit()


@app.function(image=nemo_image, volumes={VOL: vol}, gpu="L4", timeout=1800)
def score_texts(run: str = "v10pt", items: str = "yeardrop/items.json"):
    """Greedy hypothesis + TDT likelihood of candidate texts per clip (rescore.score_texts)."""
    import rescore

    vol.reload()
    for r in run.split(","):
        rescore.score_texts(VOL, r, ultra_path(), f"{VOL}/{items}", f"{VOL}/yeardrop/scores_{r}.json")
    vol.commit()


@app.function(image=nemo_image, volumes={VOL: vol}, gpu="L4", timeout=3600)
def seam_decode(run: str = "v10pt", items: str = "seamb1/items.json", out: str = "seamb1/decoded.json"):
    """Seam re-decode placement study (rescore.cut_decode): cut clips from full wavs, greedy-decode."""
    import rescore

    vol.reload()
    rescore.cut_decode(VOL, run, ultra_path(), f"{VOL}/{items}", f"{VOL}/{out}")
    vol.commit()


@app.function(image=nemo_image, volumes={VOL: vol}, gpu="L4", timeout=3600)
def year_beam(run: str = "v10pt", items: str = "yearbeam/items.json", beam: int = 4):
    """Greedy vs beam on the same clips (rescore.greedy_vs_beam)."""
    import rescore

    vol.reload()
    rescore.greedy_vs_beam(VOL, run, ultra_path(), f"{VOL}/{items}", f"{VOL}/yearbeam/{run}_b{beam}.json", beam)
    vol.commit()


def ultra_state() -> str:
    """Ultra's own decoder + joint, extracted once from the .nemo."""
    import os
    import tarfile

    import torch

    path = f"{VOL}/export/ultra/state.pt"
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with tarfile.open(ultra_path()) as t:
            member = next(m for m in t.getmembers() if m.name.endswith("model_weights.ckpt"))
            sd = torch.load(t.extractfile(member), map_location="cpu")
        torch.save({k: v for k, v in sd.items() if k.startswith(("decoder.", "joint."))}, path)
    return path


@app.function(image=nemo_image, volumes={VOL: vol}, gpu="L4", timeout=3600)
def filter_tts(name: str):
    import ft_eval

    vol.reload()
    m = ft_eval.load_model(VOL, "ultra", ultra_path())
    ft_eval.filter_tts(VOL, m, name)
    vol.commit()


@app.function(image=nemo_image, volumes={VOL: vol}, gpu="A100-40GB", cpu=8, memory=32768,
              timeout=3 * 3600)
def run(run: str, tts: str, max_steps: int = 1000, lr: float = 1e-4, lspc_h: float = 14.0,
        numwords_h: float = 5.0, fleurs_h: float = 3.0, warmup: int = 50, hold_blank_duration: bool = False,
        save_every: int = 250, spec_augment: bool = True, kl_weight: float = 0.0,
        kl_replay_only: bool = False, init_from: str = "", tts_h: float = 0.0, batch_duration: float = 600,
        icsi_h: float = 0.0, kl_terms: str = "blank,dur", seed: int = 1,
        kl_exempt: str = "", icsi_ovl_h: float = 0.0, icsi_ovl_set: str = "icsi_ovl",
        kl_blank_scale: float = 1.0, kl_punct: float = 0.0, kl_punct_rows: str = "all"):
    """Train, then score every saved checkpoint on the evaluation sets. `tts_h` < 0 leaves the TTS
    rows out (replay-only consolidation)."""
    import torch

    import ft_eval
    import ft_train

    vol.reload()
    m = ft_train.train(VOL, ultra_path(), run,
                       mix=([(f"tts/{tts}/train_ok.jsonl", tts_h)] if tts_h >= 0 else [])
                       + [("replay/lspc_train.jsonl", lspc_h),
                          ("replay/lspc_train_numwords.jsonl", numwords_h),
                          ("replay/fleurs_train_multi.jsonl", fleurs_h)]
                       + ([("replay/icsi_train.jsonl", icsi_h)] if icsi_h else [])
                       + ([(f"replay/{icsi_ovl_set}_train.jsonl", icsi_ovl_h)] if icsi_ovl_h else []),
                       max_steps=max_steps, lr=lr, warmup=warmup, freeze_blank_duration=hold_blank_duration,
                       save_every=save_every, on_save=vol.commit, spec_augment=spec_augment,
                       kl_weight=kl_weight, kl_replay_only=kl_replay_only, init_from=init_from,
                       batch_duration=batch_duration, kl_terms=kl_terms, seed=seed, kl_exempt=kl_exempt,
                       kl_blank_scale=kl_blank_scale, kl_punct=kl_punct, kl_punct_rows=kl_punct_rows)
    vol.commit()
    del m
    eval_ckpts.local(run, ",".join(str(s) for s in range(save_every, max_steps + 1, save_every)), tts,
                     reload=False)


@app.function(image=nemo_image, volumes={VOL: vol}, gpu="L4", timeout=3 * 3600)
def eval_ckpts(run: str, steps: str, tts: str, reload: bool = True):
    """Score checkpoints, each in a FRESH model: reusing one model across load_state_dict calls
    broke every second checkpoint's decode (one token, then nothing), 2026-10-07."""
    import torch

    import ft_eval

    if reload:  # not when called in the training container: its data loaders keep files open
        vol.reload()
    for step in steps.split(","):
        m = ft_eval.load_model(VOL, "ultra", ultra_path())
        m.load_state_dict(torch.load(f"{VOL}/exp/{run}/step{step}.pt", map_location="cpu"), strict=False)
        ft_eval.eval_sets(VOL, m.cuda().eval(), run, tts, suffix=f"@{step}")
        vol.commit()
        del m
        torch.cuda.empty_cache()


@app.function(image=nemo_image, volumes={VOL: vol}, gpu="L4", timeout=3600)
def rescore_dryrun(run: str = "v9pr30", clips: str = "rescore/clips"):
    """Number re-check dry run (rescore.py): one-edit alternatives of every spoken number, scored by the
    candidate and stock decoder+joint on the same audio. Writes /vol/rescore/<run>.json."""
    import os

    import rescore

    vol.reload()
    os.makedirs(f"{VOL}/rescore", exist_ok=True)
    rescore.run(VOL, run, ultra_path(), f"{VOL}/{clips}", f"{VOL}/rescore/{run}.json")
    vol.commit()


@app.function(image=tts_image, volumes={VOL: vol}, timeout=600)
def merge_tts(name: str, parts: str):
    """Concatenate filtered TTS sets (`parts` comma-separated) into /vol/tts/<name>/{train,dev}_ok.jsonl."""
    import os

    vol.reload()
    os.makedirs(f"{VOL}/tts/{name}", exist_ok=True)
    for split in ("train", "dev"):
        n = 0
        with open(f"{VOL}/tts/{name}/{split}_ok.jsonl", "w") as out:
            for p in parts.split(","):
                for line in open(f"{VOL}/tts/{p}/{split}_ok.jsonl"):
                    out.write(line)
                    n += 1
        print(f"{name}/{split}_ok: {n} clips")
    vol.commit()

@app.function(image=nemo_image, volumes={VOL: vol}, gpu="L4", timeout=1800)
def debug_ckpt(run: str = "v3free", steps: str = "500,750"):
    """Load each checkpoint into a FRESH model and decode 20 LS dev clips; report weight stats."""
    import json

    import torch

    import ft_eval

    vol.reload()
    ultra = torch.load(ultra_state(), map_location="cpu")
    rows = [json.loads(l) for l in open(f"{VOL}/replay/lspc_dev.jsonl")][:20]
    for step in steps.split(","):
        sd = torch.load(f"{VOL}/exp/{run}/step{step}.pt", map_location="cpu")
        bad = [k for k, v in sd.items() if v.is_floating_point() and not torch.isfinite(v).all()]
        drift = {k: round(float((sd[k].float() - ultra[k].float()).norm() / (ultra[k].float().norm() + 1e-9)), 4)
                 for k in ultra if k in sd}
        print(step, "keys", len(sd), "nonfinite", bad[:5], "max rel drift",
              sorted(drift.items(), key=lambda x: -x[1])[:4])
        m = ft_eval.load_model(VOL, "ultra", ultra_path())
        m.load_state_dict(sd, strict=False)
        hyps = ft_eval.decode(m.cuda().eval(), [r["audio_filepath"] for r in rows])
        print(step, "fresh-model decode:", [h["text"][:50] for h in hyps[:3]])


@app.function(image=nemo_image, volumes={VOL: vol}, cpu=2, memory=8192, timeout=600)
def shift_blank(run: str, step: int, deltas: str = "0.5,1.0,1.5"):
    """Calibration variants: add `delta` to the blank logit's bias (joint output row 8192), saved as
    exp/<run>_b<delta>/step<step>.pt. Fine-tuning lowered the model's blank confidence; this moves
    it back without retraining."""
    import os

    import torch

    vol.reload()
    sd = torch.load(f"{VOL}/exp/{run}/step{step}.pt", map_location="cpu")
    for d in deltas.split(","):
        out = dict(sd)
        b = sd["joint.joint_net.2.bias"].clone()
        b[8192] += float(d)
        out["joint.joint_net.2.bias"] = b
        os.makedirs(f"{VOL}/exp/{run}_b{d}", exist_ok=True)
        torch.save(out, f"{VOL}/exp/{run}_b{d}/step{step}.pt")
        print(f"{run}_b{d}: blank bias {float(sd['joint.joint_net.2.bias'][8192]):.3f} -> {float(b[8192]):.3f}")
    vol.commit()


@app.function(image=nemo_image, volumes={VOL: vol}, gpu="L4", timeout=1800)
def preflight(run: str, kl_weight: float = 7.0, tts: str = "", steps: int = 0, lr: float = 1e-4,
              terms: str = "blank,gnorm,durspan", lspc_h: float = 0.25, numwords_h: float = 0.5,
              fleurs_h: float = 0.25, tts_h: float = 0.25, seed: int = 1):
    """v10b pre-flight (about 3-4 min on an L4 with steps=0): span-mask audit + blank/duration
    drift of the existing checkpoint against stock (stage A); with steps>0 a micro-train with
    the v10b loss from the checkpoint plus year/money decodes (stage B, exp/<run>_pf)."""
    import preflight as pf

    vol.reload()
    pf.main(VOL, ultra_path(), run, kl_weight, tts=tts, steps=steps, lr=lr, terms=terms,
            lspc_h=lspc_h, numwords_h=numwords_h, fleurs_h=fleurs_h, tts_h=tts_h, seed=seed)
    vol.commit()



# ---- Real-speech retrain (docs/plans/spoken-form-real-speech-retrain-plan-2026-10-12.md) ----

scan_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("pyarrow", "huggingface_hub", "hf_transfer")
    .add_local_python_source("common", "s0_scan", "fetch", "labeller", "readings")
)
hf_secret = modal.Secret.from_name("hf")


@app.function(image=scan_image, volumes={VOL: vol}, secrets=[hf_secret], cpu=1, memory=4096, timeout=1800,
              retries=2, max_containers=64)
def s0_scan_shard(path: str, lang: str) -> dict:
    import s0_scan

    st = s0_scan.scan_shard(VOL, path, lang)
    vol.commit()
    return st


@app.function(image=scan_image, volumes={VOL: vol}, secrets=[hf_secret], timeout=4 * 3600)
def s0_scan(langs: str = "en,bg,cs,da,de,el,es,et,fi,fr,hr,hu,it,lt,lv,nl,pl,pt,ro,ru,sk,sv,uk",
            en_shards: int = 150, big_shards: int = 25):
    """Caption-only scan (S0): number density per language, flagged rows to /vol/s0/scan."""
    import json

    import s0_scan

    jobs = s0_scan.pick_shards(langs.split(","), en_shards, big_shards)
    print(f"{len(jobs)} shards")
    stats = []
    for st in s0_scan_shard.starmap(jobs, return_exceptions=True):
        if isinstance(st, Exception):
            print("FAIL", repr(st)[:200])
            continue
        stats.append(st)
    vol.reload()
    agg = {}
    for st in stats:
        a = agg.setdefault(st["lang"], {k: 0 for k in st if k not in ("shard", "lang")} | {"shards": 0})
        a["shards"] += 1
        for k, v in st.items():
            if k not in ("shard", "lang"):
                a[k] += v
    for lang, a in sorted(agg.items()):
        print(lang, json.dumps({k: round(v, 1) if isinstance(v, float) else v for k, v in a.items()}))
    json.dump(agg, open(f"{VOL}/s0/scan_stats.json", "w"), indent=1)
    vol.commit()


@app.function(image=scan_image, volumes={VOL: vol}, secrets=[hf_secret], cpu=2, memory=8192,
              timeout=3600, retries=2, max_containers=40)
def s3_fetch_shard(path: str, lang: str) -> dict:
    import fetch

    vol.reload()
    st = fetch.fetch_shard(VOL, path, lang)
    vol.commit()
    return st


@app.function(image=scan_image, volumes={VOL: vol}, secrets=[hf_secret], timeout=6 * 3600)
def s3_fetch(lang: str = "en", limit: int = 0):
    """Audio for every scanned shard of `lang` (S3)."""
    import glob
    import json

    vol.reload()
    stats = [json.load(open(f)) for f in sorted(glob.glob(f"{VOL}/s0/scan/{lang}/*.stats"))]
    jobs = [(st["shard"], lang) for st in stats][: limit or None]
    print(f"{len(jobs)} shards")
    tot_h = tot = 0
    for st in s3_fetch_shard.starmap(jobs, return_exceptions=True):
        if isinstance(st, Exception):
            print("FAIL", repr(st)[:300])
            continue
        tot += st.get("kept", 0)
        tot_h += st.get("h", 0)
        print(json.dumps(st))
    print(f"fetched {tot} rows, {tot_h:.1f} h")


@app.cls(image=nemo_image, volumes={VOL: vol}, secrets=[hf_secret], gpu="L4", timeout=3 * 3600, max_containers=8,
         retries=1)
class Labeller:
    @modal.enter()
    def load(self):
        import label_gpu

        vol.reload()
        self.sc = label_gpu.Scorer(VOL, ultra_path())

    @modal.method()
    def validate(self, items: str, out: str) -> int:
        import label_gpu

        n = label_gpu.validate(VOL, self.sc, f"{VOL}/{items}", f"{VOL}/{out}")
        vol.commit()
        return n

    @modal.method()
    def shard(self, man: str, tag: str, keep_audio: bool = True, only_pct: bool = False) -> dict:
        import os

        import label_gpu

        if os.path.exists(f"{VOL}/d1/crops/{tag}.jsonl"):
            return {"shard": tag, "skipped": True}
        st = label_gpu.label_shard(VOL, self.sc, man, tag, keep_audio, only_pct)
        vol.commit()
        return st


@app.function(image=scan_image, volumes={VOL: vol}, timeout=12 * 3600)
def s3_label(lang: str = "en", limit: int = 0, only: str = "", only_pct: bool = False):
    """Relabel every fetched English shard (S3); `only` = comma list of shard tags."""
    import glob
    import json
    import os

    vol.reload()
    mans = sorted(glob.glob(f"{VOL}/yodas/{lang}/manifests/*.jsonl"))
    if only:
        mans = [m for m in mans if os.path.basename(m)[:-6] in only.split(",")]
    mans = mans[: limit or None]
    print(f"{len(mans)} shards")
    tot = {"rows": 0, "crops": 0, "crop_h": 0.0, "dropped": 0}
    suffix = "_pct" if only_pct else ""
    jobs = [(m, os.path.basename(m)[:-6] + suffix, True, only_pct) for m in mans]
    for st in Labeller().shard.starmap(jobs, return_exceptions=True):
        if isinstance(st, Exception):
            print("FAIL", repr(st)[:400])
            continue
        print(json.dumps(st))
        for k in tot:
            tot[k] += st.get(k, 0)
    print("TOTAL", json.dumps(tot))


@app.function(image=nemo_image, volumes={VOL: vol}, secrets=[hf_secret], gpu="L4", timeout=4 * 3600, max_containers=8,
              retries=1)
def d2_harvest_lang(lang: str, digit_h: float = 1.2, plain_h: float = 1.2) -> dict:
    import os

    os.environ["HF_HOME"] = "/tmp/hf"  # HF/xet logs on the volume block vol.commit/reload across inputs
    import ft_eval
    import harvest

    m = ft_eval.load_model(VOL, "ultra", ultra_path())
    st = harvest.harvest(VOL, m, lang, digit_h, plain_h)
    vol.commit()
    return st


@app.function(image=scan_image, volumes={VOL: vol}, timeout=8 * 3600)
def d2_harvest(langs: str = "bg,cs,da,de,el,es,et,fi,fr,hr,hu,it,lt,lv,nl,pl,pt,ro,ru,sk,sl,sv,uk"):
    import json

    tot = 0.0
    for st in d2_harvest_lang.map(langs.split(","), return_exceptions=True):
        if isinstance(st, Exception):
            print("FAIL", repr(st)[:400])
            continue
        tot += st["digit_h"]
        print(json.dumps(st))
    print(f"digit hours total {tot:.1f}")


@app.local_entrypoint()
def validate_labeller(items: str = "lv/items.json", out: str = "lv/out.json"):
    print(Labeller().validate.remote(items, out))


@app.function(image=nemo_image, volumes={VOL: vol}, secrets=[hf_secret], gpu="L4", cpu=4, memory=32768,
              timeout=4 * 3600)
def build_beds(which: str = "vox,fleurs,noise"):
    import json
    import os

    os.environ["HF_HOME"] = "/tmp/hf"

    import beds
    import ft_eval

    vol.reload()
    if "noise" in which:
        print("noise", beds.noise(VOL))
        vol.commit()
    if "fleurs" in which:
        print("fleurs", json.dumps(beds.fleurs(VOL)))
        vol.commit()
    if "e21x" in which:
        print("e21x", json.dumps(beds.e21x(VOL)))
        vol.commit()
    if "vox" in which:
        m = ft_eval.load_model(VOL, "ultra", ultra_path())
        print("vox", json.dumps(beds.vox(VOL, m)))
        vol.commit()


mix_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .uv_pip_install("numpy", "scipy", "soundfile", "librosa")
    .add_local_python_source("common", "mix_build")
)


@app.function(image=mix_image, volumes={VOL: vol}, cpu=8, memory=32768, timeout=8 * 3600)
def build_mix(seed: int = 1):
    import json

    import mix_build

    vol.reload()
    print(json.dumps(mix_build.build(VOL, seed=seed), indent=1))
    vol.commit()


RETRAIN_K = dict(batch_duration=400, spec_augment=False, kl_weight=10.0, kl_replay_only=True, kl_punct=30.0,
                 kl_punct_rows="replay", save_every=500)


def retrain_mix(arm: str, tts_h: float = 6.0) -> list:
    """The V2 SPEC mix per arm. main/stockinit: everything; abl_en: v10pt mix + English real only;
    abl_nonen: v10pt mix + non-English real only. Hours <= 0 = all rows of that manifest."""
    base = [("tts/v10pt/train_ok.jsonl", tts_h, "free"), ("tts/d9sup/train_ok.jsonl", 0, "free"),
            ("replay/lspc_train.jsonl", 35.0, "anchor"), ("replay/lspc_train_numwords.jsonl", 5.0, "anchor"),
            ("replay/fleurs_train_multi.jsonl", 3.0, "nonen"), ("replay/icsi_train.jsonl", 0, "anchor")]
    en = [("mix/d1.jsonl", 0, "span"), ("mix/d1p.jsonl", 0, "anchor"), ("mix/d10n.jsonl", 0, "span"),
          ("mix/d10p.jsonl", 0, "anchor")]
    nonen = [("mix/d2.jsonl", 0, "nonen"), ("mix/d2p.jsonl", 0, "nonen")]
    if arm == "abl_en":
        return base + en
    if arm == "abl_nonen":
        return base + nonen
    return base + en + nonen


@app.function(image=nemo_image, volumes={VOL: vol}, gpu="A100-40GB", cpu=8, memory=32768, timeout=6 * 3600,
              retries=modal.Retries(max_retries=2, initial_delay=30.0))
def retrain(run: str, arm: str = "main", seed: int = 1, max_steps: int = 3000, kl_tok: float = 1.0,
            init_from: str = "v10pt@1500", lr: float = 1e-4, warmup: int = 50, save_every: int = 500,
            kl_terms: str = "blank,gnorm,blankspan", check_share: bool = True):
    """One arm of the real-speech retrain. Resumes from exp/<run>/last.ckpt after a preemption (retries).
    Dress rehearsal = the same mix with a small max_steps/save_every (manifests are read lazily: no frac needed)."""
    import json

    import ft_train
    import mix_build

    vol.reload()
    mix = retrain_mix(arm)
    share = mix_build.mix_share(VOL, mix)
    print("MIX", json.dumps(share))
    if check_share and arm in ("main", "stockinit"):
        assert 0.35 <= share["share"] <= 0.40, f"number share {share['share']} outside 35-40% (V2 stop rule)"
    k = dict(RETRAIN_K, save_every=save_every)
    ft_train.train(VOL, ultra_path(), run, mix=mix, max_steps=max_steps, lr=lr, warmup=warmup, seed=seed,
                   init_from="" if arm == "stockinit" else init_from, kl_terms=kl_terms, kl_tok=kl_tok,
                   on_save=vol.commit, **k)
    vol.commit()


@app.function(image=nemo_image, volumes={VOL: vol}, gpu="L4", timeout=4 * 3600, max_containers=6)
def eval_retrain(run: str, step: int = 0, sealed: bool = False):
    """Decode every retrain bed with exp/<run>/step<step>.pt ("v10pt"/"ultra" baselines: their own weights)."""
    import torch

    import ft_eval

    vol.reload()
    m = ft_eval.load_model(VOL, "ultra", ultra_path())
    if run != "ultra":
        m.load_state_dict(torch.load(f"{VOL}/exp/{run}/step{step}.pt", map_location="cpu"), strict=False)
    ft_eval.eval_retrain(VOL, m.cuda().eval(), f"{run}@{step}" if run != "ultra" else "ultra_rt",
                         sealed=sealed)
    vol.commit()


@app.cls(image=nemo_image, volumes={VOL: vol}, gpu="L4", timeout=2 * 3600, max_containers=6, retries=1)
class Vetoer:
    @modal.enter()
    def load(self):
        import veto

        self.v = veto.Veto()

    @modal.method()
    def tag(self, tag: str) -> dict:
        st = self.v.rows(VOL, tag)
        vol.commit()
        return st


@app.function(image=scan_image, volumes={VOL: vol}, timeout=6 * 3600)
def s3_veto(tags: str = ""):
    import glob
    import json
    import os

    vol.reload()
    ts = tags.split(",") if tags else sorted(os.path.basename(f)[:-6] for f in glob.glob(f"{VOL}/d1/rows/*.jsonl"))
    tot = [0, 0]
    for st in Vetoer().tag.map(ts, return_exceptions=True):
        if isinstance(st, Exception):
            print("FAIL", repr(st)[:300])
            continue
        print(json.dumps(st))
        tot[0] += st["spans"]
        tot[1] += st["vetoes"]
    print(f"TOTAL spans {tot[0]} vetoes {tot[1]}")
