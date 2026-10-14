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
                              "mix_build", "recrop", "score_retrain", "yearflip")
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
    .add_local_file(os.path.join(os.path.dirname(__file__), "carriers_biz.txt"), "/root/carriers_biz.txt")
    .add_local_file(os.path.join(os.path.dirname(__file__), "carriers_mp.txt"), "/root/carriers_mp.txt")
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


@app.function(image=tts_image, volumes={VOL: vol}, timeout=3 * 3600)
def make_tts_mp(name: str = "mp", n_pairs: int = 1500, seed: int = 31, shard: int = 200, and_frac: float = 0.0):
    """R3.31 minimal pairs (gen.minimal_pairs): each filled carrier voiced as "twenty X" and "two thousand X" with the
    same voice. Writes /vol/tts/<name>/{train,dev}.jsonl; run filter_tts, then mp_balance keeps only complete pairs."""
    import json
    import os

    import gen

    rows = list(gen.minimal_pairs(n_pairs, seed, and_frac=and_frac))
    shards = [rows[i:i + shard] for i in range(0, len(rows), shard)]  # even shard size keeps pairs together
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


@app.function(image=tts_image, volumes={VOL: vol}, timeout=600)
def mp_balance(name: str = "mp"):
    """After filter_tts: keep a pair only when BOTH readings passed, so the set stays exactly 50/50 per context."""
    import collections
    import json

    vol.reload()
    for split in ("train", "dev"):
        rows = [json.loads(l) for l in open(f"{VOL}/tts/{name}/{split}_ok.jsonl")]
        n = collections.Counter(r["pair"] for r in rows)
        keep = [r for r in rows if n[r["pair"]] == 2]
        with open(f"{VOL}/tts/{name}/{split}_ok.jsonl", "w") as f:
            for r in keep:
                f.write(json.dumps(r) + "\n")
        print(f"{name}/{split}: {len(rows)} passed filter, {len(keep)} in complete pairs "
              f"({sum(r['duration'] for r in keep) / 3600:.2f} h)")
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
def export_weights(run: str = "ultra", step: int = -1):
    """Decoder + joint tensors (fp32, NeMo names) as safetensors, for swap_weights.py."""
    import os

    import torch
    from safetensors.torch import save_file

    vol.reload()
    if run == "ultra":
        sd = torch.load(ultra_state(), map_location="cpu")
    elif step >= 0:  # step 0 = a soup checkpoint (exp/<soup>/step0.pt)
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


@app.function(image=scan_image, volumes={VOL: vol}, secrets=[hf_secret], cpu=2, memory=8192, timeout=3600,
              retries=modal.Retries(max_retries=2, initial_delay=60.0))
def d1p_fetch_shard(path: str, hours: float) -> dict:
    import fetch

    vol.reload()
    st = fetch.fetch_plain(VOL, path, hours=hours)
    vol.commit()
    return st


@app.function(image=nemo_image, volumes={VOL: vol}, gpu="L4", timeout=3 * 3600)
def d1p_filter(max_cer: float = 0.15):
    """Stock-decode every fetched plain row; keep caption-vs-stock CER <= max_cer -> /vol/d1/plain3.jsonl."""
    import glob
    import json

    import ft_eval
    import label_gpu

    vol.reload()
    rows = [json.loads(l) for f in sorted(glob.glob(f"{VOL}/yodas/en_plain/manifests/*.jsonl")) for l in open(f)]
    m = ft_eval.load_model(VOL, "ultra", ultra_path())
    hyps = ft_eval.decode(m, [r["audio_filepath"] for r in rows], batch_size=32)
    keep = []
    for r, h in zip(rows, hyps):
        if label_gpu.cer(r["text"], h["text"]) <= max_cer:
            keep.append({"audio_filepath": r["audio_filepath"], "duration": r["duration"], "text": r["text"],
                         "video": r["original_audio_id"], "stock": h["text"]})
    with open(f"{VOL}/d1/plain3.jsonl", "w") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"plain rows {len(rows)} ({sum(r['duration'] for r in rows) / 3600:.2f} h) -> kept {len(keep)} "
          f"({sum(r['duration'] for r in keep) / 3600:.2f} h)")
    vol.commit()


@app.function(image=scan_image, volumes={VOL: vol}, timeout=4 * 3600)
def d1p_fetch(hours: float = 0.5):
    """D1p: number-free rows from the 33 labelled English shards (paths from their manifests)."""
    import glob
    import json
    import os

    vol.reload()
    paths = []
    for f in sorted(glob.glob(f"{VOL}/d1/rows/*.jsonl")):  # the 33 labelled shards (rows3 fills while this runs)
        tag = os.path.basename(f)[:-6]
        if tag.endswith("_pct"):
            continue
        r = json.loads(open(f"{VOL}/yodas/en/manifests/{tag}.jsonl").readline())
        paths.append(r["shard"])
    paths = sorted(set(paths))
    tot = 0.0
    for st in d1p_fetch_shard.map(paths, kwargs={"hours": hours}, return_exceptions=True):
        print(st if not isinstance(st, Exception) else f"FAIL {repr(st)[:300]}")
        tot += st.get("h", 0) if isinstance(st, dict) else 0
    print(f"TOTAL {tot:.2f} h from {len(paths)} shards")


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
    def shard(self, man: str, tag: str, keep_audio: bool = True, only_pct: bool = False, rows_dir: str = "d1/rows",
              write_crops: bool = True) -> dict:
        import os

        import label_gpu

        done = f"{VOL}/{rows_dir}/{tag}.jsonl" if not write_crops else f"{VOL}/d1/crops/{tag}.jsonl"
        if os.path.exists(done):
            return {"shard": tag, "skipped": True}
        st = label_gpu.label_shard(VOL, self.sc, man, tag, keep_audio, only_pct, rows_dir=rows_dir,
                                   write_crops=write_crops)
        vol.commit()
        return st


@app.function(image=scan_image, volumes={VOL: vol}, timeout=12 * 3600)
def s3_label(lang: str = "en", limit: int = 0, only: str = "", only_pct: bool = False, rows_dir: str = "d1/rows",
             write_crops: bool = True, labelled_only: bool = False):
    """Relabel every fetched English shard (S3); `only` = comma list of shard tags."""
    import glob
    import json
    import os

    vol.reload()
    mans = sorted(glob.glob(f"{VOL}/yodas/{lang}/manifests/*.jsonl"))
    if only:
        mans = [m for m in mans if os.path.basename(m)[:-6] in only.split(",")]
    if labelled_only:  # retrain-3 relabel: exactly the shards labelled before
        have = {os.path.basename(f)[:-6] for f in glob.glob(f"{VOL}/d1/rows/*.jsonl") if not f.endswith("_pct.jsonl")}
        mans = [m for m in mans if os.path.basename(m)[:-6] in have]
    mans = mans[: limit or None]
    print(f"{len(mans)} shards")
    tot = {"rows": 0, "crops": 0, "crop_h": 0.0, "dropped": 0}
    suffix = "_pct" if only_pct else ""
    jobs = [(m, os.path.basename(m)[:-6] + suffix, write_crops, only_pct, rows_dir, write_crops) for m in mans]
    for st in Labeller().shard.starmap(jobs, return_exceptions=True):
        if isinstance(st, Exception):
            print("FAIL", repr(st)[:400])
            continue
        print(json.dumps(st))
        for k in tot:
            tot[k] += st.get(k, 0)
    print("TOTAL", json.dumps(tot))


@app.cls(image=nemo_image, volumes={VOL: vol}, gpu="L4", timeout=2 * 3600, max_containers=8, retries=1)
class Recropper:
    @modal.enter()
    def load(self):
        import os

        import recrop

        os.environ["HF_HOME"] = "/tmp/hf"
        self.sc = recrop.W2VScorer()

    @modal.method()
    def shard(self, tag: str, rows_dir: str = "d1/rows3") -> dict:
        import recrop

        st = recrop.recrop_shard(VOL, self.sc, tag, rows_dir=rows_dir)
        vol.commit()
        return st


@app.function(image=scan_image, volumes={VOL: vol}, timeout=6 * 3600)
def s3_recrop(only: str = "", rows_dir: str = "d1/rows3"):
    """Quiet-gap re-crop of every labelled English shard -> /vol/d1/crops2 (retrain-3)."""
    import glob
    import json
    import os

    vol.reload()
    tags = sorted(os.path.basename(f)[:-6] for f in glob.glob(f"{VOL}/{rows_dir}/*.jsonl") if not f.endswith("_pct.jsonl"))
    if only:
        tags = [t for t in tags if t in only.split(",")]
    tot = {}
    for st in Recropper().shard.map(tags, kwargs={"rows_dir": rows_dir}, return_exceptions=True):
        if isinstance(st, Exception):
            print("FAIL", repr(st)[:400])
            continue
        print(json.dumps(st))
        for k, v in st.items():
            if isinstance(v, (int, float)):
                tot[k] = round(tot.get(k, 0) + v, 3)
    print("TOTAL", json.dumps(tot))


@app.cls(image=nemo_image, volumes={VOL: vol}, gpu="L4", timeout=2 * 3600, max_containers=8, retries=1)
class CropDecoder:
    @modal.enter()
    def load(self):
        import torch

        import ft_eval

        self.m = ft_eval.load_model(VOL, "ultra", ultra_path())
        self.stock = {k: v.clone() for k, v in self.m.state_dict().items() if not k.startswith(("encoder.", "preprocessor."))}
        self.v10 = torch.load(f"{VOL}/exp/v10pt/step1500.pt", map_location="cpu")

    @modal.method()
    def shard(self, tag: str, crops_dir: str = "d1/crops3", out_dir: str = "d1/dec3") -> dict:
        """Stock + v10pt greedy decodes of one shard's crops (filters D-1b/D-2/D-4, restore_fillers, F2 report)."""
        import json
        import os

        import ft_eval

        vol.reload()
        rows = [json.loads(line) for line in open(f"{VOL}/{crops_dir}/{tag}.jsonl")]
        paths = [r["audio_filepath"] for r in rows]
        res = {p: {} for p in paths}
        for name, st in (("stock", self.stock), ("v10", self.v10)):
            self.m.load_state_dict(st, strict=False)
            for p, h in zip(paths, ft_eval.decode(self.m, paths, batch_size=32) if paths else []):
                res[p][name] = h["text"]
        os.makedirs(f"{VOL}/{out_dir}", exist_ok=True)
        json.dump(res, open(f"{VOL}/{out_dir}/{tag}.json", "w"))
        vol.commit()
        return {"shard": tag, "crops": len(paths)}


    @modal.method()
    def paths(self, paths: list, which: str = "v10") -> dict:
        import ft_eval

        self.m.load_state_dict(self.v10 if which == "v10" else self.stock, strict=False)
        return {p: h["text"] for p, h in zip(paths, ft_eval.decode(self.m, paths, batch_size=32))}


@app.function(image=scan_image, volumes={VOL: vol}, timeout=4 * 3600)
def decode_manifest(manifest: str = "d1/plain3.jsonl", out: str = "d1/plain3_v10.json", which: str = "v10"):
    """Parallel (8 x L4) decode of one manifest -> {wav: text} (D1p filler restoration, R3.17 F1)."""
    import json

    vol.reload()
    import glob

    paths = [json.loads(line)["audio_filepath"] for f in sorted(glob.glob(f"{VOL}/{manifest}")) for line in open(f)]
    parts = [paths[i::16] for i in range(16)]
    res = {}
    for d in CropDecoder().paths.map(parts, kwargs={"which": which}):
        res.update(d)
    json.dump(res, open(f"{VOL}/{out}", "w"))
    vol.commit()
    print(f"decoded {len(res)} of {len(paths)}")


@app.function(image=scan_image, volumes={VOL: vol}, timeout=4 * 3600)
def s3_decode_crops(only: str = "", crops_dir: str = "d1/crops3", out_dir: str = "d1/dec3"):
    """Parallel (8 x L4) stock + v10pt decode of the recropped shards; skips shards already decoded."""
    import glob
    import os

    vol.reload()
    tags = sorted(os.path.basename(f)[:-6] for f in glob.glob(f"{VOL}/{crops_dir}/*.jsonl"))
    if only:
        tags = [t for t in tags if t in only.split(",")]
    tags = [t for t in tags if not os.path.exists(f"{VOL}/{out_dir}/{t}.json")]
    n = 0
    for st in CropDecoder().shard.map(tags, kwargs={"crops_dir": crops_dir, "out_dir": out_dir}, return_exceptions=True):
        print(st if not isinstance(st, Exception) else f"FAIL {repr(st)[:300]}")
        n += st.get("crops", 0) if isinstance(st, dict) else 0
    print(f"TOTAL {n} crops from {len(tags)} shards")


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


@app.function(image=nemo_image, volumes={VOL: vol}, secrets=[hf_secret], gpu="L4", cpu=8, memory=32768,
              timeout=6 * 3600)
def build_beds3(which: str = "vox,fleurs,e21"):
    """Retrain-3 beds: vox LID+dedupe, FLEURS full split, e21_dev (app windows; kept where stock writes a digit or
    v10pt a number word) + the blend reproduction check with rt2_main_s2 (R3.12.5)."""
    import json
    import os
    import re

    import torch

    os.environ["HF_HOME"] = "/tmp/hf"
    import beds
    import ft_eval
    import score_retrain

    vol.reload()
    if "vox" in which:
        print("vox3", json.dumps(beds.vox3(VOL)))
        vol.commit()
    if "fleurs" in which:
        print("fleurs3", json.dumps(beds.fleurs3(VOL)))
        vol.commit()
    if "e21" in which:
        rows = beds.e21_dev_cut(VOL)
        vol.commit()
        m = ft_eval.load_model(VOL, "ultra", ultra_path())
        stock = {k: v.clone() for k, v in m.state_dict().items() if not k.startswith(("encoder.", "preprocessor."))}
        paths = [r["audio_filepath"] for r in rows]
        hyp = {}
        for name, sd in (("stock", stock), ("v10", f"{VOL}/exp/v10pt/step1500.pt"),
                         ("rt2", f"{VOL}/exp/rt2_main_s2/step3000.pt")):
            m.load_state_dict(sd if isinstance(sd, dict) else torch.load(sd, map_location="cpu"), strict=False)
            hyp[name] = [h["text"] for h in ft_eval.decode(m, paths, batch_size=32)]
        num_w = re.compile(r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|"
                           r"forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|percent)\b", re.I)
        keep = []
        for i, r in enumerate(rows):
            r = {**r, "stock": hyp["stock"][i], "v10": hyp["v10"][i], "rt2": hyp["rt2"][i]}
            if re.search(r"\d", r["stock"]) or num_w.search(r["v10"]):
                keep.append(r)
            else:
                os.remove(r["audio_filepath"])
        beds._write(f"{VOL}/beds/e21_dev/all.jsonl", keep)
        bl = {n: [(r["call"], r["start"], m_.group(0)) for r in keep for m_ in score_retrain.BLEND.finditer(r[n])]
              for n in ("stock", "v10", "rt2")}
        print("e21_dev windows", len(rows), "kept", len(keep), f"{sum(r['duration'] for r in keep) / 3600:.2f} h")
        print("BLEND hits", json.dumps({n: len(v) for n, v in bl.items()}), "rt2:", json.dumps(bl["rt2"][:20]))
        print("4366893 rt2 blends:", sum(c == "4366893" for c, _, _ in bl["rt2"]))
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


@app.function(image=mix_image, volumes={VOL: vol}, cpu=8, memory=32768, timeout=8 * 3600)
def build_mix3(seed: int = 1, d1_h: float = 20.0, d10_h: float = 6.0):
    import json

    import mix_build

    vol.reload()
    print(json.dumps(mix_build.build3(VOL, seed=seed, d1_h=d1_h, d10_h=d10_h), indent=1))
    vol.commit()


RETRAIN_K = dict(batch_duration=400, spec_augment=False, kl_weight=10.0, kl_replay_only=True, kl_punct=30.0,
                 kl_punct_rows="replay", save_every=500)


def retrain_mix(arm: str, tts_h: float = 6.0) -> list:
    """The V2 SPEC mix per arm. main/stockinit: everything; abl_en: v10pt mix + English real only;
    abl_nonen: v10pt mix + non-English real only. Hours <= 0 = all rows of that manifest."""
    base = [("tts/v10pt/train_ok.jsonl", tts_h, "free"), ("tts/d9sup/train_ok.jsonl", 0, "free"),
            ("replay/lspc_train.jsonl", 35.0, "anchor"), ("replay/lspc_train_numwords.jsonl", 5.0, "anchor"),
            ("replay/fleurs_train_multi.jsonl", 3.0, "nonen"), ("replay/icsi_train.jsonl", 0, "anchor")]
    en = [("mix/d1.jsonl", 0, "span"), ("mix/d1_up.jsonl", 0, "span"), ("mix/d1p.jsonl", 0, "anchor"),
          ("mix/d10n.jsonl", 0, "span"),
          ("mix/d10p.jsonl", 0, "anchor")]
    nonen = [("mix/d2.jsonl", 0, "nonen"), ("mix/d2p.jsonl", 0, "nonen")]
    if arm == "abl_en":
        return base + en
    if arm == "abl_nonen":
        return base + nonen
    return base + en + nonen


def retrain3_mix() -> list:
    """Retrain-3 mix with the R3.17 F6 tag table. Hours <= 0 = all rows of that manifest."""
    return [("tts/v10pt/train_ok.jsonl", 0, "free"), ("tts/d9sup/train_ok.jsonl", 0, "free"),
            ("tts/biz/train_ok.jsonl", 0, "free"),
            ("replay/lspc_train.jsonl", 35.0, "anchor_num"), ("replay/lspc_train_numwords.jsonl", 5.0, "anchor_num"),
            ("replay/icsi_train.jsonl", 0, "anchor_num"), ("mix3/d1p.jsonl", 0, "anchor"), ("mix3/d10p.jsonl", 0, "anchor"),
            ("mix3/d1.jsonl", 0, "span"), ("mix3/d10n.jsonl", 0, "span"),
            ("mix3/fleurs_train_multi.jsonl", 3.0, "nonen"), ("mix3/d2.jsonl", 0, "nonen"), ("mix3/d2p.jsonl", 0, "nonen")]


def retrain3_mix_mp() -> list:
    """R3.31 treatment mix: retrain3_mix with mix3b (2000-2009 cap undone, year rows capped per video, repeated texts
    dropped), tts/biz without k_twenty_pause, plus the tts/mp minimal pairs."""
    sub = {"mix3/d1.jsonl": "mix3b/d1.jsonl", "mix3/d10n.jsonl": "mix3b/d10n.jsonl",
           "tts/biz/train_ok.jsonl": "tts/biz/train_ok_np.jsonl"}
    return [(sub.get(p, p), h, t) for p, h, t in retrain3_mix()] + [("tts/mp/train_ok.jsonl", 0, "free")]


def retrain3_mix_v2(mp: str = "mp2", mp_repeat: int = 3, verbatim: bool = False) -> list:
    """R3.31 v2 treatment: mix3b with tts/<mp> repeated mp_repeat x (external review #4: the first warm-start saw each
    pair ~0.4 x); verbatim=True also swaps d1p/d10p for their verbatim-filtered versions."""
    sub = {"mix3/d1.jsonl": "mix3b/d1.jsonl", "mix3/d10n.jsonl": "mix3b/d10n.jsonl",
           "tts/biz/train_ok.jsonl": "tts/biz/train_ok_np.jsonl"}
    if verbatim:
        sub.update({"mix3/d1p.jsonl": "mix3b/d1p_v.jsonl", "mix3/d10p.jsonl": "mix3b/d10p_v.jsonl"})
    return [(sub.get(p, p), h, t) for p, h, t in retrain3_mix()] + [(f"tts/{mp}/train_ok_x{mp_repeat}.jsonl", 0, "free")]


@app.function(image=mix_image, volumes={VOL: vol}, cpu=8, memory=32768, timeout=3600)
def build_mix3c(mp: str = "mp2", mp_repeat: int = 3):
    import json

    import mix_build

    vol.reload()
    print(json.dumps(mix_build.build3c(VOL, mp=mp, mp_repeat=mp_repeat), indent=1))
    for v in (False, True):
        print("SHARE v2" + (" verbatim" if v else ""), json.dumps(mix_build.mix_share(VOL, retrain3_mix_v2(mp, mp_repeat, v))))
    vol.commit()


@app.function(image=mix_image, volumes={VOL: vol}, cpu=8, memory=32768, timeout=3600)
def build_mix3b(per_video: int = 12):
    import json

    import mix_build

    vol.reload()
    print(json.dumps(mix_build.build3b(VOL, per_video=per_video), indent=1))
    print("SHARE mix3", json.dumps(mix_build.mix_share(VOL, retrain3_mix())))
    print("SHARE mix3b+mp", json.dumps(mix_build.mix_share(VOL, retrain3_mix_mp())))
    vol.commit()


@app.function(image=nemo_image, volumes={VOL: vol}, cpu=16, memory=32768, timeout=2 * 3600)
def audit_mix3(fix: bool = True, tol: float = 0.1):
    """Every row of every retrain-3 mix part: the wav exists, is 16 kHz, and its real duration matches the manifest
    (P3 found FLEURS ids shared across speakers: D2 wavs named by id were overwritten). fix=True rewrites the mix3/
    parts without the bad rows (the originals are kept as <part>.pre_audit.jsonl)."""
    import json
    import os
    from concurrent.futures import ThreadPoolExecutor

    import soundfile as sf

    vol.reload()

    def chk(r):
        try:
            i = sf.info(r["audio_filepath"])
        except Exception:
            return "missing"
        if i.samplerate != 16000:
            return "sr"
        return "dur" if abs(i.frames / i.samplerate - r["duration"]) > tol else ""

    out = {}
    for path, _h, tag in retrain3_mix():
        rows = [json.loads(line) for line in open(f"{VOL}/{path}")]
        with ThreadPoolExecutor(64) as ex:
            res = list(ex.map(chk, rows))
        bad = {k: sum(x == k for x in res) for k in ("missing", "sr", "dur")}
        dup = len(rows) - len({r["audio_filepath"] for r in rows})
        out[path] = {"rows": len(rows), **bad, "dup_paths": dup,
                     "bad_examples": [(r["audio_filepath"], r["duration"], x) for r, x in zip(rows, res) if x][:5]}
        if fix and path.startswith("mix3/") and any(res):
            os.replace(f"{VOL}/{path}", f"{VOL}/{path[:-6]}.pre_audit.jsonl")
            with open(f"{VOL}/{path}", "w") as f:
                for r, x in zip(rows, res):
                    if not x:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            out[path]["rewritten"] = True
        print(path, json.dumps(out[path]))
    json.dump(out, open(f"{VOL}/mix3/audit.json", "w"), indent=1)
    vol.commit()


@app.function(image=nemo_image, volumes={VOL: vol}, cpu=16, memory=16384, timeout=3600)
def clean_manifest(src: str = "replay/fleurs_train_multi.jsonl", dst: str = "mix3/fleurs_train_multi.jsonl",
                   tol: float = 0.1):
    """Copy of a shared manifest without rows whose wav is missing / not 16 kHz / a different length (FLEURS id
    collisions), and with one row per wav path (the one matching the file)."""
    import json
    from concurrent.futures import ThreadPoolExecutor

    import soundfile as sf

    vol.reload()
    rows = [json.loads(line) for line in open(f"{VOL}/{src}")]

    def ok(r):
        try:
            i = sf.info(r["audio_filepath"])
        except Exception:
            return False
        return i.samplerate == 16000 and abs(i.frames / i.samplerate - r["duration"]) <= tol

    with ThreadPoolExecutor(64) as ex:
        good = list(ex.map(ok, rows))
    seen, keep = set(), []
    for r, g in zip(rows, good):
        if g and r["audio_filepath"] not in seen:
            seen.add(r["audio_filepath"])
            keep.append(r)
    with open(f"{VOL}/{dst}", "w") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    vol.commit()
    print(f"{src}: {len(rows)} -> {len(keep)} rows -> {dst}")


@app.function(image=nemo_image, volumes={VOL: vol}, cpu=8, memory=32768, timeout=3600)
def patch_mix3():
    """Retrain-3 review fixes to the built mix: year-ratio cap (case-insensitive) + label-noise drops on D1/D10n, and
    the Greek/thin-language "\u2047" D2 rows restored (mix_build.patch3 / restore_unk_nonen)."""
    import glob
    import json

    import mix_build

    vol.reload()
    dec = {}
    for f in glob.glob(f"{VOL}/d1/dec3/*.json"):
        dec.update(json.load(open(f)))
    out = {"patch3": mix_build.patch3(VOL, dec), "unk": mix_build.restore_unk_nonen(VOL)}
    out["share"] = mix_build.mix_share(VOL, retrain3_mix())
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{VOL}/mix3/patch3.json", "w"), indent=1)
    vol.commit()


@app.function(image=nemo_image, volumes={VOL: vol}, cpu=4, memory=16384, timeout=1800)
def soup3(run: str, steps: str = "3500,3750,4000", alphas: str = "1.0,0.85,0.7,0.5", base: str = "v10pt@1500",
          name: str = ""):
    """R3.16 #5 + R3.12.4: average `steps` of `run` (checkpoint soup), then blend with `base` (v10pt, NOT stock):
    theta = base + a * (soup - base). The written-form output rows of the joint's last layer keep the soup's values
    (a = 1, N2), so blending cannot undo the trained digit suppression. Writes exp/<name>_a<a>/step0.pt each."""
    import torch

    import ft_train

    vol.reload()
    sds = [torch.load(f"{VOL}/exp/{run}/step{s_}.pt", map_location="cpu") for s_ in steps.split(",")]
    soup = {k: sum(sd[k].float() for sd in sds) / len(sds) for k in sds[0]}
    b_run, _, b_step = base.partition("@")
    b = torch.load(f"{VOL}/exp/{b_run}/step{b_step}.pt", map_location="cpu")
    import nemo.collections.asr as nemo_asr

    m = nemo_asr.models.ASRModel.restore_from(ultra_path(), map_location="cpu")
    last = f"joint.joint_net.{len(m.joint.joint_net) - 1}."
    wf = torch.tensor(ft_train.wf_piece_ids(m.tokenizer.tokenizer))
    name = name or f"{run}_soup"
    for a in (float(x) for x in alphas.split(",")):
        out = {}
        for k, v in soup.items():
            if k not in b:
                out[k] = v
                continue
            w = b[k].float() + a * (v - b[k].float())
            if k.startswith(last) and w.shape[0] > int(wf.max()):
                w[wf] = v[wf]
            out[k] = w.to(sds[0][k].dtype)
        import os

        os.makedirs(f"{VOL}/exp/{name}_a{a}", exist_ok=True)
        torch.save(out, f"{VOL}/exp/{name}_a{a}/step0.pt")
        print("wrote", f"exp/{name}_a{a}/step0.pt")
    vol.commit()


# Screening arms (R3.12.4 + R3.14: S-D dropped). Each is a full 4000-step cosine run evaluated at step 1000; the winner
# continues as final seed 1.
# c = 1 (owner 2026-10-13, reviews: c = 0.3 pulled back < 3% of the drift by step 1000, so S-A ~ S-C); S-C is the c = 0
# contrast. Duration KL on KL rows on (owner 2026-10-13); weight 1: at 10 its start-up value (0.13-0.32 nats per cell,
# v10pt drifted from stock) outweighed RNNT ~10:1 in E2.
R3_ARMS = {"S-A": dict(wf_weight=1.0, l2sp_c=1.0, lr=1e-4), "S-B": dict(wf_weight=3.0, l2sp_c=1.0, lr=1e-4),
           "S-C": dict(wf_weight=1.0, l2sp_c=0.0, lr=1e-4), "S-E": dict(wf_weight=1.0, l2sp_c=1.0, lr=5e-5)}


@app.function(image=nemo_image, volumes={VOL: vol}, gpu="A100-40GB", cpu=8, memory=32768, timeout=24 * 3600,
              retries=modal.Retries(max_retries=2, initial_delay=30.0))
def retrain3(run: str, arm: str = "S-A", seed: int = 1, max_steps: int = 4000, save_every: int = 250,
             en_weight: float = 1.0, check_share: bool = True, warmup: int = 50, gradshare: bool = False,
             dur_weight: float = 1.0, init_from: str = "v10pt@1500", lr: float = 0.0, mix: str = "mix3",
             anchor_init: bool = False):
    """One retrain-3 run (R3.13): r3 losses, decoupled L2-SP to v10pt@1500, 200 s x 2 accumulated batches.
    Rehearsal (E2) = a small max_steps. Resumes from exp/<run>/last.ckpt after a preemption."""
    import json
    import os

    if gradshare:
        os.environ["R3_GRADSHARE"] = "1"
    import ft_train
    import mix_build

    vol.reload()
    cfg = dict(R3_ARMS[arm])
    if lr:  # R3.31 warm start: a lower peak LR than the arm's
        cfg["lr"] = lr
    mix = {"mp": retrain3_mix_mp, "v2": lambda: retrain3_mix_v2(), "v2v": lambda: retrain3_mix_v2(verbatim=True)}.get(
        mix, retrain3_mix)()
    # external review #4: a warm start anchors L2-SP to its own init, not v10pt (the prior being trained away)
    i_run, _, i_step = init_from.partition("@")
    anchor = f"{VOL}/exp/{i_run}/step{i_step}.pt" if anchor_init else f"{VOL}/exp/v10pt/step1500.pt"
    print("L2SP ANCHOR", anchor)
    share = mix_build.mix_share(VOL, mix)
    import shutil

    print("MIX", json.dumps(share), "ARM", arm, json.dumps(cfg), "DISK /tmp free GB",
          round(shutil.disk_usage("/tmp").free / 1e9, 1))
    if check_share:
        assert 0.35 <= share["share"] <= 0.40, f"number share {share['share']} outside 35-40% (R3.17 F4)"
    ft_train.train(VOL, ultra_path(), run, mix=mix, max_steps=max_steps, lr=cfg["lr"], warmup=warmup, seed=seed,
                   init_from=init_from, batch_duration=200, accumulate=2, spec_augment=False, kl_weight=10.0,
                   kl_punct=30.0, kl_tok=1.0, save_every=save_every, on_save=vol.commit, stage_dir="/tmp/stage",
                   r3=True, wf_weight=cfg["wf_weight"], en_weight=en_weight, l2sp_c=cfg["l2sp_c"],
                   l2sp_anchor=anchor, dur_weight=dur_weight)
    vol.commit()


@app.function(image=nemo_image, volumes={VOL: vol}, gpu="A100-40GB", cpu=8, memory=32768, timeout=6 * 3600,
              retries=modal.Retries(max_retries=2, initial_delay=30.0))
def retrain(run: str, arm: str = "main", seed: int = 1, max_steps: int = 3000, kl_tok: float = 1.0,
            init_from: str = "v10pt@1500", lr: float = 1e-4, warmup: int = 50, save_every: int = 500,
            kl_terms: str = "blank,gnorm,blankspan", check_share: bool = True, kl_eq_check: bool = False):
    """One arm of the real-speech retrain. Resumes from exp/<run>/last.ckpt after a preemption (retries).
    Dress rehearsal = the same mix with a small max_steps/save_every (manifests are read lazily: no frac needed)."""
    import json
    import os

    if kl_eq_check:
        os.environ["KL_EQ_CHECK"] = "1"
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
                   on_save=vol.commit, stage_dir="/tmp/stage", **k)
    vol.commit()


@app.function(image=nemo_image, volumes={VOL: vol}, gpu="L4", timeout=3600)
def eval_mp_dev(runs: str, name: str = "mp"):
    """R3.31 diagnostic (not a gate; TTS beds measure our own templates): each checkpoint on the held-out minimal pairs
    (held-out carriers + voices) -> hyps/<run>/mp_dev.jsonl. Reading accuracy per side comes from score_mp.py."""
    import torch

    import ft_eval

    vol.reload()
    m = ft_eval.load_model(VOL, "ultra", ultra_path())
    stock = {k: v.clone() for k, v in m.state_dict().items()}
    for spec in runs.split(","):
        r, _, step = spec.partition("@")
        m.load_state_dict(stock if r == "ultra" else torch.load(f"{VOL}/exp/{r}/step{step}.pt", map_location="cpu"),
                          strict=False)
        ft_eval.transcribe_loaded(VOL, m, spec, f"tts/{name}/dev_ok.jsonl", "mp_dev")
    vol.commit()


@app.function(image=nemo_image, volumes={VOL: vol}, gpu="L4", timeout=4 * 3600, max_containers=6)
def eval_retrain(run: str, step: int = 0, sealed: bool = False, r3: bool = False):
    """Decode every retrain bed with exp/<run>/step<step>.pt ("v10pt"/"ultra" baselines: their own weights)."""
    import torch

    import ft_eval

    vol.reload()
    m = ft_eval.load_model(VOL, "ultra", ultra_path())
    if run != "ultra":
        m.load_state_dict(torch.load(f"{VOL}/exp/{run}/step{step}.pt", map_location="cpu"), strict=False)
    ft_eval.eval_retrain(VOL, m.cuda().eval(), f"{run}@{step}" if run != "ultra" else "ultra_rt",
                         sealed=sealed, r3=r3)
    vol.commit()


@app.function(image=nemo_image, volumes={VOL: vol}, gpu="L4", cpu=8, memory=65536, timeout=4 * 3600)
def cache_beds3(sealed: bool = False):
    """R3.15 E-a: encode every retrain-3 bed once (frozen encoder) -> /vol/enc/."""
    import json

    import ft_eval

    vol.reload()
    m = ft_eval.load_model(VOL, "ultra", ultra_path())
    sets = ft_eval.RETRAIN3_SETS + ([(a, b, 0) for a, b in ft_eval.SEALED3_SETS] if sealed else [])
    print(json.dumps(ft_eval.cache_beds(VOL, m, sets)))
    vol.commit()


@app.function(image=nemo_image, volumes={VOL: vol}, gpu="L4", cpu=8, memory=65536, timeout=6 * 3600)
def eval_multi(runs: str, sealed: bool = False, tag: str = "", skip: str = ""):
    """Decode every cached retrain-3 bed with each checkpoint in `runs` ("v10pt@1500,ultra,S-A@1000,...";
    "<path>.pt" also works). One encoder pass serves all of them. Hyps go to hyps/<run>@<step><tag>."""
    import os

    import torch

    import ft_eval

    vol.reload()
    m = ft_eval.load_model(VOL, "ultra", ultra_path())
    stock = {k: v.clone() for k, v in m.state_dict().items() if not k.startswith(("encoder.", "preprocessor."))}
    sets = ft_eval.RETRAIN3_SETS + ([(a, b, 0) for a, b in ft_eval.SEALED3_SETS] if sealed else [])
    sets = [x for x in sets if x[1] not in skip.split(",")]  # screening: skip=e21_dev (report-only, 25 h of audio)
    for run in runs.split(","):
        if run == "ultra":
            sd, name = stock, "ultra_rt"
        elif run.endswith(".pt"):
            sd, name = torch.load(f"{VOL}/{run}", map_location="cpu"), os.path.basename(run)[:-3]
        else:
            r, _, step = run.partition("@")
            sd, name = torch.load(f"{VOL}/exp/{r}/step{step}.pt", map_location="cpu"), run
        m.load_state_dict(stock, strict=False)
        m.load_state_dict(sd, strict=False)
        ft_eval.eval_cached(VOL, m, name + tag, sets)
        vol.commit()


@app.cls(image=nemo_image, volumes={VOL: vol}, gpu="L4", timeout=2 * 3600, max_containers=6, retries=1)
class Vetoer:
    @modal.enter()
    def load(self):
        import veto

        self.v = veto.Veto()

    @modal.method()
    def tag(self, tag: str, rows_dir: str = "d1/rows", out_dir: str = "d1/veto") -> dict:
        st = self.v.rows(VOL, tag, rows_dir=rows_dir, out_dir=out_dir)
        vol.commit()
        return st


@app.function(image=scan_image, volumes={VOL: vol}, timeout=6 * 3600)
def s3_veto(tags: str = "", rows_dir: str = "d1/rows", out_dir: str = "d1/veto"):
    import glob
    import json
    import os

    vol.reload()
    ts = tags.split(",") if tags else sorted(os.path.basename(f)[:-6] for f in glob.glob(f"{VOL}/{rows_dir}/*.jsonl"))
    tot = [0, 0]
    for st in Vetoer().tag.map(ts, kwargs={"rows_dir": rows_dir, "out_dir": out_dir}, return_exceptions=True):
        if isinstance(st, Exception):
            print("FAIL", repr(st)[:300])
            continue
        print(json.dumps(st))
        tot[0] += st["spans"]
        tot[1] += st["vetoes"]
    print(f"TOTAL spans {tot[0]} vetoes {tot[1]}")


@app.function(image=nemo_image, volumes={VOL: vol}, gpu="L4", timeout=3 * 3600)
def decode_mix(run: str = "v10pt", step: int = 1500, parts: str = "mix/d1.jsonl,mix/d1p.jsonl",
               out: str = "mix/v10_decode.json"):
    """Greedy decodes of mix manifests (filler restoration: what v10pt hears where the captions dropped fillers)."""
    import json

    import torch

    import ft_eval

    vol.reload()
    m = ft_eval.load_model(VOL, "ultra", ultra_path())
    m.load_state_dict(torch.load(f"{VOL}/exp/{run}/step{step}.pt", map_location="cpu"), strict=False)
    m = m.cuda().eval()
    res = {}
    for p in parts.split(","):
        rows = [json.loads(line) for line in open(f"{VOL}/{p}")]
        hyps = ft_eval.decode(m, [r["audio_filepath"] for r in rows], batch_size=32)
        for r, h in zip(rows, hyps):
            res[r["audio_filepath"]] = h["text"]
        print(p, len(rows))
    json.dump(res, open(f"{VOL}/{out}", "w"))
    vol.commit()
