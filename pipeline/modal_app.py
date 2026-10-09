"""Modal entry points for the spoken-form fine-tune.

Run from the repo root, e.g.:
    modal run pipeline/modal_app.py::env_check
    modal run --detach pipeline/modal_app.py::killtest
"""

import modal

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
    .env({"HF_HOME": f"{VOL}/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("common", "data_e22", "data_replay", "data_fleurs", "ft_train", "ft_eval")
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
    .add_local_file("carriers.txt", "/root/carriers.txt")
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
def make_tts(name: str, n_rows: int, seed: int = 1, shard: int = 200, multi_frac: float = 0.0):
    """Generate rows, voice them in parallel shards, write train/dev manifests under /vol/tts/<name>."""
    import json
    import os

    import gen

    rows = list(gen.rows(n_rows, seed, multi_frac=multi_frac))
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
