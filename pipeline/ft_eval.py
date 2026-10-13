"""Decoding for evaluation: Ultra, or Ultra with a trained decoder+joint."""

import json
import os

import torch

from common import WRITTEN_FORM


def load_model(vol: str, run: str, ultra: str):
    import nemo.collections.asr as nemo_asr

    m = nemo_asr.models.ASRModel.restore_from(ultra, map_location="cpu")
    if run != "ultra":
        state = torch.load(f"{vol}/exp/{run}/decoder_joint.pt", map_location="cpu")
        missing, unexpected = m.load_state_dict(state, strict=False)
        assert not unexpected, unexpected
        assert all(k.startswith("encoder.") or k.startswith("preprocessor.") for k in missing)
    return m.cuda().eval()


def decode(m, paths, batch_size=32):
    hyps = m.transcribe(paths, batch_size=batch_size, return_hypotheses=True, verbose=False)
    if isinstance(hyps, tuple):
        hyps = hyps[0]
    out = []
    for h in hyps:
        ids = h.y_sequence.tolist() if torch.is_tensor(h.y_sequence) else list(h.y_sequence)
        out.append({"text": h.text, "ids": ids})
    return out


def transcribe(vol: str, run: str, manifest: str, out_name: str, ultra: str) -> None:
    transcribe_loaded(vol, load_model(vol, run, ultra), run, manifest, out_name)


def transcribe_loaded(vol: str, m, run: str, manifest: str, out_name: str, limit: int = 0) -> None:
    if not os.path.exists(f"{vol}/{manifest}"):
        print(f"skip {manifest}: not prepared")
        return
    rows = [json.loads(l) for l in open(f"{vol}/{manifest}")]
    if limit:
        rows = rows[:limit]
    hyps = decode(m, [r["audio_filepath"] for r in rows])
    out_dir = f"{vol}/hyps/{run}"
    os.makedirs(out_dir, exist_ok=True)
    n_digit = 0
    with open(f"{out_dir}/{out_name}.jsonl", "w") as f:
        for r, h in zip(rows, hyps):
            n_digit += bool(WRITTEN_FORM.search(h["text"]))
            f.write(json.dumps({"id": r.get("id", os.path.basename(r["audio_filepath"])), "ref": r["text"],
                                "hyp": h["text"], "ids": h["ids"]}) + "\n")
    print(f"{run} {manifest}: {len(rows)} decoded, {n_digit} with written-form characters")


def literal_words(template: str) -> list[str]:
    """The carrier's own words (slots removed), normalized."""
    import re

    text = re.sub(r"\{\w+\}", " ", template.replace(" || ", " "))
    return re.sub(r"[^\w' ]", " ", text.lower()).split()


def filter_tts(vol: str, m, name: str) -> None:
    """Drop clips whose carrier words the stock model does not hear: the synthesizer misread the
    text (a respelling, a name, a filler), so the label would not match the audio."""
    import re

    for split in ("train", "dev"):
        rows = [json.loads(l) for l in open(f"{vol}/tts/{name}/{split}.jsonl")]
        hyps = decode(m, [r["audio_filepath"] for r in rows])
        keep, dropped = [], []
        for r, h in zip(rows, hyps):
            heard = set(re.sub(r"[^\w' ]", " ", h["text"].lower()).split())
            words = literal_words(r["template"])
            ok = not words or sum(w in heard for w in words) / len(words) >= 0.9
            (keep if ok else dropped).append(r)
            if not ok and len(dropped) <= 15:
                print(f"drop {split}: {r['text'][:90]!r} -> {h['text'][:90]!r}")
        long = sum(r["duration"] > 20 for r in keep)
        with open(f"{vol}/tts/{name}/{split}_ok.jsonl", "w") as f:
            for r in keep:
                f.write(json.dumps(r) + "\n")
        print(f"{name}/{split}: kept {len(keep)} ({sum(r['duration'] for r in keep) / 3600:.2f} h), "
              f"dropped {len(dropped)}, over 20 s: {long}")


def e22_other(vol: str) -> str:
    """E22 utterances without numbers: an out-of-domain forgetting check (conversational)."""
    path = f"{vol}/e22/other.jsonl"
    if not os.path.exists(path):
        num = {json.loads(l)["id"] for l in open(f"{vol}/e22/numbers.jsonl")}
        with open(path, "w") as f:
            for l in open(f"{vol}/e22/all.jsonl"):
                if json.loads(l)["id"] not in num:
                    f.write(l)
    return "e22/other.jsonl"


def eval_sets(vol: str, m, run: str, tts: str, suffix: str = "") -> None:
    transcribe_loaded(vol, m, run, "e22/numbers.jsonl", "e22_numbers" + suffix)
    transcribe_loaded(vol, m, run, e22_other(vol), "e22_other" + suffix)
    transcribe_loaded(vol, m, run, f"tts/{tts}/dev_ok.jsonl", "tts_dev" + suffix)
    transcribe_loaded(vol, m, run, "replay/lspc_dev.jsonl", "lspc_dev" + suffix, limit=600)


# Real-speech retrain (2026-10-12): every bed, per checkpoint. Sealed beds are decoded with the rest but their hyps sit
# under hyps/<run>/sealed/ and are scored once, at the end, for the chosen model only.
RETRAIN_SETS = [
    ("e22/numbers.jsonl", "e22_numbers", 0), ("e22/other.jsonl", "e22_other", 0),
    ("replay/lspc_dev.jsonl", "lspc_dev", 600), ("mix/d1_dev.jsonl", "d1_dev", 0),
    ("beds/vox/dev.jsonl", "vox_dev", 0), ("beds/fleurs/wer.jsonl", "fleurs_wer", 0),
    ("beds/fleurs/num.jsonl", "fleurs_num", 0), ("beds/noise/all.jsonl", "noise", 0),
    ("beds/adv/all.jsonl", "adv", 0), ("beds/fast/all.jsonl", "fast", 0), ("tts/d9sup/dev_ok.jsonl", "d9_dev", 0),
]
SEALED_SETS = [("beds/vox/sealed.jsonl", "sealed/vox"), ("beds/e21x/sealed.jsonl", "sealed/e21x")]

# Retrain-3 beds (R3.12.5/R3.15/R3.17): vox LID+dedupe, FLEURS one-wav-per-row (screen = 100/lang), mix3 d1_dev,
# e21_dev (app-geometry windows; STOP checks only, never selection), tts/biz dev. FLEURS full split = final gate only.
RETRAIN3_SETS = [
    ("e22/numbers.jsonl", "e22_numbers", 0), ("e22/other.jsonl", "e22_other", 0),
    ("replay/lspc_dev.jsonl", "lspc_dev", 600), ("mix3/d1_dev.jsonl", "d1_dev", 0),
    ("beds/vox/dev3.jsonl", "vox_dev", 0), ("beds/fleurs3/wer.jsonl", "fleurs_wer", 0),
    ("beds/fleurs3/num.jsonl", "fleurs_num", 0), ("beds/noise/all.jsonl", "noise", 0),
    ("beds/adv/all.jsonl", "adv", 0), ("beds/fast/all.jsonl", "fast", 0), ("tts/d9sup/dev_ok.jsonl", "d9_dev", 0),
    ("tts/biz/dev_ok.jsonl", "biz_dev", 0), ("beds/e21_dev/all.jsonl", "e21_dev", 0),
]
SEALED3_SETS = [("beds/vox/sealed3.jsonl", "sealed/vox"), ("beds/e21x/sealed.jsonl", "sealed/e21x"),
                ("beds/fleurs3/wer_full.jsonl", "sealed/fleurs_full")]


def eval_retrain(vol: str, m, run: str, suffix: str = "", sealed: bool = False, r3: bool = False) -> None:
    e22_other(vol)
    os.makedirs(f"{vol}/hyps/{run}/sealed", exist_ok=True)
    for man, name, limit in (RETRAIN3_SETS if r3 else RETRAIN_SETS):
        transcribe_loaded(vol, m, run, man, name + suffix, limit=limit)
    if sealed:
        for man, name in (SEALED3_SETS if r3 else SEALED_SETS):
            transcribe_loaded(vol, m, run, man, name + suffix)


# ---- R3.15 E-a: multi-head eval. The encoder is frozen in every retrain, so a bed's encoder output is computed once
# and every checkpoint (screening arms, soups, alpha blends) only runs the decoder + joint over the cache. -------------


def _load_audio(path):
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    assert sr == 16000, (path, sr)
    return torch.from_numpy(x)


@torch.no_grad()
def encode_rows(m, rows, batch_size: int = 16) -> list:
    """Encoder output per row, fp16 [D, T] on CPU, in row order (batches sorted by length, like NeMo's bucketing)."""
    order = sorted(range(len(rows)), key=lambda i: rows[i]["duration"])
    out = [None] * len(rows)
    for k in range(0, len(order), batch_size):
        idx = order[k:k + batch_size]
        wavs = [_load_audio(rows[i]["audio_filepath"]) for i in idx]
        lens = torch.tensor([len(w) for w in wavs])
        x = torch.zeros(len(wavs), int(lens.max()))
        for j, w in enumerate(wavs):
            x[j, :len(w)] = w
        feats, flen = m.preprocessor(input_signal=x.cuda(), length=lens.cuda())
        enc, elen = m.encoder(audio_signal=feats, length=flen)
        for j, i in enumerate(idx):
            out[i] = enc[j, :, :int(elen[j])].half().cpu()
    return out


@torch.no_grad()
def decode_encoded(m, encs: list, batch_size: int = 32) -> list:
    order = sorted(range(len(encs)), key=lambda i: encs[i].shape[1])
    out = [None] * len(encs)
    for k in range(0, len(order), batch_size):
        idx = order[k:k + batch_size]
        T = max(encs[i].shape[1] for i in idx)
        D = encs[idx[0]].shape[0]
        x = torch.zeros(len(idx), D, T, device="cuda")
        lens = torch.tensor([encs[i].shape[1] for i in idx], device="cuda")
        for j, i in enumerate(idx):
            x[j, :, :encs[i].shape[1]] = encs[i].float().cuda()
        hyps = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=x, encoded_lengths=lens,
                                                          return_hypotheses=True)
        if isinstance(hyps, tuple):
            hyps = hyps[0]
        for j, i in enumerate(idx):
            h = hyps[j]
            ids = h.y_sequence.tolist() if torch.is_tensor(h.y_sequence) else list(h.y_sequence)
            text = h.text if isinstance(h.text, str) else m.tokenizer.ids_to_text(ids)
            out[i] = {"text": text, "ids": ids}
    return out


def cache_beds(vol: str, m, sets: list, cache_dir: str = "enc") -> dict:
    """Encode each bed once -> /vol/<cache_dir>/<name>.pt (rows + fp16 encoder outputs)."""
    os.makedirs(f"{vol}/{cache_dir}", exist_ok=True)
    st = {}
    for man, name, limit in sets:
        out = f"{vol}/{cache_dir}/{name.replace('/', '__')}.pt"
        if not os.path.exists(f"{vol}/{man}"):
            st[name] = "not prepared"
            continue
        if os.path.exists(out) and os.path.getmtime(out) > os.path.getmtime(f"{vol}/{man}"):
            st[name] = "cached"
            continue
        rows = [json.loads(line) for line in open(f"{vol}/{man}")]
        rows = rows[:limit] if limit else rows
        torch.save({"rows": rows, "enc": encode_rows(m, rows)}, out)
        st[name] = len(rows)
    return st


def eval_cached(vol: str, m, run: str, sets: list, cache_dir: str = "enc") -> None:
    """Same hyps files as transcribe_loaded, decoded from the cached encoder outputs."""
    out_dir = f"{vol}/hyps/{run}"
    os.makedirs(f"{out_dir}/sealed", exist_ok=True)
    for _man, name, _limit in sets:
        p = f"{vol}/{cache_dir}/{name.replace('/', '__')}.pt"
        if not os.path.exists(p):
            print(f"skip {name}: no cache")
            continue
        c = torch.load(p)
        hyps = decode_encoded(m, c["enc"])
        n_digit = 0
        with open(f"{out_dir}/{name}.jsonl", "w") as f:
            for r, h in zip(c["rows"], hyps):
                n_digit += bool(WRITTEN_FORM.search(h["text"]))
                f.write(json.dumps({"id": r.get("id", os.path.basename(r["audio_filepath"])), "ref": r["text"],
                                    "hyp": h["text"], "ids": h["ids"]}) + "\n")
        print(f"{run} {name}: {len(hyps)} decoded (cached encoder), {n_digit} with written-form characters")
