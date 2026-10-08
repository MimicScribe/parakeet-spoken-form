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
