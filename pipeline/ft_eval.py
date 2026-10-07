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
