#!/usr/bin/env python3
"""Run the never-reject CTC labeller over a manifest on this machine (torch + transformers + soundfile; CPU, CUDA or MPS).

  python3 label_local.py --manifest manifest.jsonl --out scored.jsonl [--model facebook/wav2vec2-large-960h-lv60-self] [--device cpu] [--threads 2] [--limit N]

The manifest comes from manifest.py (one row per label with its candidate readings per span). The output has one record per input row, in order
(labeller_core.score_row: status, chosen label, per-span top-2 losses, margins, ties, greedy transcript, word times); an existing output file is resumed.
The model must be a character-level English CTC model whose vocabulary is labeller_core.VOCAB (the wav2vec2 960h family): checked at start.
Env: HF_HOME (model cache). Nothing is uploaded; the model is downloaded from the model hub on first use unless it is already cached."""
import argparse, json, os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import labeller_core as C


def done_ids(path):
    ids = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    ids.add(json.loads(line)["id"])
                except (ValueError, KeyError):
                    pass            # a torn last line from an interrupted run is simply redone
    return ids


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="facebook/wav2vec2-large-960h-lv60-self")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--load-workers", type=int, default=2)
    ap.add_argument("--score-workers", type=int, default=1)
    a = ap.parse_args(argv)
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    import label_runner as R, w2v_torch as W
    torch.set_num_threads(a.threads)
    proc = Wav2Vec2Processor.from_pretrained(a.model)
    model = Wav2Vec2ForCTC.from_pretrained(a.model).eval().to(a.device)
    vocab = proc.tokenizer.get_vocab()
    if vocab != C.VOCAB:
        sys.exit("model vocabulary differs from labeller_core.VOCAB: this driver needs a character-level English CTC model (wav2vec2 960h family)")
    with open(a.manifest, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    t0 = time.time()
    summary, _ = R.run_part(rows, a.out, proc, model, vocab, C, W, done_ids=done_ids(a.out), load_workers=a.load_workers, score_workers=a.score_workers,
                            window=8, max_pending=4, log=lambda s: print(s, file=sys.stderr), gpu_reader=None)
    summary["sec_total"] = round(time.time() - t0, 1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
