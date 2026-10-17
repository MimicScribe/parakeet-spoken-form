"""Second, non-Parakeet acoustic veto on the labeller's decided readings.

wav2vec2-large-960h CTC decides the reading and is also the absolute-fit check, so a systematic wav2vec2 bias would go
unseen. HuBERT-large (a different self-supervised objective and pretraining run, CTC fine-tuned on LibriSpeech 960 h)
scores every multi-candidate span of every labelled row: a reading is VETOED when HuBERT's best candidate has
different words. Vetoed readings drop the crops that contain them (mix_build).

Writes /vol/d1/veto/<tag>.json: {"utt_id": [vetoed reading, ...]} plus counts.
"""

import json
import os

import labeller as L
from label_gpu import norm_ctc

HUBERT = "facebook/hubert-large-ls960-ft"


class Veto:
    def __init__(self):
        import torch
        from transformers import HubertForCTC, Wav2Vec2Processor

        self.torch = torch
        self.proc = Wav2Vec2Processor.from_pretrained(HUBERT, cache_dir="/tmp/hub")
        self.m = HubertForCTC.from_pretrained(HUBERT, cache_dir="/tmp/hub").cuda().eval()
        self.vocab = self.proc.tokenizer.get_vocab()

    def nll(self, lp, text):
        torch = self.torch
        ids = [self.vocab[c] for c in norm_ctc(text).replace(" ", "|") if c in self.vocab]
        if not ids or len(ids) > lp.shape[0]:
            return float("inf")
        return float(torch.nn.functional.ctc_loss(lp[:, None], torch.tensor([ids], device="cuda"),
                                                  torch.tensor([lp.shape[0]]), torch.tensor([len(ids)]),
                                                  blank=self.proc.tokenizer.pad_token_id, reduction="sum"))

    def rows(self, vol: str, tag: str, rows_dir: str = "d1/rows", out_dir: str = "d1/veto",
             man_dir: str = "yodas/en/manifests") -> dict:
        import soundfile as sf

        man = {}
        recs = [json.loads(x) for x in open(f"{vol}/{rows_dir}/{tag}.jsonl")]
        shard = tag.removesuffix("_pct")
        for x in open(f"{vol}/{man_dir}/{shard}.jsonl"):
            r = json.loads(x)
            man[r["utt_id"]] = r["audio_filepath"]
        out, hub, n_span, n_veto = {}, {}, 0, 0
        for rec in recs:
            spans_out = rec.get("spans") or []
            multi = [s for s in spans_out if s.get("decided") and s.get("n", 1) > 1]
            if not multi or rec.get("drop"):
                continue
            x, _ = sf.read(man[rec["utt_id"]], dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            with self.torch.no_grad():
                iv = self.proc(x, sampling_rate=16000, return_tensors="pt").input_values.cuda()
                lp = self.m(iv).logits.float().log_softmax(-1)[0]
            sp = L.spans(rec["caption"])
            picks = [s["pick"] for s in spans_out]
            if len(sp) != len(spans_out):
                continue
            for i, s in enumerate(spans_out):
                if not (s.get("decided") and s.get("n", 1) > 1):
                    continue
                n_span += 1
                scores = []
                for c in range(len(sp[i].cands)):
                    ch = picks[:]
                    ch[i] = c
                    scores.append(self.nll(lp, L.render(rec["caption"], sp, ch)))
                best = min(range(len(scores)), key=scores.__getitem__)
                hub.setdefault(rec["utt_id"], {})[str(i)] = {"best": sp[i].cands[best], "pick": sp[i].cands[s["pick"]],
                                                              "nll": [round(v, 2) for v in scores]}
                if norm_ctc(sp[i].cands[best]) != norm_ctc(sp[i].cands[s["pick"]]):
                    n_veto += 1
                    out.setdefault(rec["utt_id"], []).append(sp[i].cands[s["pick"]])
        os.makedirs(f"{vol}/{out_dir}", exist_ok=True)
        json.dump({"vetoed": out, "hubert": hub, "spans": n_span, "vetoes": n_veto},
                  open(f"{vol}/{out_dir}/{tag}.json", "w"))
        return {"tag": tag, "spans": n_span, "vetoes": n_veto}
