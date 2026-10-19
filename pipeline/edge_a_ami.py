"""Real English disfluent / colloquial number speech from the AMI meeting corpus (train split of the IHM headset config, CC BY
4.0): human verbatim transcripts already spell numbers and keep fillers ("TWENTY UH THIRTY", "ABOUT UM FORTY PERCENT"), so
the label is the transcript itself (cased as a sentence, letter names as capital letters) and the audio is checked by the
same strict probes as every edge row. Text scan + audio fetch; new volume paths only (ml/edgeA/real/ami/en/).

  candidate(text)   an utterance -> row skeleton (text, edge_spans, construction, values) or None
  scan / fetch      Modal: the text columns of the train shards, then the audio of the picked utterances

    modal run edge_a_ami.py::scan --out hits_ami.jsonl
    modal run edge_a_ami.py::fetch --hits sel_ami.jsonl
"""
import json
import os
import re

import edge_a as E
import edge_a_real as RL

REPO = "edinburghcstr/ami"
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
SCALE = {"hundred", "thousand", "million", "billion"}
NUM = set(ONES + TENS) | SCALE
FILL = {"uh", "um", "mm", "hmm", "er", "erm", "uhm"}
FIX = [("sorry",), ("i", "mean"), ("no", "wait"), ("or", "rather"), ("no",), ("actually",), ("sorry", "no")]
LEAD = {"about", "around", "like", "maybe", "roughly", "is", "was", "of", "at", "to", "for"}


def tokens(text: str) -> list | None:
    """[(word, is_letter_name)] or None when the utterance holds a truncated word, a digit or anything but words."""
    out = []
    for raw in text.split():
        is_letter = bool(re.fullmatch(r"[A-Z]\.", raw))
        w = raw.rstrip(".").lower() if is_letter else raw.lower()
        if not re.fullmatch(r"[a-z]+(?:'[a-z]+)*", w):
            return None
        out.append((w, is_letter))
    return out


def _numrun(ws: list, i: int, step: int = 1) -> int:
    """Length of the run of number words starting at i (forward) - 'and' only between number words, 'a' only before a scale."""
    n = 0
    while 0 <= i + n * step < len(ws):
        w = ws[i + n * step]
        if w in NUM or (w == "a" and n == 0 and i + 1 < len(ws) and ws[i + 1] in SCALE):
            n += 1
        elif w == "and" and n and 0 <= i + (n + 1) * step < len(ws) and ws[i + (n + 1) * step] in NUM:
            n += 1
        else:
            break
    return n


def candidate(text: str, split: dict | None = None) -> dict | None:
    tk = tokens(text)
    if tk is None or len(tk) < 5:
        return None
    ws = [w for w, _ in tk]
    found = None
    for i in range(len(ws)):
        # number FILLER number
        n1 = _numrun(ws, i)
        if n1 and i + n1 < len(ws) and ws[i + n1] in FILL:
            j = i + n1 + 1
            n2 = _numrun(ws, j)
            if n2 and ws[i:i + n1] != ws[j:j + n2]:
                found = ("en.dis.uh_pair", i, j + n2)
                break
        # number FIX number (self-correction, two different numbers)
        for fx in FIX:
            k = i + n1
            if n1 and tuple(ws[k:k + len(fx)]) == fx:
                j = k + len(fx)
                n2 = _numrun(ws, j)
                if n2 and ws[i:i + n1] != ws[j:j + n2]:
                    found = ("en.dis.self_fix", i, j + n2)
                    break
        if found:
            break
        # lead FILLER number
        if ws[i] in LEAD and i + 2 < len(ws) and ws[i + 1] in FILL and _numrun(ws, i + 2):
            found = ("en.dis.filler_before", i, i + 2 + _numrun(ws, i + 2) + (1 if i + 2 + _numrun(ws, i + 2) < len(ws) and ws[i + 2 + _numrun(ws, i + 2)] == "percent" else 0))
            break
    if found is None:
        d, rx = RL.index("en")
        low = " ".join(ws)
        m = rx.search(low)
        if not m:
            return None
        cid, key = d[RL.norm(m.group(0))]
        a = len(low[:m.start()].split())
        found = (cid, a, a + len(m.group(0).split()))
    cid, a, b = found
    words = []
    for k, (w, is_letter) in enumerate(tk):
        words.append(w.upper() if is_letter else ("I" + w[1:] if (w == "i" or w.startswith("i'")) else w))
    label = " ".join(words)
    label = label[:1].upper() + label[1:] + "."
    pos = [0]
    for w in words:
        pos.append(pos[-1] + len(w) + 1)
    span = [pos[a], pos[b] - 1]
    key = " ".join(ws[a:b])
    side = "A"
    if split is not None:
        side = E.side_of(split, cid, key)
        if side == "?":
            side = E.hash_side("en", cid, key)
    return {"text": label, "caption": text, "edge_spans": [span], "num_chars": [span],
            "spans": [{"text": label[span[0]:span[1]], "kind": "edge", "reading": label[span[0]:span[1]], "canonical": label[span[0]:span[1]],
                       "idx": 0, "n_cands": 1, "slot": "X"}],
            "constructions": [cid], "values": [key], "construction": cid, "family": cid.split(".")[1], "edge": [cid.split(".")[1]],
            "side": side, "label_style": "verbatim_unpunctuated"}


try:
    import modal

    app = modal.App("parakeet-edge-a-ami")
    image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("pyarrow", "huggingface_hub", "hf_transfer", "soundfile", "numpy", "librosa")
             .apt_install("libsndfile1", "ffmpeg")
             .add_local_python_source("edge_a_ami", "edge_a_real", "edge_a", "edge_a_content", "edge_a_frames", "edge_a_words", "ml_cands",
                                      "readings")
             .add_local_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_grammar_table.json"), "/root/ml_grammar_table.json")
             .add_local_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "edge_a_split"), "/root/edge_a_split"))
    hf = modal.Secret.from_name("hf")
    vol = modal.Volume.from_name("parakeet-spoken-form")

    @app.function(image=image, secrets=[hf], cpu=1, memory=4096, timeout=3600, retries=2, max_containers=8)
    def scan_shard(path: str) -> list:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        split = json.load(open("/root/edge_a_split/ec1split_en.json", encoding="utf-8"))
        pf = pq.ParquetFile(HfFileSystem().open(f"datasets/{REPO}/{path}", block_size=1 << 20))
        rows = pf.read(columns=["meeting_id", "audio_id", "text", "begin_time", "end_time", "speaker_id"]).to_pylist()
        out = []
        for r in rows:
            if not 2.5 <= r["end_time"] - r["begin_time"] <= 25 and False:
                continue
            c = candidate(r["text"] or "", split)
            if c:
                out.append(dict(r, shard=path, cand=c, utt_id=r["audio_id"]))
        return out

    @app.local_entrypoint()
    def scan(out: str = "hits_ami.jsonl"):
        import collections

        paths = [f"ihm/train-{i:05d}-of-00042.parquet" for i in range(42)]
        tot, n = collections.Counter(), 0
        with open(out, "w", encoding="utf-8") as f:
            for res in scan_shard.map(paths, return_exceptions=True, order_outputs=False):
                if isinstance(res, Exception):
                    print("FAIL", repr(res)[:200])
                    continue
                for h in res:
                    tot[h["cand"]["construction"]] += 1
                    n += 1
                    f.write(json.dumps(h, ensure_ascii=False) + "\n")
        print(json.dumps({"shards": len(paths), "hits": n, "by_construction": dict(tot)}))

    @app.function(image=image, secrets=[hf], volumes={"/vol": vol}, cpu=2, memory=8192, timeout=3600, retries=3, max_containers=6)
    def fetch_shard(path: str, rows: list) -> list:
        import io

        import librosa
        import numpy as np
        import pyarrow.parquet as pq
        import soundfile as sf
        from huggingface_hub import hf_hub_download

        want = {r["utt_id"]: r for r in rows}
        out_dir = "/vol/ml/edgeA/real/ami/en/wav"
        os.makedirs(out_dir, exist_ok=True)
        local = hf_hub_download(REPO, path, repo_type="dataset", local_dir="/tmp/shard")
        pf = pq.ParquetFile(local)
        kept = []
        for rg in range(pf.metadata.num_row_groups):
            for x in pf.read_row_group(rg, columns=["audio_id", "audio"]).to_pylist():
                r = want.get(x["audio_id"])
                if r is None:
                    continue
                a, sr = sf.read(io.BytesIO(x["audio"]["bytes"]), dtype="float32")
                a = a.mean(1) if a.ndim > 1 else a
                if sr != 16000:
                    a = librosa.resample(a, orig_sr=sr, target_sr=16000)
                wav = f"{out_dir}/{x['audio_id']}.wav"
                sf.write(wav, a.astype(np.float32), 16000, subtype="PCM_16")
                kept.append(dict(r, audio_filepath=wav, duration=round(len(a) / 16000, 3)))
        os.remove(local)
        vol.commit()
        return kept

    @app.local_entrypoint()
    def fetch(hits: str):
        import collections

        rows = [json.loads(l) for l in open(hits) if l.strip()]
        by = collections.defaultdict(list)
        for r in rows:
            by[r["shard"]].append(r)
        kept = []
        for res in fetch_shard.starmap([(p, rs) for p, rs in by.items()], return_exceptions=True):
            if isinstance(res, Exception):
                print("FAIL", repr(res)[:200])
                continue
            kept += res
        out = hits.replace(".jsonl", "_fetched.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(json.dumps({"rows": len(kept), "hours": round(sum(r["duration"] for r in kept) / 3600, 2)}))
except ImportError:
    pass
