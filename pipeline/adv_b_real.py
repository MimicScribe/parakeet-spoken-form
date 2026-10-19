"""Real public examples for the adversarial TEST set (advB): AMI meeting-corpus utterances (train split of the IHM headset config,
CC BY 4.0) with a disfluent number ("twelve fifty twelve uh twelve", "fifty mil fifty million", "ten ten out of ten"). The text scan is
edge_a_ami's (hits it did not put into the tier-A pack are the candidates here: a real example is never in a pack AND a test set); this
module only fetches the audio of the picked utterances to a NEW volume path. The label is the human transcript, cased; the audio is then
checked by adv_b_ctc in real mode (CER <= 0.10, the edge words decode) and by the two blind transcripts, like every other row.

    modal run adv_b_real.py::fetch --hits sel_ami_test.jsonl           # writes ml/advB/real/ami/en/wav/<id>.wav, sel_ami_test_fetched.jsonl
"""
import collections
import json
import os

REPO = "edinburghcstr/ami"
try:
    import modal

    app = modal.App("parakeet-adv-b-real")
    image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("pyarrow", "huggingface_hub", "hf_transfer", "soundfile", "numpy", "librosa")
             .apt_install("libsndfile1", "ffmpeg"))
    hf = modal.Secret.from_name("hf")
    vol = modal.Volume.from_name("parakeet-spoken-form")

    @app.function(image=image, secrets=[hf], volumes={"/vol": vol}, cpu=2, memory=8192, timeout=3600, retries=3, max_containers=6)
    def fetch_shard(path: str, rows: list) -> list:
        import io

        import librosa
        import numpy as np
        import pyarrow.parquet as pq
        import soundfile as sf
        from huggingface_hub import hf_hub_download

        want = {r["utt_id"]: r for r in rows}
        out_dir = "/vol/ml/advB/real/ami/en/wav"
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
