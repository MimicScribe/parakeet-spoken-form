"""S3 audio fetch: the selected rows of one YODAS-Granary shard -> 16 kHz wav files on the volume.

The whole shard is downloaded once (one request; column-projected range reads hit HF rate limits), rows are kept by
the S0 scan file of that shard: English = number rows the labeller can parse + the 2% number-free sample. The parquet
audio is already 16 kHz 16-bit mono WAV, so bytes are written as they are.
"""

import json
import os


def fetch_shard(vol: str, path: str, lang: str) -> dict:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    import labeller

    name = path.split("/")[1] + "_" + os.path.basename(path)[:-8]
    scan = f"{vol}/s0/scan/{lang}/{name}.jsonl"
    out_dir = f"{vol}/yodas/{lang}/{name}"
    man = f"{vol}/yodas/{lang}/manifests/{name}.jsonl"
    if os.path.exists(man):
        return {"shard": name, "skipped": True}
    rows = {}
    drop = {}
    for line in open(scan):
        r = json.loads(line)
        if lang == "en" and r["kind"] != "plain":
            why = labeller.row_ok(r["text"], labeller.spans(r["text"]))
            if why:
                drop[why] = drop.get(why, 0) + 1
                continue
        rows[r["utt_id"]] = r
    local = hf_hub_download("espnet/yodas-granary", path, repo_type="dataset", local_dir="/tmp/shard")
    pf = pq.ParquetFile(local)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(man), exist_ok=True)
    kept = []
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=["utt_id", "audio"]).to_pylist()
        for x in t:
            r = rows.get(x["utt_id"])
            if r is None:
                continue
            wav = f"{out_dir}/{x['utt_id']}.wav"
            with open(wav, "wb") as f:
                f.write(x["audio"]["bytes"])
            r["audio_filepath"] = wav
            kept.append(r)
    os.remove(local)
    with open(man, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"shard": name, "kept": len(kept), "h": sum(r["duration"] for r in kept) / 3600, "drop": drop}
