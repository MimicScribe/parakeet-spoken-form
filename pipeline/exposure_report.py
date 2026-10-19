"""Measured exposure of a finished (or running) training run: rows the dataloader actually yielded, per part / tag /
language / targeted cell, against what the manifest listed.

    python3 exposure_report.py <run folder with seen_rows.jsonl, row_index.*.jsonl, train_manifest.jsonl> [--json out]

The run folder is <volume>/exp/<run>. Needs no torch. Reads the log with resumes applied (exposure.effective_steps).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exposure  # noqa: E402


def check_manifest(exp: str) -> list[str]:
    """Warnings: the run's train_manifest.jsonl must be the one the (last) row index describes."""
    import hashlib

    warn = []
    path = f"{exp}/train_manifest.jsonl"
    seen = f"{exp}/{exposure.SEEN}"
    if not os.path.exists(seen):
        return [f"{seen} missing: the run logged no rows"]
    shas = [r.get("manifest") for r in map(json.loads, filter(str.strip, open(seen))) if "start" in r]
    if not os.path.exists(path):
        return [f"{path} missing: listed counts come from the row index only"]
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += chunk.count(b"\n")
    if shas and h.hexdigest()[:12] != shas[-1]:
        warn.append(f"train_manifest.jsonl sha {h.hexdigest()[:12]} != the last attempt's manifest {shas[-1]}")
    if shas and os.path.exists(exposure.index_path(exp, shas[-1])):
        k = len(exposure.load_index(exp, shas[-1]))
        if k != n:
            warn.append(f"train_manifest.jsonl has {n} rows, the row index {k}")
    if len(set(shas)) > 1:
        warn.append(f"{len(set(shas))} different manifests across attempts: draws span all, listed counts are the last")
    return warn


def main(argv):
    exp = argv[0]
    for w in check_manifest(exp):
        print(f"WARNING: {w}")
    print(exposure.report(exp))
    if "--json" in argv:
        with open(argv[argv.index("--json") + 1], "w") as f:
            json.dump(exposure.exposure(exp), f, indent=1, sort_keys=True)


if __name__ == "__main__":
    main(sys.argv[1:])
