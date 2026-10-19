"""Measured exposure: which training rows the dataloader actually yielded, per optimizer step.

"Rows seen" used to be estimated from batch math (batch_duration x accumulate seconds per step). This module logs the
real thing and reads it back.

Files in the run's folder (<exp>):
  row_index.<sha12>.jsonl  one line per line of that train manifest, in manifest order: {"i", "h", "d", "part", "tag",
                           "lang", "kinds"}. `h` is the row's token-id hash (the trainer sees tokens, not text), `d` its
                           duration in centiseconds. Written once per manifest (sha12 = the manifest's sha256 prefix).
  seen_rows.jsonl          append-only. A {"start": step0, "manifest": sha12} line when an attempt begins, then one
                           {"step": N, "h": [...], "d": [...]} line per optimizer step (the hashes and centisecond
                           durations of every row of its micro-batches, repeats included).
Resume safety: an attempt that starts at step0 supersedes every earlier line with step >= step0 (those steps were redone
from the checkpoint), so the reader (`effective_steps`) drops them.
No torch here: batches are read through `.tolist()`.
"""

import collections
import hashlib
import json
import os

SEEN = "seen_rows.jsonl"
SAMPLE_RATE = 16000


def token_hash(ids) -> str:
    """40-bit hash of a row's token ids (10 hex chars): collisions are rare enough at ~1e5 rows and the reader resolves
    the rest by duration."""
    return hashlib.blake2b(",".join(str(int(i)) for i in ids).encode(), digest_size=5).hexdigest()


def row_cells(row: dict) -> list[str]:
    """The number kinds a row carries (spans, a window's own kinds, or its class), for the per-cell exposure."""
    if row.get("spans"):
        return sorted({str(s.get("kind", "?")) for s in row["spans"]})
    if row.get("origin") == "window" and row.get("kinds"):
        return sorted({str(k) for k in row["kinds"]})
    if row.get("cls"):
        return [str(row["cls"])]
    return []


def row_lang(row: dict) -> str:
    return (row.get("lang") or "en")[:2]


def row_meta(row: dict, part: str, tag) -> dict:
    return {"part": part, "tag": tag or "", "lang": row_lang(row), "kinds": row_cells(row)}


def index_path(exp: str, sha12: str) -> str:
    return f"{exp}/row_index.{sha12}.jsonl"


def write_row_index(exp: str, sha12: str, manifest_rows: list[dict], metas: list[dict], token_ids) -> str:
    """Write the index for one manifest (once: an existing file is kept). `token_ids(text)` = the trainer's tokenizer."""
    path = index_path(exp, sha12)
    if os.path.exists(path):
        return path
    assert len(manifest_rows) == len(metas), (len(manifest_rows), len(metas))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for i, (x, meta) in enumerate(zip(manifest_rows, metas)):
            f.write(json.dumps({"i": i, "h": token_hash(token_ids(x["text"])), "d": round(x["duration"] * 100), **meta},
                               ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return path


def batch_rows(batch) -> tuple[list[str], list[int]]:
    """(token hashes, centisecond durations) of the rows of one trainer batch (signal, signal_len, tokens, tokens_len)."""
    signal_len, tokens, tokens_len = batch[1].tolist(), batch[2].tolist(), batch[3].tolist()
    hashes = [token_hash(tokens[i][:int(tokens_len[i])]) for i in range(len(tokens_len))]
    return hashes, [round(int(n) * 100 / SAMPLE_RATE) for n in signal_len]


class SeenLog:
    """Collects the micro-batches of one optimizer step and appends them as one line. Not a Lightning class (the
    callback in ft_train is a thin wrapper), so it runs against a fake dataloader."""

    def __init__(self, exp: str, sha12: str, step0: int):
        self.path = f"{exp}/{SEEN}"
        self.step, self.h, self.d = None, [], []
        self._write({"start": int(step0), "manifest": sha12})

    def _write(self, rec: dict):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def add(self, step: int, batch):
        if self.step is not None and step != self.step:
            self.flush()
        h, d = batch_rows(batch)
        self.step = int(step)
        self.h += h
        self.d += d

    def end_batch(self, step_now: int):
        """After a micro-batch: the optimizer step completed (the trainer's step moved on), so the step's line is final."""
        if self.step is not None and int(step_now) != self.step:
            self.flush()

    def flush(self):
        if self.step is not None and self.h:
            self._write({"step": self.step, "h": self.h, "d": self.d})
        self.step, self.h, self.d = None, [], []


def effective_steps(lines):
    """Yield (manifest sha12, step record) for the steps that count: a later attempt starting at step0 drops the earlier
    records with step >= step0. Returns the records in step order within the final attempt chain."""
    kept: list[tuple[str, dict]] = []
    sha = ""
    for ln in lines:
        if not ln.strip():
            continue
        rec = json.loads(ln)
        if "start" in rec:
            sha = rec.get("manifest", "")
            kept = [(s, r) for s, r in kept if r["step"] < rec["start"]]
        else:
            kept.append((sha, rec))
    return kept


def load_index(exp: str, sha12: str) -> list[dict]:
    with open(index_path(exp, sha12), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _resolver(index: list[dict]):
    by_h = collections.defaultdict(list)
    for r in index:
        by_h[r["h"]].append(r)

    def resolve(h: str, d: int, tol: int = 2):
        """Candidate index rows for one seen row: the rows with that token hash, narrowed to the duration when several."""
        cand = by_h.get(h, [])
        if len(cand) > 1:
            near = [r for r in cand if abs(r["d"] - d) <= tol]
            cand = near or cand
        return cand

    return resolve


def exposure(exp: str, group_by=("part",)) -> dict:
    """Per part and per (lang, kind) cell: listed rows/seconds vs drawn rows/seconds, from the effective seen log.
    A draw whose hash matches several listed rows is split evenly between them (reported as `ambiguous`)."""
    path = f"{exp}/{SEEN}"
    lines = open(path, encoding="utf-8").read().splitlines() if os.path.exists(path) else []
    steps = effective_steps(lines)
    indexes: dict[str, list[dict]] = {}
    resolvers = {}
    out = {"steps": 0, "rows_drawn": 0, "seconds_drawn": 0.0, "unmatched": 0, "ambiguous": 0, "manifests": [],
           "parts": {}, "cells": {}, "tags": {}, "langs": {}}
    fresh = lambda: {"listed_rows": 0, "listed_unique": 0, "listed_sec": 0.0, "draws": 0.0, "draw_sec": 0.0,  # noqa: E731
                     "distinct": 0.0}

    def bucket(kind, key):
        return out[kind].setdefault(key, fresh())

    drawn: dict[tuple[str, int], float] = collections.defaultdict(float)  # (sha, row index) -> draws
    for sha, rec in steps:
        if sha not in indexes:
            indexes[sha] = load_index(exp, sha)
            resolvers[sha] = _resolver(indexes[sha])
            out["manifests"].append(sha)
        out["steps"] += 1
        for h, d in zip(rec["h"], rec["d"]):
            out["rows_drawn"] += 1
            out["seconds_drawn"] += d / 100
            cand = resolvers[sha](h, d)
            if not cand:
                out["unmatched"] += 1
                continue
            if len(cand) > 1:
                out["ambiguous"] += 1
            for r in cand:
                drawn[(sha, r["i"])] += 1 / len(cand)
    # listed: the LAST manifest of the run (the one a finished run trained on); draws from earlier manifests of a
    # non-r3 resume are folded in by part/cell key, so only the draws side spans manifests.
    # A part listed n times has n index lines with the same (part, token hash, duration): a draw cannot tell the copies
    # apart, so draws are split between them. "distinct" counts such a group ONCE (a unique row drawn at least once is
    # one distinct row, however many copies the listing holds); listed_unique is the matching denominator.
    last = out["manifests"][-1] if out["manifests"] else None
    group_draws: dict[tuple, float] = collections.defaultdict(float)
    for sha, idx in indexes.items():
        for r in idx:
            group_draws[(sha, r["part"], r["h"], r["d"])] += drawn.get((sha, r["i"]), 0.0)
    seen_groups: dict[tuple, set] = collections.defaultdict(set)
    for sha, idx in indexes.items():
        for r in idx:
            n = drawn.get((sha, r["i"]), 0.0)
            g = (sha, r["part"], r["h"], r["d"])
            keys = [("parts", r["part"]), ("tags", r["tag"] or "(none)"), ("langs", r["lang"])]
            keys += [("cells", f"{r['lang']}/{k}") for k in r["kinds"]]
            for kind, key in keys:
                b = bucket(kind, key)
                if sha == last:
                    b["listed_rows"] += 1
                    b["listed_sec"] += r["d"] / 100
                b["draws"] += n
                b["draw_sec"] += n * r["d"] / 100
                if g not in seen_groups[(kind, key)]:  # one count per unique row in this bucket
                    seen_groups[(kind, key)].add(g)
                    if sha == last:
                        b["listed_unique"] += 1
                    b["distinct"] += min(group_draws[g], 1.0)
    out["listed_rows"] = len(indexes[last]) if last else 0
    return out


def report(exp: str) -> str:
    e = exposure(exp)
    if not e["steps"]:
        return f"{exp}: no seen_rows.jsonl records"
    L = [f"{exp}: {e['steps']} optimizer steps, {e['rows_drawn']} rows drawn ({e['seconds_drawn'] / 3600:.2f} h), "
         f"{e['rows_drawn'] / e['steps']:.1f} rows and {e['seconds_drawn'] / e['steps']:.0f} s per step; "
         f"manifest(s) {','.join(e['manifests'])}; unmatched {e['unmatched']}, ambiguous {e['ambiguous']}"]
    for title, kind in (("PART", "parts"), ("TAG", "tags"), ("LANG", "langs"), ("CELL (lang/kind; a row counts in each "
                                                                                   "of its kinds)", "cells")):
        L += ["", f"{title:<46} {'listed':>8} {'drawn':>9} {'draws/row':>9} {'distinct%':>9} {'listed h':>8} {'drawn h':>8}"]
        for key, b in sorted(e[kind].items()):
            lr, lu = b["listed_rows"], b["listed_unique"]
            L.append(f"{key:<46} {lr:>8} {b['draws']:>9.0f} {b['draws'] / lr if lr else 0:>9.2f} "
                     f"{100 * b['distinct'] / lu if lu else 0:>9.1f} {b['listed_sec'] / 3600:>8.2f} "
                     f"{b['draw_sec'] / 3600:>8.2f}")
    return "\n".join(L)
