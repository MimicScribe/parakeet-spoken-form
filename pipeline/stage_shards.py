"""Tar-shard staging of a mix's audio.

Staging a mix means copying every audio file its training manifest names (~360k unique small files) from the network
volume to the container's local disk. Read file by file, the speed depends on per-file latency and swung from 22 to
1,700 files/s between identical containers (2026-10-18). The same bytes in a few large tar shards are read
sequentially and do not.

    stage_shards/<manifest sha12>/shard-00000.tar ...   the audio files, sorted by path, name = the path without its "/"
    stage_shards/<manifest sha12>/index.json            written last: the shards, counts, sizes and the sha256 of the path list

`<manifest sha12>` is the first 12 hex digits of the sha256 of the training manifest as write_mix writes it (staged
paths), the sha the training log prints as `manifest sha256 ...`; a mix is the same mix exactly when it is the same sha.

pack() writes the shards (idempotent), stage_from_shards() reads them, stage_rows() reads them and falls back to
common.stage_files whenever the shards are missing or do not match the manifest. Nothing here touches training.
"""

import hashlib
import json
import os
import tarfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

ROOT = "stage_shards"
VERSION = 1
TARGET_BYTES = 2_000_000_000  # a shard is closed before it would pass this
INDEX = "index.json"
_BUF = 8 << 20


class ShardError(Exception):
    """The shards are missing, incomplete or do not match the manifest: stage file by file instead."""


def manifest_line(x: dict) -> str:
    """One line of a training manifest, exactly as write_mix writes it."""
    return json.dumps({k: x[k] for k in ("audio_filepath", "duration", "text")}) + "\n"


def manifest_sha(rows: list) -> str:
    """sha256 (hex) of the manifest write_mix would write for `rows` (their audio_filepath as written)."""
    h = hashlib.sha256()
    for x in rows:
        h.update(manifest_line(x).encode("utf-8"))
    return h.hexdigest()


def unique_paths(rows: list) -> list:
    """The distinct audio paths of `rows`, sorted (the shard order)."""
    return sorted({x["audio_filepath"] for x in rows})


def paths_sha(paths: list) -> str:
    """sha256 (hex) of the sorted distinct path list, one per line."""
    h = hashlib.sha256()
    for p in sorted(set(paths)):
        h.update(p.encode("utf-8") + b"\n")
    return h.hexdigest()


def shard_dir(vol: str, sha: str) -> str:
    return f"{vol}/{ROOT}/{sha[:12]}"


def shard_name(i: int) -> str:
    return f"shard-{i:05d}.tar"


def _member(path: str, size: int) -> tarfile.TarInfo:
    if not path.startswith("/"):
        raise ValueError(f"audio path is not absolute: {path!r}")
    ti = tarfile.TarInfo(path[1:])
    ti.size, ti.mtime, ti.mode, ti.uid, ti.gid = size, 0, 0o644, 0, 0
    return ti


def read_index(out_dir: str, sha: str = "") -> dict:
    """The shard directory's index when it is complete: every shard exists with its indexed size. ShardError if not."""
    path = f"{out_dir}/{INDEX}"
    try:
        with open(path, encoding="utf-8") as f:
            idx = json.load(f)
    except FileNotFoundError:
        raise ShardError(f"no index at {out_dir}")
    except (OSError, ValueError) as e:
        raise ShardError(f"unreadable index at {out_dir}: {e}")
    if idx.get("version") != VERSION:
        raise ShardError(f"index version {idx.get('version')!r}, expected {VERSION}")
    if sha and idx.get("manifest_sha", "")[:12] != sha[:12]:
        raise ShardError(f"index is for manifest {idx.get('manifest_sha', '')[:12]}, not {sha[:12]}")
    shards = idx.get("shards") or []
    if sum(s["n"] for s in shards) != idx.get("n_files") or sum(s["payload_bytes"] for s in shards) != idx.get("payload_bytes"):
        raise ShardError("index totals do not add up")
    for s in shards:
        try:
            size = os.path.getsize(f"{out_dir}/{s['name']}")
        except OSError:
            raise ShardError(f"shard {s['name']} is missing")
        if size != s["bytes"]:
            raise ShardError(f"shard {s['name']} is {size} bytes, the index says {s['bytes']}")
    return idx


def _read_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def pack(paths: list, out_dir: str, sha: str, mix_dir: str = "", target: int = TARGET_BYTES, read=_read_file,
         workers: int = 64, window: int = 1024, every: int = 5000, say=print, clock=time.time, force: bool = False) -> dict:
    """Pack every distinct path of `paths` into tar shards under out_dir (sorted order, a shard closed before it would
    pass `target` bytes) and write index.json last. Idempotent: a directory whose index matches (same manifest sha, same
    path list, every shard at its indexed size) is left alone. Returns the index."""
    names = sorted(set(paths))
    psha = paths_sha(names)
    if not force:
        try:
            idx = read_index(out_dir, sha)
            if idx.get("paths_sha256") == psha and idx.get("n_files") == len(names):
                say(f"pack: {out_dir} already has {len(idx['shards'])} shards for {len(names)} files; nothing to do")
                return idx
        except ShardError as e:
            say(f"pack: {e}; packing")
    os.makedirs(out_dir, exist_ok=True)
    t0, shards, payload = clock(), [], 0
    cur = {"tf": None, "f": None, "n": 0, "payload": 0, "name": ""}

    def close():
        if cur["tf"] is None:
            return
        cur["tf"].close()
        cur["f"].flush()
        os.fsync(cur["f"].fileno())
        cur["f"].close()
        final = f"{out_dir}/{cur['name']}"
        os.replace(final + ".part", final)
        shards.append({"name": cur["name"], "n": cur["n"], "payload_bytes": cur["payload"], "bytes": os.path.getsize(final)})
        cur.update(tf=None, f=None, n=0, payload=0)

    def open_next():
        cur["name"] = shard_name(len(shards))
        cur["f"] = open(f"{out_dir}/{cur['name']}.part", "wb", buffering=_BUF)
        cur["tf"] = tarfile.open(fileobj=cur["f"], mode="w", format=tarfile.GNU_FORMAT, encoding="utf-8")

    pending, nxt, done = deque(), 0, 0
    with ThreadPoolExecutor(workers) as ex:
        while pending or nxt < len(names):
            while nxt < len(names) and len(pending) < window:
                pending.append((names[nxt], ex.submit(read, names[nxt])))
                nxt += 1
            path, fut = pending.popleft()
            data = fut.result()
            if cur["tf"] is not None and cur["n"] > 0 and cur["f"].tell() + 1024 + len(data) + 512 > target:
                close()
            if cur["tf"] is None:
                open_next()
            cur["tf"].addfile(_member(path, len(data)), fileobj=_BytesReader(data))
            cur["n"] += 1
            cur["payload"] += len(data)
            payload += len(data)
            done += 1
            if done % every == 0 or done == len(names):
                say(f"packing {done}/{len(names)} files {done / max(clock() - t0, 1e-6):.0f} files/s "
                    f"{payload / 1e9:.1f} GB {len(shards)} shards closed", flush=True)
    close()
    idx = {"version": VERSION, "manifest_sha": sha, "mix_dir": mix_dir, "n_files": len(names), "payload_bytes": payload,
           "paths_sha256": psha, "target_bytes": target, "shards": shards}
    tmp = f"{out_dir}/{INDEX}.part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, f"{out_dir}/{INDEX}")
    say(f"pack: {len(shards)} shards, {sum(s['bytes'] for s in shards) / 1e9:.1f} GB for {len(names)} files "
        f"in {clock() - t0:.0f}s", flush=True)
    return idx


class _BytesReader:
    """A minimal readable for TarFile.addfile (io.BytesIO would copy `data` once more)."""

    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def read(self, n: int = -1) -> bytes:
        end = len(self.data) if n is None or n < 0 else min(len(self.data), self.pos + n)
        out, self.pos = self.data[self.pos:end], end
        return out


def _write(dst: str, data: bytes) -> None:
    """Write dst through a temporary name, so an interrupted write never leaves a file that looks staged."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = f"{dst}.part{threading.get_ident()}"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dst)


def stage_from_shards(rows: list, stage_dir: str, vol: str, sha: str, workers: int = 8, every: int = 5000, say=print,
                      clock=time.time) -> int:
    """Copy every distinct file of `rows` from the shards for manifest `sha` to stage_dir + its path, shards in parallel.
    Checks before reading: the index is complete and its path-list sha and file count equal the manifest's. Checks while
    reading: each member is wanted, appears once, and has the indexed shard's byte count in total; at the end the files
    staged equal the distinct paths. Raises ShardError on any mismatch (files already staged stay: the fallback skips
    them). Returns the number of distinct files."""
    out_dir = shard_dir(vol, sha)
    idx = read_index(out_dir, sha)
    paths = unique_paths(rows)
    if idx["n_files"] != len(paths) or idx["paths_sha256"] != paths_sha(paths):
        raise ShardError(f"index lists {idx['n_files']} files; the manifest has {len(paths)} distinct (not the same list)")
    wanted = {p[1:] for p in paths}
    t0, lock, count, stop = clock(), threading.Lock(), [0], threading.Event()
    seen_lock, seen = threading.Lock(), set()

    def one(shard: dict) -> None:
        n = payload = 0
        with open(f"{out_dir}/{shard['name']}", "rb", buffering=_BUF) as f, tarfile.open(fileobj=f, mode="r|") as tf:
            for m in tf:
                if stop.is_set():
                    return
                if not m.isreg() or m.name not in wanted:
                    raise ShardError(f"{shard['name']}: unexpected member {m.name!r}")
                data = tf.extractfile(m).read()
                if len(data) != m.size:
                    raise ShardError(f"{shard['name']}: {m.name} is {len(data)} of {m.size} bytes")
                dst = stage_dir + "/" + m.name
                if not (os.path.exists(dst) and os.path.getsize(dst) == len(data)):
                    _write(dst, data)
                with seen_lock:
                    if m.name in seen:
                        raise ShardError(f"{m.name} is in two shards")
                    seen.add(m.name)
                n += 1
                payload += len(data)
                with lock:
                    count[0] += 1
                    if count[0] % every == 0 or count[0] == len(paths):
                        say(f"staging {count[0]}/{len(paths)} files {count[0] / max(clock() - t0, 1e-6):.0f} files/s",
                            flush=True)
        if n != shard["n"] or payload != shard["payload_bytes"]:
            raise ShardError(f"{shard['name']}: read {n} files / {payload} bytes, the index says "
                             f"{shard['n']} / {shard['payload_bytes']}")

    say(f"staging from {len(idx['shards'])} shards ({sum(s['bytes'] for s in idx['shards']) / 1e9:.1f} GB, "
        f"{len(paths)} distinct files) of manifest {sha[:12]}", flush=True)
    try:
        with ThreadPoolExecutor(max(1, min(workers, len(idx["shards"])))) as ex:
            futs = [ex.submit(one, s) for s in idx["shards"]]
            for fu in futs:
                try:
                    fu.result()
                except BaseException:
                    stop.set()
                    raise
    except (tarfile.TarError, EOFError, OSError) as e:
        raise ShardError(f"reading the shards failed: {type(e).__name__}: {e}")
    if len(seen) != len(wanted) or seen != wanted:
        raise ShardError(f"staged {len(seen)} of {len(wanted)} files")
    return len(paths)


def find_manifest(vol: str, sha: str, stage_dir: str = "/tmp/stage", say=print) -> tuple:
    """(manifest path, rows) of the training manifest with this sha: a run's exp/<run>/train_manifest.jsonl whose recorded
    sha256 starts with `sha` (at least 12 hex digits) and whose file still hashes to it. Rows come back with the staged
    prefix `stage_dir` taken off their paths, i.e. as write_mix holds them before staging. ShardError if none matches."""
    import glob

    if len(sha) < 12:
        raise ShardError(f"manifest sha {sha!r} is shorter than 12 hex digits")
    for marker in sorted(glob.glob(f"{vol}/exp/*/train_manifest.sha256")):
        try:
            recorded = open(marker, encoding="utf-8").read().strip()
        except OSError:
            continue
        path = os.path.join(os.path.dirname(marker), "train_manifest.jsonl")
        if not recorded.startswith(sha) or not os.path.exists(path):
            continue
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != recorded:
            say(f"{path} no longer hashes to its recorded sha; skipped")
            continue
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        bad = [x["audio_filepath"] for x in rows if not x["audio_filepath"].startswith(stage_dir + "/")]
        if bad:
            raise ShardError(f"{path}: {len(bad)} paths are not under {stage_dir}/ (first: {bad[0]!r})")
        return path, [{**x, "audio_filepath": x["audio_filepath"][len(stage_dir):]} for x in rows]
    raise ShardError(f"no exp/*/train_manifest.jsonl on the volume has sha {sha}")


def stage_rows(rows: list, stage_dir: str, vol: str, sha: str, fallback, say=print, **kw) -> list:
    """The rows pointing at their staged copies (as common.stage_files returns them): from the shards for manifest `sha`
    when they are there and match, else `fallback(rows, stage_dir)` (common.stage_files: file by file)."""
    try:
        stage_from_shards(rows, stage_dir, vol, sha, say=say, **kw)
    except ShardError as e:
        say(f"staging shards not used ({e}); staging file by file", flush=True)
        return fallback(rows, stage_dir)
    return [{**x, "audio_filepath": stage_dir + x["audio_filepath"]} for x in rows]
