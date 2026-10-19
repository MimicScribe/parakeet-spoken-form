"""Tar-shard staging (stage_shards.py): pack / unpack round trip, index verification, the file-by-file fallback and
byte-identical staged files. No network, no GPU: python -m pytest test_stage_shards.py -q

The write_mix tests need torch (ft_train imports it); they are skipped without it:
uv run --no-project --python 3.11 --with torch==2.4.1 --with "numpy<2" --with lightning --with omegaconf --with pytest \
    --with modal python -m pytest test_stage_shards.py -q
"""
import hashlib
import json
import os
import random

import pytest

import common
import stage_shards as S

SHA = "0123456789abcdef" * 4


def make_audio(tmp_path, n=40, seed=1):
    """n source files of varied sizes (some empty) under tmp/src, with odd names; rows list some of them twice."""
    r = random.Random(seed)
    src = tmp_path / "src"
    rows = []
    for i in range(n):
        d = src / f"lang{i % 3}" / ("sub" * (i % 4))
        d.mkdir(parents=True, exist_ok=True)
        name = f"clip_{i}_é ü:{i}.wav" if i % 5 == 0 else f"clip_{i}.wav"
        p = d / name
        p.write_bytes(bytes(r.randrange(256) for _ in range(r.choice([0, 1, 511, 512, 513, 4000, 20000]))))
        rows.append({"audio_filepath": str(p), "duration": 1.0 + i, "text": f"t{i}"})
    rows += [dict(rows[3]), dict(rows[7]), dict(rows[3])]  # a part listed twice repeats its paths
    random.Random(seed + 1).shuffle(rows)
    return rows


def tree(root):
    out = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            out[os.path.relpath(p, root)] = open(p, "rb").read()
    return out


def packed(tmp_path, rows, target=30000, **kw):
    vol = str(tmp_path / "vol")
    out = S.shard_dir(vol, SHA)
    idx = S.pack(S.unique_paths(rows), out, SHA, mix_dir="mixX", target=target, workers=4, say=lambda *a, **k: None, **kw)
    return vol, out, idx


def test_round_trip_every_file_byte_for_byte_in_several_shards(tmp_path):
    rows = make_audio(tmp_path)
    vol, out, idx = packed(tmp_path, rows)
    paths = S.unique_paths(rows)
    assert idx["n_files"] == len(paths) == 40
    assert len(idx["shards"]) > 3  # the small target split it
    assert sum(s["n"] for s in idx["shards"]) == 40
    assert idx["payload_bytes"] == sum(os.path.getsize(p) for p in paths)
    stage = str(tmp_path / "stage")
    assert S.stage_from_shards(rows, stage, vol, SHA, workers=3, say=lambda *a, **k: None) == 40
    for p in paths:
        assert open(stage + p, "rb").read() == open(p, "rb").read()
    assert len(tree(stage)) == 40  # nothing else, no temporary leftovers


def test_pack_is_deterministic_and_in_sorted_order(tmp_path):
    rows = make_audio(tmp_path)
    _, out1, idx1 = packed(tmp_path / "a", rows)
    _, out2, idx2 = packed(tmp_path / "b", list(reversed(rows)))  # the listing order of the manifest does not matter
    assert idx1 == idx2
    for s in idx1["shards"]:
        assert open(f"{out1}/{s['name']}", "rb").read() == open(f"{out2}/{s['name']}", "rb").read()
    import tarfile

    names = []
    for s in idx1["shards"]:
        with tarfile.open(f"{out1}/{s['name']}") as tf:
            names += tf.getnames()
    assert names == [p[1:] for p in S.unique_paths(rows)]
    assert max(s["bytes"] for s in idx1["shards"]) <= 30000 + 21000  # a shard closes before a file would pass the target


def test_pack_is_idempotent_and_repacks_a_damaged_directory(tmp_path):
    rows = make_audio(tmp_path)
    vol, out, idx = packed(tmp_path, rows)
    shard = f"{out}/{idx['shards'][0]['name']}"
    stamp = os.stat(shard).st_mtime_ns
    lines = []
    again = S.pack(S.unique_paths(rows), out, SHA, target=30000, say=lambda m, **k: lines.append(m))
    assert again == idx and os.stat(shard).st_mtime_ns == stamp and "nothing to do" in lines[0]
    with open(shard, "ab") as f:  # a shard that is not the indexed size
        f.write(b"x")
    S.pack(S.unique_paths(rows), out, SHA, target=30000, say=lambda *a, **k: None, workers=2)
    assert os.stat(shard).st_size == idx["shards"][0]["bytes"]
    S.read_index(out, SHA)


@pytest.mark.parametrize("damage", ["no_index", "other_sha", "short_shard", "missing_shard", "bad_totals", "bad_version",
                                    "not_json"])
def test_index_verification_rejects_a_directory_that_is_not_the_manifests(tmp_path, damage):
    rows = make_audio(tmp_path)
    vol, out, idx = packed(tmp_path, rows)
    S.read_index(out, SHA)  # fine as packed
    first = f"{out}/{idx['shards'][0]['name']}"
    if damage == "no_index":
        os.remove(f"{out}/{S.INDEX}")
    elif damage == "other_sha":
        with pytest.raises(S.ShardError, match="not"):
            S.read_index(out, "f" * 64)
        return
    elif damage == "short_shard":
        os.truncate(first, os.path.getsize(first) - 10)
    elif damage == "missing_shard":
        os.remove(first)
    elif damage == "bad_totals":
        idx["n_files"] += 1
        json.dump(idx, open(f"{out}/{S.INDEX}", "w"))
    elif damage == "bad_version":
        idx["version"] = 99
        json.dump(idx, open(f"{out}/{S.INDEX}", "w"))
    elif damage == "not_json":
        open(f"{out}/{S.INDEX}", "w").write("{")
    with pytest.raises(S.ShardError):
        S.read_index(out, SHA)


def fallback_spy(calls):
    def fb(rows, stage_dir):
        calls.append(len(rows))
        return common.stage_files(rows, stage_dir, workers=2, say=lambda *a, **k: None)
    return fb


def test_staged_bytes_and_rows_are_identical_with_and_without_shards(tmp_path):
    rows = make_audio(tmp_path)
    vol, out, _ = packed(tmp_path, rows)
    calls, lines = [], []
    a = S.stage_rows(rows, str(tmp_path / "from_shards"), vol, SHA, fallback_spy(calls), say=lambda m, **k: lines.append(m),
                     workers=3)
    b = common.stage_files(rows, str(tmp_path / "per_file"), workers=2, say=lambda *a, **k: None)
    assert calls == []  # the shards were used
    assert tree(str(tmp_path / "from_shards")) == {k: v for k, v in tree(str(tmp_path / "per_file")).items()}
    assert [r["audio_filepath"].replace("from_shards", "per_file") for r in a] == [r["audio_filepath"] for r in b]
    assert [{k: v for k, v in r.items() if k != "audio_filepath"} for r in a] == \
           [{k: v for k, v in r.items() if k != "audio_filepath"} for r in b]  # order and the other fields kept
    # the rate line the run watchdog reads, over the distinct files
    rate = [m for m in lines if m.startswith("staging ") and "files/s" in m]
    assert rate and rate[-1].startswith("staging 40/40 files ") and rate[-1].endswith(" files/s")


def test_missing_shards_fall_back_to_the_per_file_copy(tmp_path):
    rows = make_audio(tmp_path)
    calls, lines = [], []
    out = S.stage_rows(rows, str(tmp_path / "stage"), str(tmp_path / "vol"), SHA, fallback_spy(calls),
                       say=lambda m, **k: lines.append(m))
    assert calls == [len(rows)] and any("not used" in m for m in lines)
    assert len(tree(str(tmp_path / "stage"))) == 40
    assert all(open(r["audio_filepath"], "rb").read() == open(r["audio_filepath"][len(str(tmp_path / "stage")):], "rb").read()
               for r in out)


@pytest.mark.parametrize("damage", ["corrupt_tar", "other_path_list", "short_shard", "extra_member"])
def test_invalid_shards_fall_back_and_still_stage_every_file(tmp_path, damage):
    rows = make_audio(tmp_path)
    vol, out, idx = packed(tmp_path, rows)
    second = f"{out}/{idx['shards'][1]['name']}"
    use = rows
    if damage == "corrupt_tar":  # same size, scrambled header block
        data = bytearray(open(second, "rb").read())
        data[100:110] = b"\xff" * 10
        open(second, "wb").write(bytes(data))
    elif damage == "other_path_list":  # a manifest with one more file than the shards were packed for
        extra = tmp_path / "src" / "extra.wav"
        extra.write_bytes(b"extra")
        use = rows + [{"audio_filepath": str(extra), "duration": 1.0, "text": "x"}]
    elif damage == "short_shard":
        os.truncate(second, os.path.getsize(second) - 3)
    elif damage == "extra_member":  # an index that lies about a shard's file count
        idx["shards"][1]["n"] += 1
        idx["n_files"] += 1
        json.dump(idx, open(f"{out}/{S.INDEX}", "w"))
    calls, lines = [], []
    stage = str(tmp_path / "stage")
    res = S.stage_rows(use, stage, vol, SHA, fallback_spy(calls), say=lambda m, **k: lines.append(m), workers=2)
    assert calls == [len(use)] and any("not used" in m for m in lines)
    for r in res:
        assert open(r["audio_filepath"], "rb").read() == open(r["audio_filepath"][len(stage):], "rb").read()
    assert len({r["audio_filepath"] for r in res}) == len(S.unique_paths(use))


def test_files_already_staged_are_kept_and_a_wrong_sized_one_is_replaced(tmp_path):
    rows = make_audio(tmp_path)
    vol, out, _ = packed(tmp_path, rows)
    stage = str(tmp_path / "stage")
    paths = S.unique_paths(rows)
    ok, bad = paths[1], paths[2]
    for p, content in ((ok, open(ok, "rb").read()), (bad, b"stale")):
        os.makedirs(os.path.dirname(stage + p), exist_ok=True)
        open(stage + p, "wb").write(content)
    stamp = os.stat(stage + ok).st_mtime_ns
    S.stage_from_shards(rows, stage, vol, SHA, say=lambda *a, **k: None)
    assert os.stat(stage + ok).st_mtime_ns == stamp  # same size: left alone, as stage_files does
    assert open(stage + bad, "rb").read() == open(bad, "rb").read()


def test_manifest_sha_is_the_sha256_of_the_manifest_file_write_mix_writes(tmp_path):
    F = pytest.importorskip("ft_train")
    rows = make_audio(tmp_path)
    vol = tmp_path / "vol"
    (vol / "mix").mkdir(parents=True)
    with open(vol / "mix" / "a.jsonl", "w") as f:
        for r in rows[:25]:
            f.write(json.dumps(r) + "\n")
    with open(vol / "mix" / "b.jsonl", "w") as f:
        for r in rows[25:]:
            f.write(json.dumps({**r, "extra": 1}) + "\n")
    mix = [("mix/a.jsonl", 0), ("mix/b.jsonl", 0), ("mix/a.jsonl", 0)]
    stage = str(tmp_path / "stage")
    out = str(tmp_path / "train_manifest.jsonl")
    F.write_mix(str(vol), out, mix, seed=5, stage_dir=stage)
    written = [json.loads(l) for l in open(out)]
    unstaged = [{**x, "audio_filepath": x["audio_filepath"][len(stage):]} for x in written]
    assert S.manifest_sha([{**x, "audio_filepath": stage + x["audio_filepath"]} for x in unstaged]) == F.manifest_sha256(out)


def test_write_mix_stages_the_same_bytes_and_manifest_from_shards_or_per_file(tmp_path):
    F = pytest.importorskip("ft_train")
    rows = make_audio(tmp_path, n=30)
    vol = tmp_path / "vol"
    (vol / "mix").mkdir(parents=True)
    with open(vol / "mix" / "a.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    mix = [("mix/a.jsonl", 0), ("mix/a.jsonl", 0)]
    stage = str(tmp_path / "stage")
    per_file_out = str(tmp_path / "m_per_file.jsonl")
    F.write_mix(str(vol), per_file_out, mix, seed=2, stage_dir=stage)  # no shards on the volume yet: per file
    per_file = tree(stage)
    sha = F.manifest_sha256(per_file_out)
    unstaged = [{**json.loads(l), } for l in open(per_file_out)]
    paths = sorted({x["audio_filepath"][len(stage):] for x in unstaged})
    S.pack(paths, S.shard_dir(str(vol), sha), sha, target=25000, say=lambda *a, **k: None, workers=2)
    import shutil

    shutil.rmtree(stage)
    shards_out = str(tmp_path / "m_shards.jsonl")
    F.write_mix(str(vol), shards_out, mix, seed=2, stage_dir=stage)  # shards found by the manifest sha
    assert tree(stage) == per_file
    assert open(shards_out, "rb").read() == open(per_file_out, "rb").read()
    shutil.rmtree(stage)
    os.truncate(f"{S.shard_dir(str(vol), sha)}/shard-00000.tar", 100)  # damaged: falls back, same result
    F.write_mix(str(vol), str(tmp_path / "m_fb.jsonl"), mix, seed=2, stage_dir=stage)
    assert tree(stage) == per_file


def test_find_manifest_by_recorded_sha_and_strips_the_stage_prefix(tmp_path):
    vol = tmp_path / "vol"
    rows = [{"audio_filepath": f"/tmp/stage/vol/a/{i}.wav", "duration": 1.0, "text": str(i)} for i in range(5)]
    for run, rr in (("r1", rows[:2]), ("r2", rows)):
        d = vol / "exp" / run
        d.mkdir(parents=True)
        body = "".join(S.manifest_line(x) for x in rr)
        (d / "train_manifest.jsonl").write_text(body)
        (d / "train_manifest.sha256").write_text(hashlib.sha256(body.encode()).hexdigest())
    sha = S.manifest_sha(rows)
    path, got = S.find_manifest(str(vol), sha[:12], say=lambda *a, **k: None)
    assert path.endswith("exp/r2/train_manifest.jsonl") and len(got) == 5 and got[0]["audio_filepath"] == "/vol/a/0.wav"
    with pytest.raises(S.ShardError):
        S.find_manifest(str(vol), "0" * 12)
    with pytest.raises(S.ShardError):
        S.find_manifest(str(vol), sha[:8])
    (vol / "exp" / "r2" / "train_manifest.jsonl").write_text("changed\n")  # no longer hashes to its record
    with pytest.raises(S.ShardError):
        S.find_manifest(str(vol), sha[:12], say=lambda *a, **k: None)


def test_pack_refuses_a_relative_path_and_a_missing_file(tmp_path):
    with pytest.raises(ValueError):
        S.pack(["relative.wav"], str(tmp_path / "o"), SHA, read=lambda p: b"x", say=lambda *a, **k: None)
    with pytest.raises(OSError):
        S.pack([str(tmp_path / "nope.wav")], str(tmp_path / "o2"), SHA, say=lambda *a, **k: None)
    assert not os.path.exists(str(tmp_path / "o2" / S.INDEX))  # no index: the directory is not valid
