"""Shared lists on the volume are read fail-closed (judge_plro --upload, carve_heldout): a list that exists but cannot be
fetched stops the writer and the volume copy is left as it was. A fake `modal` CLI on PATH serves a local directory.
python3 test_volume_io.py"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent

FAKE_MODAL = r'''#!/usr/bin/env python3
import os, shutil, sys
root = os.environ["FAKE_VOL"]
fail_get = set(filter(None, os.environ.get("FAKE_FAIL_GET", "").split(",")))
a = sys.argv[1:]
assert a[0] == "volume", a
op, rest = a[1], [x for x in a[2:] if x != "--force"]
if op == "ls":
    d = os.path.join(root, rest[1])
    if not os.path.isdir(d):
        sys.exit(1)
    for n in sorted(os.listdir(d)):
        print(rest[1] + "/" + n)
elif op == "get":
    src = os.path.join(root, rest[1])
    if rest[1] in fail_get or not os.path.exists(src):
        sys.stderr.write("fetch failed\n")
        sys.exit(1)
    shutil.copyfile(src, rest[2])
elif op == "put":
    dst = os.path.join(root, rest[2])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(rest[1], dst)
'''


class FakeVolume:
    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fakevol_"))
        self.vol = self.tmp / "vol"
        self.bin = self.tmp / "bin"
        self.bin.mkdir(parents=True)
        (self.bin / "modal").write_text(FAKE_MODAL)
        (self.bin / "modal").chmod(0o755)

    def write(self, rel, text):
        p = self.vol / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def read(self, rel):
        return (self.vol / rel).read_text()

    def env(self, fail_get=""):
        return dict(os.environ, PATH=f"{self.bin}{os.pathsep}{os.environ['PATH']}", FAKE_VOL=str(self.vol),
                    FAKE_FAIL_GET=fail_get)

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class JudgeUpload(unittest.TestCase):
    def setUp(self):
        self.v = FakeVolume()
        self.root = self.v.tmp / "judge"
        w = self.root / "pl" / "p1"
        w.mkdir(parents=True)
        rows = [{"id": "a", "text": "mam siedem", "src": "tts", "spans": [{"text": "7", "kind": "int", "reading": "siedem"}]}]
        (w / "p1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (w / "ext.jsonl").write_text(json.dumps({"id": "a", "text": "mam 5"}) + "\n")  # a -> span_missing (reject)
        self.v.write("ml/p3v2/pl/reject_ids.txt", "old1\nold2\n")

    def tearDown(self):
        self.v.close()

    def judge(self, fail_get=""):
        return subprocess.run([sys.executable, str(HERE / "judge_plro.py"), "pl", "p1", "--upload"], capture_output=True,
                              text=True, env=dict(self.v.env(fail_get), SPOKEN_FORM_JUDGE_ROOT=str(self.root)))

    def test_failed_fetch_never_overwrites(self):
        # red-on-base: the failed get was read as "no list yet" and the merged list (only "a") was force-put over it
        r = self.judge(fail_get="ml/p3v2/pl/reject_ids.txt")
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(self.v.read("ml/p3v2/pl/reject_ids.txt"), "old1\nold2\n")

    def test_rejudge_takes_back_only_uploaded_rejects(self):
        # red-on-base: a local re-judge (no --upload) rewrote jr/reject_ids.txt, which the next --upload read as "this
        # pass's own earlier rejects": the id rejected and uploaded first could never be taken back
        w = self.root / "pl" / "p1"
        env = dict(self.v.env(), SPOKEN_FORM_JUDGE_ROOT=str(self.root))
        cmd = [sys.executable, str(HERE / "judge_plro.py"), "pl", "p1"]
        self.assertEqual(subprocess.run(cmd + ["--upload"], capture_output=True, env=env).returncode, 0)
        self.assertIn("a\n", self.v.read("ml/p3v2/pl/reject_ids.txt"))
        (w / "ext.jsonl").write_text(json.dumps({"id": "a", "text": "mam 7"}) + "\n")  # a is now kept
        self.assertEqual(subprocess.run(cmd, capture_output=True, env=env).returncode, 0)  # local run, no upload
        self.assertEqual(subprocess.run(cmd + ["--upload"], capture_output=True, env=env).returncode, 0)
        self.assertEqual(self.v.read("ml/p3v2/pl/reject_ids.txt"), "old1\nold2\n")

    def test_merge_when_readable(self):  # red-on-base: control
        r = self.judge()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.v.read("ml/p3v2/pl/reject_ids.txt"), "a\nold1\nold2\n")


class CarveFetch(unittest.TestCase):
    def setUp(self):
        self.v = FakeVolume()
        rows = [{"id": f"20190101-0900-PLENARY_es_{i}", "src": "vox", "spans": [{"kind": "roman"}]} for i in range(3)]
        rows += [{"id": f"201902{d}-0900-PLENARY_es_0", "src": "vox", "spans": [{"kind": "roman"}]} for d in range(10, 30)]
        self.v.write("ml/p3v2/es/rows.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
        self.v.write("ml/p3v2/es/reject_ids.txt", "")
        self.v.write("ml/p3v2/heldout_es.txt", "# list\nkeep_me\n")

    def tearDown(self):
        self.v.close()

    def carve(self, fail_get=""):
        return subprocess.run([sys.executable, str(HERE / "carve_heldout.py"), "es", "--target", "2", "--work",
                               str(self.v.tmp / "work")], capture_output=True, text=True, env=self.v.env(fail_get))

    def test_failed_heldout_fetch_is_an_error(self):
        # red-on-base: a failed fetch of an existing held-out list started a "first list" and force-put it over
        r = self.carve(fail_get="ml/p3v2/heldout_es.txt")
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(self.v.read("ml/p3v2/heldout_es.txt"), "# list\nkeep_me\n")

    def test_failed_reject_fetch_is_an_error(self):
        # red-on-base: a failed reject fetch was ignored (rejected rows could be carved into the held-out set)
        r = self.carve(fail_get="ml/p3v2/es/reject_ids.txt")
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(self.v.read("ml/p3v2/heldout_es.txt"), "# list\nkeep_me\n")

    def test_ext_ctx_rows_never_carved(self):
        # red-on-base: the carve dropped only by == "ext" rows; ext_ctx rows could enter the held-out set
        rows = [{"id": f"201903{d}-0900-PLENARY_es_0", "src": "vox", "spans": [{"kind": "code", "by": "ext_ctx"}]}
                for d in range(10, 18)]
        rows += [{"id": f"201902{d}-0900-PLENARY_es_0", "src": "vox", "spans": [{"kind": "roman"}]} for d in range(10, 30)]
        self.v.write("ml/p3v2/es/rows.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
        r = self.carve()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("vox:201903", self.v.read("ml/p3v2/heldout_es.txt"))

    def test_appends_when_readable(self):  # red-on-base: control
        r = self.carve()
        self.assertEqual(r.returncode, 0, r.stderr)
        got = self.v.read("ml/p3v2/heldout_es.txt")
        self.assertTrue(got.startswith("# list\nkeep_me\n") and "# group vox:" in got, got)


if __name__ == "__main__":
    unittest.main()
