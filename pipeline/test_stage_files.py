"""common.stage_files: order, copies, skip of existing files, rate lines (no network): python -m pytest test_stage_files.py -q"""
import os

import common


def test_stage_files_copies_in_order_skips_existing_and_reports_the_rate(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    rows = []
    for i in range(7):
        (src / f"a{i}.wav").write_text(f"audio{i}")
        rows.append({"audio_filepath": str(src / f"a{i}.wav"), "duration": 1.0, "text": str(i)})
    stage = str(tmp_path / "stage")
    # one file is already staged: it must not be copied again
    pre = stage + rows[3]["audio_filepath"]
    os.makedirs(os.path.dirname(pre))
    open(pre, "w").write("kept")
    lines, copied = [], []
    ticks = iter(x * 0.5 for x in range(1000))
    out = common.stage_files(rows, stage, workers=1, every=3, say=lambda m, **k: lines.append(m),
                             copy=lambda a, b: (copied.append(a), open(b, "w").write(open(a).read())),
                             clock=lambda: next(ticks))
    assert [r["text"] for r in out] == [str(i) for i in range(7)]  # order kept
    assert all(r["audio_filepath"].startswith(stage) and os.path.exists(r["audio_filepath"]) for r in out)
    assert len(copied) == 6 and rows[3]["audio_filepath"] not in copied
    assert open(pre).read() == "kept"
    assert [m.split(" files ")[0] for m in lines] == ["staging 3/7", "staging 6/7", "staging 7/7"]
    assert all(m.endswith(" files/s") for m in lines)
