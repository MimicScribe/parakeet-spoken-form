"""Streaming-geometry windows and seam pairs from a judged real-speech pass (2026-10-18), placing LONG number readings where
the streaming decoder's windows put them.

The decoder reads windows of 187 frames (14.96 s) every 118 frames (9.44 s; 5.52 s overlap); each next window may move up
to +-2 s to a quiet frame (overlap 3.5 - 7.5 s) and the first 6 frames (0.48 s) of every non-first window are trimmed. From the pass's rows (stitched per video, quiet-gap cuts only, never inside a reading unless the cut is a seam
pair whose halves are labelled with what each side hears) this builds:

  (a) straddle   seam pairs cut inside a long reading (ml_p3._seam_pass --min-words rule, 0.9 number-word edge rule)
  (b) head       windows whose left edge sits right before a long reading: it starts U(0, 0.8) s into the window (the
                 cold left start of the next window; the first 0.48 s of a non-first window are trimmed) -- win15.plan_edge_windows
                 "head" mode
  (c) tail       windows whose long reading ENDS U(1, 3) s before the right edge -- "tail" mode
  (d) overlap    pairs of windows that overlap by U(3.5, 5.52) s INSIDE a reading longer than the overlap: the left
                 window ends in the reading, the right window starts in it, neither holds it whole; both halves are
                 labelled by the same edge rule as every other crop (ml_balance.crop_label, 0.9 for number words) and
                 the union of the two labels is the reading

Labels stay exact: cuts only in quiet gaps between words (windows) or at the gap between two words of the reading (pairs);
every window carries `parents` and `parent_sha` (win15.window_row), every pair its parents. Outputs (new paths only):
/vol/ml/esfr_dense1/<lang>/windows.jsonl (+ win/, win_stats.json) -- the win15 layout, so a build reads the directory with
--win15-dir -- and seam_dense1.jsonl (+ seam_dense1/, seam_stats_dense1.json) in the seam_long row format.

    modal run dense_win.py::main --langs es,fr --src ml/esfr_dense1/{lang}/rows_dense1.jsonl --hours 0.6
"""
import json
import os
import random
import sys

import ml_p3_core as C
import win15 as W

VOL = "/vol"
STREAM_OVERLAP = (3.5, 5.52)  # s: the decoder's overlap (stride 118 frames +- 2 s => 3.5 .. 7.5; nominal 5.52)
HEAD_COLD = (0.0, 0.8)  # s a long reading starts after the window's left edge ("cold left start")
TAIL_LAST = (1.0, 3.0)  # s a long reading ends before the window's right edge
OVL_MIN_READING_S = 3.6  # s: a reading shorter than the smallest overlap fits whole in both windows


def usable_rows(rows: list) -> tuple:
    """Real rows with a time for every word, none decided by a second recognizer (win15.pick_sources / windows2 rule)."""
    from census_common import ext_row

    st = {"in": len(rows), "ext": 0, "no_times": 0, "synthetic": 0}
    out = []
    for r in rows:
        if r.get("synthetic") or r.get("src") == "tts":
            st["synthetic"] += 1
        elif ext_row(r):
            st["ext"] += 1
        elif not C.has_times(r):
            st["no_times"] += 1
        else:
            out.append(r)
    st["kept"] = len(out)
    return out, st


def overlap_pairs(words: list, total: float, rd: dict, rng, num_keep: float, tries: int = 6):
    """Pairs of windows that overlap by U(STREAM_OVERLAP) s inside reading `rd` (longer than the overlap): left = [e - WIN, e],
    right = [s, s + WIN] with e - s = the overlap, s and e in gaps between two words of the reading. Yields dicts shaped
    like ml_p3._long_pairs (pos "ovl"); None-labelled or non-spanning choices are skipped."""
    import ml_balance as B

    ws = [words[j] for j in rd["idx"]]
    if any(w[1] is None for w in ws) or len(ws) < 4:
        return
    r0, r1 = ws[0][1] - C.LEAD, ws[-1][2] + C.TAIL
    if r1 - r0 < OVL_MIN_READING_S:
        return
    gaps = [((a[2] + b[1]) / 2, j) for j, (a, b) in enumerate(zip(ws, ws[1:]))
            if len(C.letters(a[0])) > B.SHORT_LETTERS and len(C.letters(b[0])) > B.SHORT_LETTERS]
    reading = " ".join(w[0] for w in ws)
    made = 0
    for _ in range(tries):
        ov = rng.uniform(*STREAM_OVERLAP)
        opts = [(s, e) for s, _ in gaps for e, _ in gaps if abs((e - s) - ov) <= 0.6 and s < e]
        if not opts:
            continue
        s, e = rng.choice(opts)
        L = B._far_edge(words, max(0.0, e - C.WIN), e, "left", num_keep=num_keep, total=total)
        R = B._far_edge(words, s, min(total, s + C.WIN), "right", num_keep=num_keep, total=total)
        if L is None or R is None:
            continue
        (la, lb, ll), (ra, rb, rl) = L, R
        if lb - la < B.MIN_HALF or rb - ra < B.MIN_HALF or not ll["inside"] or not rl["inside"]:
            continue
        want = [id(w) for w in ws]
        lin = [id(w) for w in ll["inside"] if id(w) in want]
        rin = [id(w) for w in rl["inside"] if id(w) in want]
        # neither window holds the whole reading; together they hold all of it, in order, each window's part contiguous
        if len(lin) in (0, len(want)) or len(rin) in (0, len(want)) or set(lin) | set(rin) != set(want):
            continue
        if lin != want[:len(lin)] or rin != want[len(want) - len(rin):]:
            continue
        made += 1
        pos = "ovl"
        yield {"reading": reading, "kind": rd["kind"], "pos": pos, "n_words": len(rd["idx"]),
               "overlap_s": round(e - s, 2), "reading_s": round(r1 - r0, 2),
               "left": B._half_row(words, {"a": round(la, 3), "b": round(lb, 3), "lab": ll}, "left", rd, pos),
               "right": B._half_row(words, {"a": round(ra, 3), "b": round(rb, 3), "lab": rl}, "right", rd, pos)}
        if made >= 2:
            return


try:
    import modal
except ImportError:  # pure helpers only
    modal = None

if modal is not None:
    vol = modal.Volume.from_name("parakeet-spoken-form")
    image = (modal.Image.debian_slim(python_version="3.11")
             .apt_install("ffmpeg", "libsndfile1")
             .pip_install("torch==2.4.1", "torchaudio==2.4.1", "transformers==4.44.2", "soundfile", "librosa", "numpy<2")
             .add_local_python_source("ml_cands", "ml_decide", "ml_p3_core", "census_common", "label_gpu", "labeller",
                                      "readings", "en_win", "ml_slav", "ml_balance", "compound_split", "win15", "ml_p3")
             .add_local_file("ml_grammar_table.json", "/root/ml_grammar_table.json")
             .add_local_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mix_build.py"),
                             "/root/mix_build_src.py"))
    app = modal.App("parakeet-dense-win")

    def _rows(p):
        return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []

    @app.function(image=image, volumes={VOL: vol}, gpu=["L4", "A10G", "L40S"], timeout=3 * 3600, max_containers=4)
    def build_lang(lang: str, src: str, tag: str = "", hours: float = 0.6, seed: int = 1, seam_positions: int = 3,
                   seam_limit: int = 0, phases: int = 3) -> dict:
        from collections import Counter

        import librosa
        import soundfile as sf

        import ml_balance as B
        import ml_p3
        import ml_decide as D

        out = f"{VOL}/ml/esfr_dense1/{lang}"
        src_paths = [f"{VOL}/{x.format(lang=lang)}" for x in src.split(",")]  # several judged pass files of the language
        for sp_ in src_paths:
            assert os.path.exists(f"{out}/judged_{os.path.basename(sp_)[:-6]}.json"), f"{lang}: {sp_} never judged"
        win_p, seam_p = f"{out}/windows{tag}.jsonl", f"{out}/seam_dense1{tag}.jsonl"
        assert not os.path.exists(win_p) and not os.path.exists(seam_p), f"{lang}: outputs exist, use a new tag"
        os.makedirs(f"{out}/win{tag}", exist_ok=True)
        rows, st_src = usable_rows([r for sp_ in src_paths for r in _rows(sp_)])
        ctc = D.LangCTC(lang)
        cache = {}

        def read(p):
            if p not in cache:
                x, sr = sf.read(p, dtype="float32")
                x = x.mean(1) if x.ndim > 1 else x
                cache[p] = x if sr == 16000 else librosa.resample(x, orig_sr=sr, target_sr=16000)
            return cache[p]

        # ------------------------------------------------------------------ windows: head (b) and tail (c)
        W.EDGE_HEAD, W.EDGE_TAIL, W.EDGE_MODES = HEAD_COLD, TAIL_LAST, ("head", "tail")
        rng = random.Random(f"dense-win-{lang}-{seed}")
        st, kept_s, rows_out, rejects = Counter(), 0.0, [], []
        streams_all = C.group_streams(rows, rng, max_rows=W.MAX_ROWS)  # a video's rows in time order; leftover single rows in random groups (ml_p3_core)
        with open(win_p, "w", encoding="utf-8") as f:
            for ph in range(phases):
                if kept_s >= hours * 3600:
                    break
                for si, stream in enumerate(C.group_streams(rows, rng, max_rows=W.MAX_ROWS)):
                    if kept_s >= hours * 3600:
                        break
                    got = []
                    for r in stream:
                        try:
                            x = read(r["audio_filepath"])
                        except Exception:  # noqa: BLE001
                            st["row_unreadable"] += 1
                            continue
                        r, x = W.trim_row(dict(r, duration=len(x) / 16000), x)
                        got.append(dict(r, _x=x))
                    stream = got
                    if len(stream) < W.MIN_NUM_ROWS:
                        continue
                    gaps = W.stream_gaps(len(stream), rng)
                    x_all = W.stitch(stream, gaps, seed=W.zlib_seed(lang, 'dense', ph, si))
                    total = len(x_all) / 16000
                    quiet = lambda t0, t1: W.safe_quiet(x_all, t0, t1)  # noqa: E731
                    flags = {}

                    def edge_ok(i, side, stream=stream, flags=flags):
                        r = stream[i]
                        if (i, side) not in flags:
                            n = int(W.EDGE_S * 16000)
                            g = ctc.greedy(ctc.lp(r["_x"][:n] if side == "start" else r["_x"][-n:]))
                            flags[(i, side)] = W.end_close(r["text"], g, side)
                            st[f"row_{side}_bad"] += not flags[(i, side)]
                        return flags[(i, side)]

                    for j, w in enumerate(W.plan_edge_windows(stream, gaps, quiet, rng, edge_ok=edge_ok)):
                        if kept_s >= hours * 3600:
                            break
                        st["planned"] += 1
                        x = x_all[int(round(w["a"] * 16000)):int(round(min(w["b"], total) * 16000))]
                        lp = ctc.lp(x)
                        g = ctc.greedy(lp)
                        cer = C.cer(w["text"], g)
                        if cer > W.WIN_MAX_CER:
                            st["cer"] += 1
                            continue
                        if not W.edges_ok(w["text"], g):
                            st["edge_word"] += 1
                            if len(rejects) < 40:
                                rejects.append({"label": w["text"], "greedy": g})
                            continue
                        why = W.check_alignment(dict(w, src=[[stream[i]["id"], k] for i, k in w["src"]],
                                                     seg=[[stream[i]["id"], a, b] for i, a, b in w["seg"]]),
                                                ctc.word_times(lp, w["text"]))
                        if W.boundary_fail(why):
                            st["align_boundary"] += 1
                            continue
                        wid = f"w15dense{tag}_e{ph}_{si}_{j}"
                        wav = f"{out}/win{tag}/{wid}.wav"
                        assert not os.path.exists(wav), wav
                        sf.write(wav, x, 16000, subtype="PCM_16")
                        row = W.window_row(w, stream, lang, wid, wav, len(x) / 16000, g, cer)
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        rows_out.append(row)
                        kept_s += len(x) / 16000
                        st["kept"] += 1
                    if si % 20 == 0:
                        f.flush()
                        vol.commit()
        gap_hist = Counter()
        for r in rows_out:
            e, g = r["edge"], r["edge_gap_s"]
            gap_hist[f"{e}_{'0-0.4' if g < 0.4 else '0.4-0.8' if g < 0.8 else '1-2' if g < 2 else '2-3' if g < 3 else '3+'}"] += 1
        win_res = {"lang": lang, "tag": tag, "src": src, "source_rows": st_src, "streams": len(streams_all),
                   "kept_h": round(kept_s / 3600, 3), "counts": dict(st), "modes": dict(Counter(r["edge"] for r in rows_out)),
                   "position": dict(Counter(W.edge_bucket(r) for r in rows_out)), "gap_hist": dict(gap_hist),
                   "reading_words": dict(Counter(min(r["edge_words"], 12) for r in rows_out)), "edge_rejects": rejects}
        json.dump(win_res, open(f"{out}/win_stats{tag}.json", "w"), indent=1, ensure_ascii=False)
        vol.commit()

        # ------------------------------------------------------------------ seam pairs: straddle (a) + overlap (d)
        orig_long = ml_p3._long_pairs

        def both(stream, rng_, positions, num_keep, min_words):
            yield from orig_long(stream, rng_, positions, num_keep, min_words)
            _, words, total = C.stream_words(stream)
            for rd in B.readings(words, stream):
                if len(rd["idx"]) >= min_words:
                    yield from overlap_pairs(words, total, rd, rng_, num_keep)

        ml_p3._long_pairs = both
        by_vid = Counter(C.stream_key(r)[0] for r in rows)
        solo = [[r] for r in rows if by_vid[C.stream_key(r)[0]] == 1 and C.stream_key(r)[0] != "misc"]
        rng2 = random.Random(f"dense-seam-{lang}-{seed}")
        # _seam_pass writes <out>/seam<tag>.jsonl: "_dense1<tag>" keeps its own file name
        seam_res = ml_p3._seam_pass(lang, out, rows, rng2, ctc, read, cache, 8, seam_positions, seam_limit, f"_dense1{tag}",
                                    0, solo=solo, min_words=3)
        ml_p3._long_pairs = orig_long
        vol.commit()
        return {"lang": lang, "windows": win_res, "seam": seam_res}

    @app.local_entrypoint()
    def main(langs: str, src: str = "ml/esfr_dense1/{lang}/rows_dense1.jsonl", tag: str = "", hours: float = 0.6,
             seed: int = 1, seam_positions: int = 3, seam_limit: int = 0):
        for r in build_lang.starmap([(l, src, tag, hours, seed, seam_positions, seam_limit) for l in langs.split(",")]):
            print(json.dumps(r, ensure_ascii=False)[:3000])
