"""Streaming-geometry adversarial windows for the promoted real rows (realx1), cut where the app cuts.

The app decodes 14.96 s windows (ml_p3_core.WIN) every 9.44 s (STRIDE), so consecutive windows overlap by 5.52 s; the
aligner moves the next window up to +-2 s to a quiet frame (overlap 3.5 - 7.5 s); the first 6 encoder frames (0.48 s) of a
non-first window are trimmed. A number the stream carries lands wherever the cuts fall. For every long number reading
(>= LONG_MIN_WORDS words) of a promoted row this planner places windows of the stream so the number sits:

  straddle_end    the window's right edge cuts through the reading (quiet gap between two of its words);
  straddle_start  the window's left edge cuts through the reading: a cold start inside the number;
  head            the reading starts 0 - 1.5 s after the left edge (the previous window ended just before it);
  tail            the reading ends 1 - 3 s before the right edge;
  overlap_pair    two consecutive windows: k ends inside the reading, k+1 starts OVERLAP s earlier (3.5 - 7.5 s). When
                  the reading is longer than that overlap, k+1 also starts inside it (span_overlap): neither window
                  holds the number whole.

All edges are cuts between two words (a letter-frame gap >= MIN_GAP and a quiet 10 ms frame; inside a number, where the
words run together and no quiet frame exists, the middle of a letter gap >= MID_GAP, as the seam pairs cut), so every
labelled word is whole and every heard word is labelled (the re-decode check below refuses what is not); the trimmed start of a non-first window is the cut itself (the overlap is
measured to the trimmed start). Windows are stitched from rows of one session / video exactly as win15 does and checked
the same way (re-decode CER, first / last word, forced alignment against the source rows); the manifest rows are
win15.window_row rows (parents, parent_sha, seg) plus geom / pair / cut_in / overlap_s. A build reads them as another
win15 directory (build4 win15_dir "ml/win15b,ml/win15sym,ml/realx1w").

    modal run realx1_geom.py::run --lang es --tag a --hours 2.0
"""

import json
import os
import random

import ml_balance as B
import win15 as W

C = W.C
MODES = ("straddle_end", "straddle_start", "head", "tail", "overlap_pair")
OVERLAP = (3.5, 7.5)  # s: app overlap, 14.96 - 9.44 = 5.52 moved by +-2 s
TAIL = (1.0, 3.0)
HEAD = (0.0, 1.5)
LONG = W.LONG_MIN_WORDS
MID_GAP = 0.10  # s: a cut inside a reading with no quiet frame needs this much between the letter frames of its words (ml_p3_core.MIN_GAP)
TRIM_S = 6 * 0.08  # the 6 trimmed frames of a non-first window: folded into the cut (documented, not re-applied)


def edges_ok_cut(label: str, greedy: str, drop_left, drop_right, cut_in: list) -> bool:
    """win15.edges_ok, except at an edge that cuts through a number: the re-decode may begin with the tail of the word
    the cut dropped on the left (end with the head of the one dropped on the right), as the seam pairs' edge check allows
    (ml_balance.seam_edges_ok); the labelled edge word itself must still be heard whole."""
    lw = [C.letters(w) for w in label.split() if C.letters(w)]
    g = C.letters(greedy)
    if not lw or not g:
        return False
    dl, dr = C.letters(drop_left or ""), C.letters(drop_right or "")
    start = g.startswith(lw[0]) or (cut_in[0] and any(g.startswith(dl[-k:] + lw[0]) for k in range(1, len(dl))))
    end = g.endswith(lw[-1]) or (cut_in[1] and any(g.endswith(lw[-1] + dr[:k]) for k in range(1, len(dr))))
    return bool(start and end)


def cut_list(words: list, total: float, quiet) -> list:
    """[(time, j, key)]: a quiet cut between words[j-1] and words[j]; key = the reading both belong to (a cut inside a
    reading) else None. The stream's start and end are cuts."""
    out = [(0.0, 0, None)]
    for j in range(1, len(words)):
        p, n = words[j - 1], words[j]
        if n[1] - p[2] < C.MIN_GAP:
            continue
        inside = p[3] is not None and p[3] == n[3]
        c = quiet(p[2], n[1])
        if c is None:
            if not (inside and n[1] - p[2] >= MID_GAP):
                continue
            c = (p[2] + n[1]) / 2  # inside a number, words run together: the middle of a clear letter gap (seam rule)
        out.append((round(c, 3), j, p[3] if inside else None))
    out.append((round(total, 3), len(words), None))
    return out


def long_pairs(stream: list, rng, promoted: set, min_words: int = LONG, positions: int = 3):
    """Mirrored seam pairs (ml_balance.seam_pair) cut inside the long readings of promoted rows, at the gaps between their
    words (ml_p3._long_pairs with the 0.9 number-word edge rule: no cut through a word)."""
    import ml_balance as B

    _, words, total = C.stream_words(stream)
    for rd in B.readings(words, stream):
        if rd["row"] not in promoted or len(rd["idx"]) < min_words:
            continue
        cands = [(pp, c) for pp, c in B.cut_positions(words, rd) if not pp.startswith("in")]
        if len(cands) > positions:
            cands = rng.sample(cands, positions)
        reading = " ".join(words[j][0] for j in rd["idx"])
        for pos, c in cands:
            pr = B.seam_pair(words, total, c, C.WIN, C.NUM_KEEP_FRAC)
            if pr is None or not B.pair_partitions(words, pr, rd):
                continue
            yield {"reading": reading, "kind": rd["kind"], "pos": pos, "n_words": len(rd["idx"]),
                   "left": B._half_row(words, pr["left"], "left", rd, pos),
                   "right": B._half_row(words, pr["right"], "right", rd, pos)}


def targets(words: list, promoted: set, min_words: int = LONG) -> list:
    """[(key, t0, t1, n words)] of the long readings of promoted rows (row index in `promoted`)."""
    by = {}
    for x in words:
        if x[3] is not None and x[3][0] in promoted:
            by.setdefault(x[3], []).append(x)
    return [(k, xs[0][1], xs[-1][2], len(xs)) for k, xs in by.items() if len(xs) >= min_words]


def _window(words, cuts, ka, kb):
    (a, j0, _), (b, j1, _) = cuts[ka], cuts[kb]
    return {"a": a, "b": b, "j0": j0, "j1": j1} if j1 > j0 else None


def plan_geom(rows: list, gaps: list, quiet, rng, promoted: set, per_target: int = 3, win: float = C.WIN,
              min_len: float = W.MIN_LEN, edge_ok=None) -> list:
    """Windows of one stream that place each long reading of a promoted row at the geometry positions (module doc).
    Returns window dicts as win15.plan_edge_windows plus geom / pair / cut_in / overlap_s."""
    starts, words, total = W.layout(rows, gaps)
    cuts = cut_list(words, total, quiet)
    out, seen = [], set()

    def far_left(b, n):  # cuts to the left of b, win-min_len away, not inside a reading
        return [k for k, (c, _, key) in enumerate(cuts) if key is None and min_len <= b - c <= win]

    def far_right(a):
        return [k for k, (c, _, key) in enumerate(cuts) if key is None and min_len <= c - a <= win]

    def emit(w, mode, key, t0, t1, pair=None, overlap=None):
        if w is None or (w["j0"], w["j1"]) in seen:
            return False
        ws = words[w["j0"]:w["j1"]]
        if not ws or not any(x[3] == key for x in ws):
            return False
        if not W.ends_ok(rows, words, w, edge_ok):
            return False
        rws = sorted({x[4] for x in ws})
        a, b = w["a"], w["b"]
        if any(min(starts[i] + rows[i]["duration"], b) - max(starts[i], a) < W.MIN_PIECE_S for i in rws):
            return False
        seen.add((w["j0"], w["j1"]))
        cut_in = [bool(w["j0"] > 0 and words[w["j0"] - 1][3] == key and words[w["j0"]][3] == key),
                  bool(w["j1"] < len(words) and words[w["j1"] - 1][3] == key and words[w["j1"]][3] == key)]
        n_rows = len({x[4] for x in ws if x[3] is not None})
        out.append(dict(w, text=" ".join(x[0] for x in ws), num_tokens=[x[3] is not None for x in ws],
                        word_times=[[round(x[1] - a, 3), round(x[2] - a, 3)] for x in ws],
                        src=[[x[4], x[5]] for x in ws], rows=rws, n_num_rows=n_rows,
                        cut_left=bool(w["j0"] > 0 and words[w["j0"] - 1][4] == words[w["j0"]][4]),
                        cut_right=bool(w["j1"] < len(words) and words[w["j1"]][4] == words[w["j1"] - 1][4]),
                        seg=[[i, round(max(starts[i], a) - a, 3), round(min(starts[i] + rows[i]["duration"], b) - a, 3)]
                             for i in rws],
                        edge=mode, edge_pos_s=[round(t0 - a, 3), round(b - t1, 3)], geom=mode, pair=pair,
                        cut_in=cut_in, overlap_s=overlap, edge_gap_s=round(max(t0 - a, b - t1), 3),
                        drop_left=words[w["j0"] - 1][0] if w["j0"] > 0 else None,
                        drop_right=words[w["j1"]][0] if w["j1"] < len(words) else None,
                        edge_words=sum(1 for x in words if x[3] == key)))
        return True

    for key, t0, t1, nw in targets(words, promoted):
        inner = [k for k, (c, j, kk) in enumerate(cuts) if kk == key]  # cuts inside this reading
        modes = [m for m in MODES if m not in ("straddle_end", "straddle_start", "overlap_pair") or inner]
        rng.shuffle(modes)
        done = 0
        for mode in modes * 2:  # a second round with new random offsets when a mode found no window
            if done >= per_target:
                break
            if mode == "straddle_end":
                kb = rng.choice(inner)
                far = far_left(cuts[kb][0], 0)
                if far:
                    done += emit(_window(words, cuts, rng.choice(far), kb), mode, key, t0, t1)
            elif mode == "straddle_start":
                ka = rng.choice(inner)
                far = far_right(cuts[ka][0])
                if far:
                    done += emit(_window(words, cuts, ka, rng.choice(far)), mode, key, t0, t1)
            elif mode == "head":
                want = t0 - rng.uniform(*HEAD)
                near = [k for k, (c, _, kk) in enumerate(cuts) if kk is None and t0 - HEAD[1] - 0.6 <= c <= t0 + 1e-6]
                if near:
                    ka = min(near, key=lambda k: abs(cuts[k][0] - want))
                    far = far_right(cuts[ka][0])
                    if far:
                        done += emit(_window(words, cuts, ka, rng.choice(far)), mode, key, t0, t1)
            elif mode == "tail":
                want = t1 + rng.uniform(*TAIL)
                near = [k for k, (c, _, kk) in enumerate(cuts) if kk is None and t1 + TAIL[0] - 0.4 <= c <= t1 + TAIL[1] + 0.4]
                if near:
                    kb = min(near, key=lambda k: abs(cuts[k][0] - want))
                    far = far_left(cuts[kb][0], 0)
                    if far:
                        done += emit(_window(words, cuts, rng.choice(far), kb), mode, key, t0, t1)
            else:  # overlap_pair
                kb = rng.choice(inner)
                c1 = cuts[kb][0]
                ov = rng.uniform(*OVERLAP)
                near2 = [k for k, (c, _, kk) in enumerate(cuts) if OVERLAP[0] - 0.4 <= c1 - c <= OVERLAP[1] + 0.4
                         and (kk is None or kk == key)]
                far = far_left(c1, 0)
                if not near2 or not far:
                    continue
                k2 = min(near2, key=lambda k: abs((c1 - cuts[k][0]) - ov))
                far2 = far_right(cuts[k2][0])
                if not far2:
                    continue
                w1 = _window(words, cuts, rng.choice(far), kb)
                w2 = _window(words, cuts, k2, rng.choice(far2))
                if w1 is None or w2 is None:
                    continue
                real_ov = round(c1 - cuts[k2][0], 2)
                pid = f"p{len(out)}"
                n0 = len(out)
                ok1 = emit(w1, "overlap_pair", key, t0, t1, pair=pid, overlap=real_ov)
                ok2 = emit(w2, "overlap_pair", key, t0, t1, pair=pid, overlap=real_ov)
                if ok2 and cuts[k2][2] == key:
                    out[-1]["geom"] = "span_overlap"  # k+1 starts inside the reading too: the reading is longer than the overlap
                if ok1 != ok2:  # keep a pair whole
                    del out[n0:]
                    for w_ in (w1, w2):
                        seen.discard((w_["j0"], w_["j1"]))
                else:
                    done += ok1
    return out


try:
    import modal
except ImportError:  # local tests need no Modal
    modal = None

if modal is not None:
    VOL = "/vol"
    vol = modal.Volume.from_name("parakeet-spoken-form")
    image = (modal.Image.debian_slim(python_version="3.11")
             .apt_install("ffmpeg", "libsndfile1")
             .pip_install("torch==2.4.1", "torchaudio==2.4.1", "transformers==4.44.2", "soundfile", "librosa", "numpy<2")
             .add_local_python_source("ml_cands", "ml_decide", "ml_p3_core", "census_common", "label_gpu", "labeller",
                                      "readings", "en_win", "ml_slav", "ml_balance", "compound_split", "win15")
             .add_local_file("ml_grammar_table.json", "/root/ml_grammar_table.json")
             .add_local_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mix_build.py"),
                             "/root/mix_build_src.py"))
    app = modal.App("parakeet-realx1-geom")

    def _rows(p):
        return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []

    @app.function(image=image, volumes={VOL: vol}, gpu="L4", timeout=3 * 3600, max_containers=6)
    def build_lang(lang: str, tag: str, rows_path: str, ctx_path: str, hours: float, seed: int = 1,
                   seam_hours: float = 1.0) -> dict:
        from collections import Counter

        import librosa
        import soundfile as sf

        vol.reload()
        out = f"{VOL}/ml/realx1w{tag}/{lang}"
        assert not os.path.exists(f"{out}/windows.jsonl"), f"{out}/windows.jsonl exists: use a new tag"
        os.makedirs(f"{out}/wav", exist_ok=True)
        rng = random.Random(f"realx1w-{lang}-{seed}")
        st = Counter()
        ctc = W._ctc(lang)
        prom = [dict(r, _promoted=True) for r in _rows(rows_path) if C.has_times(r)]
        ids = {r["id"] for r in prom}
        ctx = [dict(r, _promoted=False) for r in _rows(ctx_path) if r["id"] not in ids and C.has_times(r)]
        st["promoted_rows"], st["context_rows"] = len(prom), len(ctx)

        def read(p):
            x, sr = sf.read(p, dtype="float32")
            x = x.mean(1) if x.ndim > 1 else x
            return x if sr == 16000 else librosa.resample(x, orig_sr=sr, target_sr=16000)

        rows_out, kept_s, rejects = [], 0.0, []
        os.makedirs(f"{out}/seam", exist_ok=True)
        sf_ = open(f"{out}/seam.jsonl", "w", encoding="utf-8")
        seam_s, seam_pos = 0.0, Counter()
        f = open(f"{out}/windows.jsonl", "w", encoding="utf-8")
        for ph in range(W.PHASES + 1):
            if kept_s >= hours * 3600:
                break
            streams = W.group(prom + ctx, rng, max_rows=8)
            streams = [s for s in streams if any(r["_promoted"] for r in s)]
            rng.shuffle(streams)
            st["streams"] = len(streams)
            for si, stream in enumerate(streams):
                if kept_s >= hours * 3600:
                    break
                got = []
                for r in stream:
                    try:
                        x = read(r["audio_filepath"])
                    except Exception:  # noqa: BLE001
                        st["row_unreadable"] += 1
                        continue
                    r2, x = W.trim_row(dict(r, duration=len(x) / 16000), x)
                    got.append(dict(r2, _x=x))
                stream = got
                if len(stream) < 2:
                    continue
                gaps = W.stream_gaps(len(stream), rng)
                x_all = W.stitch(stream, gaps, seed=W.zlib_seed(lang, "geom", ph, si))
                total = len(x_all) / 16000
                quiet = lambda t0, t1: W.safe_quiet(x_all, t0, t1)  # noqa: E731
                promoted = {i for i, r in enumerate(stream) if r["_promoted"]}
                flags = {}

                def edge_ok(i, side, stream=stream, flags=flags):
                    r = stream[i]
                    if (i, side) not in flags:
                        n = int(W.EDGE_S * 16000)
                        g = ctc.greedy(ctc.lp(r["_x"][:n] if side == "start" else r["_x"][-n:]))
                        flags[(i, side)] = W.end_close(r["text"], g, side)
                        st[f"row_{side}_bad"] += not flags[(i, side)]
                    return flags[(i, side)]

                planned = plan_geom(stream, gaps, quiet, rng, promoted, edge_ok=edge_ok)
                for j, w in enumerate(planned):
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
                    if not edges_ok_cut(w["text"], g, w.get("drop_left"), w.get("drop_right"), w["cut_in"]):
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
                    wid = f"rxw{tag}_{ph}_{si}_{j}"
                    wav = f"{out}/wav/{wid}.wav"
                    assert not os.path.exists(wav), wav
                    sf.write(wav, x, 16000, subtype="PCM_16")
                    row = W.window_row(w, stream, lang, wid, wav, len(x) / 16000, g, cer)
                    for k in ("geom", "pair", "cut_in", "overlap_s"):
                        row[k] = w[k]
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows_out.append(row)
                    kept_s += len(x) / 16000
                    st["kept"] += 1
                    st["kept_" + w["geom"]] += 1
                if seam_s < seam_hours * 3600 and ph == 0:
                    paths = {r["audio_filepath"]: r["_x"] for r in stream}
                    try:
                        pairs = long_pairs(stream, rng, promoted)
                    except Exception as e:  # noqa: BLE001
                        st["seam_stream_error"] += 1
                        pairs = []
                    for k, p in enumerate(pairs):
                        if seam_s >= seam_hours * 3600:
                            break
                        st["seam_planned"] += 1
                        halves, ok = [], True
                        for h in (p["left"], p["right"]):
                            x = C.window_audio(stream, h["a"], h["b"], lambda q: paths[q])
                            g = ctc.greedy(ctc.lp(x))
                            if C.cer(h["text"], g) > W.WIN_MAX_CER:
                                st[f"seam_{h['side']}_cer"] += 1
                                ok = False
                            elif not B.seam_edges_ok(h, g):
                                st[f"seam_{h['side']}_edge_word"] += 1
                                ok = False
                            halves.append((h, x, g))
                        if not ok:
                            continue
                        pid = f"rs{si}_{k}"
                        for h, x, g in halves:
                            hid = f"{pid}_{h['side'][0]}"
                            wav = f"{out}/seam/{hid}.wav"
                            sf.write(wav, x, 16000, subtype="PCM_16")
                            par = [stream[i] for i in h["rows"]]
                            sf_.write(json.dumps({
                                "audio_filepath": wav, "duration": round(len(x) / 16000, 3), "text": h["text"], "lang": lang,
                                "id": f"{lang}_seamrx{tag}_{hid}", "origin": "seam", "pair": pid, "side": h["side"],
                                "pos": p["pos"], "pos_class": B.pos_class(p["pos"], p["n_words"]), "kind": p["kind"],
                                "reading": p["reading"], "reading_part": h["reading_part"], "num_tokens": h["num_tokens"],
                                "cut_words": h["cut_words"], "kept_cut_words": h["kept_cut_words"],
                                "edge_fracs": h["edge_fracs"], "cut_number": True, "edge_number": True, "greedy": g,
                                "parents": [r["id"] for r in par], "parent_sha": {r["id"]: W.sha(r["text"]) for r in par},
                                "src": [[stream[i]["id"], kk] for i, kk in h["src"]], "synthetic": False,
                                "n_words": p["n_words"]}, ensure_ascii=False) + "\n")
                            seam_s += len(x) / 16000
                        st["seam_kept_pairs"] += 1
                        seam_pos[B.pos_class(p["pos"], p["n_words"])] += 1
                if si % 40 == 0:
                    f.flush()
                    vol.commit()
        f.close()
        sf_.close()
        res = {"lang": lang, "tag": tag, "hours_target": hours, "kept_h": round(kept_s / 3600, 3), "counts": dict(st),
               "seam_h": round(seam_s / 3600, 3), "seam_positions": dict(seam_pos),
               "geom": dict(Counter(r["geom"] for r in rows_out)),
               "position": dict(Counter(W.edge_bucket(r) for r in rows_out)),
               "cut_inside": {"left": sum(r["cut_in"][0] for r in rows_out), "right": sum(r["cut_in"][1] for r in rows_out)},
               "overlap_s": sorted({r["overlap_s"] for r in rows_out if r["overlap_s"]})[:200],
               "edge_rejects": rejects}
        json.dump(res, open(f"{out}/stats.json", "w"), indent=1, ensure_ascii=False)
        vol.commit()
        return res

    @app.function(image=image, volumes={VOL: vol}, gpu="L4", timeout=3600, max_containers=6)
    def check_windows(lang: str, tag: str, n: int, seed: int = 7) -> dict:
        """win15.check_lang on this module's windows (ml/realx1w<tag>/<lang>): a random sample of n windows -- each source
        row's stretch re-decoded alone must give its labelled words (piece_ok), the forced alignment must keep every word in
        its own source row (boundary) and near where the source puts it (moved); the two wrong-label controls
        (win15.shift_control, swap_control) must be caught."""
        import librosa
        import soundfile as sf

        vol.reload()
        d = f"{VOL}/ml/realx1w{tag}/{lang}"
        dest = f"{d}/check_{n}_{seed}.json"
        assert not os.path.exists(dest), dest
        ws = _rows(f"{d}/windows.jsonl")
        random.Random(f"chk-{lang}-{seed}").shuffle(ws)
        ws = ws[:n]
        ctc = W._ctc(lang)

        def read(p):
            x, sr = sf.read(p, dtype="float32")
            x = x.mean(1) if x.ndim > 1 else x
            return x if sr == 16000 else librosa.resample(x, orig_sr=sr, target_sr=16000)

        res = {"lang": lang, "n": len(ws), "piece_fail": 0, "pieces": 0, "pieces_bad": 0, "align_boundary_fail": 0,
               "align_moved": 0, "control_n": 0, "control_piece_caught": 0, "control_align_caught": 0, "fails": []}
        for w in ws:
            x = read(w["audio_filepath"])
            lp = ctc.lp(x)
            why = W.check_alignment(w, ctc.word_times(lp, w["text"]))
            res["align_boundary_fail"] += W.boundary_fail(why)
            res["align_moved"] += any(r.startswith("moved") for r in why)
            bad = []
            for pid, a, b, lab in W.pieces(w):
                g = ctc.greedy(ctc.lp(x[int(a * 16000):int(b * 16000)]))
                res["pieces"] += 1
                if not W.piece_ok(lab, g):
                    res["pieces_bad"] += 1
                    bad.append({"parent": pid, "label": lab, "greedy": g})
            if bad:
                res["piece_fail"] += 1
                res["fails"].append({"id": w["id"], "geom": w.get("geom"), "pieces": bad[:3], "align": why[:4]})
            cw, sw = W.swap_control(w), W.shift_control(w)
            if cw is not None and sw is not None:
                res["control_n"] += 1
                res["control_align_caught"] += W.boundary_fail(W.check_alignment(cw, ctc.word_times(lp, cw["text"])))
                res["control_piece_caught"] += any(
                    not W.piece_ok(lab, ctc.greedy(ctc.lp(x[int(a * 16000):int(b * 16000)]))) for _, a, b, lab in W.pieces(sw))
        for k in ("piece_fail", "align_boundary_fail", "align_moved"):
            res[f"{k}_rate"] = round(res[k] / max(res["n"], 1), 4)
        res["piece_bad_rate"] = round(res["pieces_bad"] / max(res["pieces"], 1), 4)
        for k in ("control_piece_caught", "control_align_caught"):
            res[f"{k}_rate"] = round(res[k] / max(res["control_n"], 1), 4)
        json.dump(res, open(dest, "w"), indent=1, ensure_ascii=False)
        vol.commit()
        return res

    @app.local_entrypoint()
    def check(langs: str = "es,fr,de,pt,it,nl", tag: str = "b", n: int = 150):
        for r in check_windows.starmap([(l, tag, n) for l in langs.split(",")]):
            print(json.dumps({k: v for k, v in r.items() if k != "fails"}, ensure_ascii=False))

    @app.local_entrypoint()
    def run(lang: str, tag: str = "a", hours: float = 2.0, seed: int = 1, rows: str = "rows_realx1", context: str = "ctx",
            seam_hours: float = 1.0):
        """rows: the promoted rows file under ml/realx1/<lang>/; context: context rows (the same sessions) under _in/<lang>/."""
        r = build_lang.remote(lang, tag, f"{VOL}/ml/realx1/{lang}/{rows}.jsonl", f"{VOL}/ml/realx1/_in/{lang}/{context}.jsonl",
                              hours, seed, seam_hours)
        print(json.dumps({k: r[k] for k in ("lang", "kept_h", "seam_h", "seam_positions", "counts", "geom", "position", "cut_inside")},
                         ensure_ascii=False))
