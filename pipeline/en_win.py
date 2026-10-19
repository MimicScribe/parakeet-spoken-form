"""English streaming-geometry windows and mirrored seam pairs (the wave-1 ml_p3.windows2_lang / seam_crops path for
English, whose real rows live in the English mix parts, not under ml/p3v2/en).

1. prep (local, CPU): English mix-part crops (text, audio, parent utterance id) -> labelled rows for ml_p3_core streams:
   spans and num_chars come from the parent row's decided labeller spans (the reading the labeller picked, found in the
   crop text); one crop per parent (two crops of one parent can overlap in audio); held-out videos / plenary days and
   rows with a digit or symbol in the label never enter.
2. run (Modal GPU): per shard, word times by CTC forced alignment with the English labeller's reader (label_gpu.W2V,
   LibriSpeech spelling via label_gpu.norm_ctc), then ml_p3_core.build_windows and ml_balance.seam_crops unchanged over
   stitched streams, each window / half re-decoded (CER <= WIN_MAX_CER and the same edge checks as wave-1). One extra
   rule on top of wave-1: a window never splits a number reading (labeller span, or a run of English number words the
   labeller did not mark); a seam pair may split only its own reading, and its two halves together hold that reading
   exactly once.
3. select (local): cap the windows to the hour target (numbers at an edge first) and the pairs to a count (cut
   positions first), whole pairs only.

    python en_win.py prep <out_rows.jsonl> --dev <dev.jsonl> --held-days <days.txt> \
        --part <name>=<crops.jsonl>@<rows3 root>[,<root>...] [--part ...] [--videos name=N] [--heldout]
    (--heldout, seam gate 2026-10-17: the inverse selection -- ONLY rows of held-out videos / days -- for eval windows
     and seam pairs; never a training part)
    modal run en_win.py --rows <rows.jsonl> --out <dir> [--tag t] [--shards 6] [--win-h 3] [--seam-pairs 900]
    python en_win.py select <dir> [--tag t] [--win-h 2.5] [--pairs 800]
"""

import json
import os
import random
import re
import sys
import zlib

import ml_p3_core as C

VOL = "/vol"
OUT_VOL = "d1/en4_ok"  # English top-up parts live here (mix4/win_ and mix4/seam_ prefixes are non-English parts)
WIN_MAX_CER = 0.15  # ml_p3.WIN_MAX_CER

# English number words for the split guard (labeller spans are the primary readings; this catches unmarked runs)
_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen " \
        "eighteen nineteen"
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety"
_ORD = "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth " \
       "fifteenth sixteenth seventeenth eighteenth nineteenth twentieth thirtieth fortieth fiftieth sixtieth " \
       "seventieth eightieth ninetieth hundredth thousandth millionth billionth"
NUM_WORDS = set(f"{_ONES} {_TENS} {_ORD} hundred thousand million billion trillion".split())
NUM_JOIN = {"and", "point", "oh", "o", "a"}  # inside a run only when number words stand on both sides


# ------------------------------------------------------------------------------------------------- 1. prep
def _reading(sp: dict) -> str:
    c = sp.get("cands") or []
    if c:
        p = sp.get("pick") or 0
        return c[p] if 0 <= p < len(c) else ""
    return sp.get("gold") or ""


def reading_offsets(text: str, spans: list) -> list:
    """[(a, b, span)] character ranges of each decided span's reading inside text (case-insensitive, whole words), in
    text order, non-overlapping. A reading searched after the previous match first, then anywhere still free."""
    out, pos = [], 0
    for sp in spans or []:
        if not sp.get("decided", True):
            continue
        rd = _reading(sp).strip()
        if not rd:
            continue
        pat = re.compile(r"(?<![\w'])" + re.escape(rd) + r"(?![\w'])", re.I)
        hit = None
        for start in (pos, 0):
            for m in pat.finditer(text, start):
                if not any(m.start() < b and a < m.end() for a, b, _ in out):
                    hit = m
                    break
            if hit:
                break
        if hit is None:
            continue
        out.append((hit.start(), hit.end(), sp))
        pos = hit.end()
    return sorted(out, key=lambda t: t[0])


def label_row(crop: dict, parent_spans: list, part: str) -> dict | None:
    """One crop -> a stream row (ml_p3_core fields), or None when the label holds a digit / symbol."""
    text = re.sub(r"\s+", " ", crop["text"]).strip()
    if not text or C.WF.search(text):
        return None
    offs = reading_offsets(text, parent_spans)
    src = "vox" if "PLENARY" in str(crop.get("src", "")) else "yodas"
    return {"id": os.path.basename(crop["audio_filepath"])[:-4], "audio_filepath": crop["audio_filepath"],
            "duration": crop["duration"], "text": text, "src": src, "part": part, "parent": crop.get("src"),
            "video": crop.get("video"), "num_chars": [[a, b] for a, b, _ in offs],
            "spans": [{"kind": sp.get("kind", "?"), "text": sp.get("text", ""), "reading": text[a:b],
                       "by": sp.get("by", "ctc")} for a, b, sp in offs]}


DAY = re.compile(r"(?:# group vox:)?(\d{8})")


def held_out(row: dict, dev_videos: set, held_days: set) -> bool:
    """A row from a held-out YouTube video or a held-out plenary day."""
    if row.get("video") in dev_videos:
        return True
    # any session type of the day (COMMITTEE / SPECIAL too: 2026-10-17, 563 of 663 held-out crops were committee
    # sessions the PLENARY-only test missed -- in training windows that would have let a held-out day in)
    m = re.match(r"(\d{8})-\d{4}-", str(row.get("parent") or row.get("id") or ""))
    return bool(m and m.group(1) in held_days)


def one_per_parent(rows: list) -> list:
    """The crop with the most readings per parent utterance (ties: the first)."""
    best = {}
    for r in rows:
        k = r.get("parent") or r["id"]
        if k not in best or len(r["num_chars"]) > len(best[k]["num_chars"]):
            best[k] = r
    keep = {id(r) for r in best.values()}
    return [r for r in rows if id(r) in keep]


def sample_videos(rows: list, n: int, seed: int) -> list:
    """Whole videos / sessions (streams need neighbours): n of them, seeded."""
    vids = sorted({r.get("video") for r in rows})
    random.Random(f"vid-{seed}").shuffle(vids)
    keep = set(vids[:n])
    return [r for r in rows if r.get("video") in keep]


def prep(argv: list) -> None:
    out = argv[0]
    opt = lambda k: argv[argv.index(k) + 1] if k in argv else None  # noqa: E731
    parts = [argv[i + 1] for i, a in enumerate(argv) if a == "--part"]
    nvid = dict(a.split("=") for i, a in enumerate(argv) if i and argv[i - 1] == "--videos")
    dev = {json.loads(l).get("video") for l in open(opt("--dev")) if l.strip()}
    held = {m.group(1) for l in open(opt("--held-days")) if (m := DAY.match(l.strip()))}
    assert dev and held, "held-out videos / days missing: refusing to build windows without them"
    ho_mode = "--heldout" in argv  # eval sets: keep only held-out rows
    rep, rows = {"dev_videos": len(dev), "held_days": len(held)}, []
    for spec in parts:
        name, rest = spec.split("=", 1)
        crops_p, roots = rest.split("@", 1)
        parents = {}
        for root in roots.split(","):
            for fn in sorted(os.listdir(f"{root}/rows3")):
                for l in open(f"{root}/rows3/{fn}"):
                    r = json.loads(l)
                    parents[r["utt_id"]] = r.get("spans") or []
        st = {"crops": 0, "no_parent": 0, "symbol": 0, "held_out": 0, "train_side": 0}
        got = []
        for l in open(crops_p):
            c = json.loads(l)
            st["crops"] += 1
            if c.get("src") not in parents:
                st["no_parent"] += 1
                continue
            r = label_row(c, parents[c["src"]], name)
            if r is None:
                st["symbol"] += 1
                continue
            if held_out(r, dev, held) != ho_mode:
                st["held_out" if not ho_mode else "train_side"] += 1
                continue
            got.append(r)
        got = one_per_parent(got)
        if name in nvid:
            got = sample_videos(got, int(nvid[name]), 1)
        st["rows"], st["h"] = len(got), round(sum(r["duration"] for r in got) / 3600, 2)
        st["readings"] = sum(len(r["num_chars"]) for r in got)
        rep[name] = st
        rows += got
    assert all(held_out(r, dev, held) == ho_mode for r in rows)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(rep, open(out[:-6] + "_report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


# ------------------------------------------------------------------------------------- 2. number-reading guard
def _norm_tok(t: str) -> str:
    return re.sub(r"[^a-z'-]", "", t.lower())


def number_groups(text: str, num_chars: list, edge_spans: list | None = None) -> list:
    """Token-index groups of one row that must not be split: each labeller reading (tokens overlapping its characters),
    each edge phrase (edge_spans: "a billion two", "fifty uh sixty", the whole multi-word phrase of an edge-case row) and
    each run of >= 2 English number words (joined by and / point / oh / a) that no reading touches. A run ends at a
    token carrying sentence or list punctuation."""
    toks = text.split(" ")
    starts, p = [], 0
    for t in toks:
        starts.append((p, p + len(t)))
        p += len(t) + 1
    groups, marked = [], set()
    for a, b in list(edge_spans or []) + list(num_chars or []):
        g = [k for k, (x, y) in enumerate(starts) if x < b and a < y]
        if g:
            groups.append(frozenset(g))
            marked |= set(g)
    parts = [[w for w in _norm_tok(t).split("-") if w] for t in toks]
    isnum = [bool(ws) and all(w in NUM_WORDS for w in ws) for ws in parts]
    run = []

    def flush():
        while run and not isnum[run[-1]]:
            run.pop()
        if len(run) >= 2 and not set(run) & marked:
            groups.append(frozenset(run))
        run.clear()

    for k, t in enumerate(toks):
        w = _norm_tok(t)
        lead_oh = w in ("oh", "o") and k + 1 < len(toks) and isnum[k + 1]  # "oh two oh one": a phone run may open on oh
        if isnum[k] or (run and w in NUM_JOIN) or lead_oh:
            run.append(k)
            if re.search(r"[,.;:!?]$", t):
                flush()
        else:
            flush()
    flush()
    return groups


def split_groups(stream: list, labelled: set, rows_i: list, groups_of=None) -> list:
    """[(row i, group)] of the given stream rows that `labelled` ({(row i, token k)}) holds only part of."""
    out = []
    for i in rows_i:
        r = stream[i]
        gs = groups_of(i) if groups_of else number_groups(r["text"], r.get("num_chars"), r.get("edge_spans"))
        for g in gs:
            n = sum((i, k) in labelled for k in g)
            if 0 < n < len(g):
                out.append((i, g))
    return out


def window_ok(stream: list, w: dict) -> bool:
    """No number reading split by the window's edges (a reading wholly in or wholly out is fine)."""
    return not split_groups(stream, {tuple(x) for x in w["src"]}, w["rows"])


def pair_ok(stream: list, p: dict) -> bool:
    """A seam pair may split exactly one group, its own reading, in both halves; together the halves hold each of that
    reading's words once. Every other group is wholly in or out of each half."""
    L = {tuple(x) for x in p["left"]["src"]}
    R = {tuple(x) for x in p["right"]["src"]}
    rows_i = sorted(set(p["left"]["rows"]) | set(p["right"]["rows"]))
    sp = set(split_groups(stream, L, rows_i)) | set(split_groups(stream, R, rows_i))
    if not sp:
        return True  # cut before / after the reading
    if len(sp) != 1:
        return False
    i, g = next(iter(sp))
    lk, rk = {k for k in g if (i, k) in L}, {k for k in g if (i, k) in R}
    toks = stream[i]["text"].split(" ")
    words = " ".join(toks[k] for k in sorted(g))
    return bool(lk and rk) and not lk & rk and lk | rk == set(g) and max(lk) < min(rk) and words == p["reading"]


# ------------------------------------------------------------------------------------------- CTC normalisation
def greedy_norm(g: str) -> str:
    """wav2vec2-960h greedy text -> the label spelling (LibriSpeech writes PER CENT, A M, O for oh)."""
    g = re.sub(r"\s+", " ", g.lower()).strip()
    g = re.sub(r"\bper cent\b", "percent", g)
    g = re.sub(r"\b([ap]) m\b", r"\1m", g)
    return re.sub(r"\bo\b", "oh", g)


def en_ids(text: str, vocab: dict, delim: int) -> list:
    """CTC ids of text in the reader's spelling (label_gpu.norm_ctc), words joined by the delimiter."""
    from label_gpu import norm_ctc

    out = []
    for w in norm_ctc(text).split():
        cs = [vocab[c] for c in w if c in vocab]
        if cs:
            if out:
                out.append(delim)
            out += cs
    return out


def window_kinds(stream: list, w: dict) -> list:
    """Labeller kinds of the readings whose words the window labels (ml_p3._window_kinds)."""
    out = set()
    for i, k in w["src"]:
        r = stream[i]
        starts, pos = [], 0
        for tok in r["text"].split(" "):
            starts.append(pos)
            pos += len(tok) + 1
        c = starts[k] if k < len(starts) else -1
        for sp, (a, b) in zip(r.get("spans") or [], r.get("num_chars") or []):
            if a <= c < b:
                out.add(sp.get("kind", "?"))
    return sorted(out)


def shard_of(row: dict, n: int) -> int:
    """Rows of one video / session stay in one shard (streams never cross shards)."""
    k = C.stream_key(row)[0]
    return zlib.crc32((k if k != "misc" else row["id"]).encode()) % n


# ----------------------------------------------------------------------------------------------- 3. select
def select_windows(win: list, hours: float, seed: int = 1) -> list:
    """<= hours of windows: a number at an edge first, then the rest; shuffled inside each group."""
    rng = random.Random(f"wsel-{seed}")
    win = list(win)
    rng.shuffle(win)
    win.sort(key=lambda w: not w.get("edge_number"))
    out, s = [], 0.0
    for w in win:
        if s + w["duration"] > hours * 3600:
            continue
        out.append(w)
        s += w["duration"]
    return out


def select_pairs(seam: list, n: int, seed: int = 1) -> list:
    """<= n whole pairs: cut positions (gap / in_word) first, round-robin over position class x kind, then before /
    after."""
    by = {}
    for h in seam:
        by.setdefault(h["pair"], []).append(h)
    pairs = [v for v in by.values() if len(v) == 2]
    rng = random.Random(f"psel-{seed}")
    rng.shuffle(pairs)
    cells = {}
    for v in pairs:
        cells.setdefault((v[0]["pos_class"] in ("before", "after"), v[0]["pos_class"], v[0]["kind"]), []).append(v)
    order = sorted(cells)
    out = []
    for cut_first in (False, True):
        keys = [k for k in order if k[0] == cut_first]
        while len(out) < n and any(cells[k] for k in keys):
            for k in keys:
                if cells[k] and len(out) < n:
                    out.append(cells[k].pop())
    return [h for v in out for h in sorted(v, key=lambda h: h["side"] != "left")]


def select(argv: list) -> None:
    d = argv[0]
    opt = lambda k, dv: argv[argv.index(k) + 1] if k in argv else dv  # noqa: E731
    tag = opt("--tag", "")
    rd = lambda p: [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []  # noqa: E731
    win = select_windows(rd(f"{d}/win_en{tag}_all.jsonl"), float(opt("--win-h", 2.5)))
    seam = select_pairs(rd(f"{d}/seam_en{tag}_all.jsonl"), int(opt("--pairs", 800)))
    for name, rows in (("win", win), ("seam", seam)):
        with open(f"{d}/{name}_en{tag}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter

    print(json.dumps({"windows": len(win), "win_h": round(sum(w["duration"] for w in win) / 3600, 2),
                      "edge_number": sum(bool(w["edge_number"]) for w in win),
                      "by_part": Counter(w["part"] for w in win),
                      "pairs": len(seam) // 2, "seam_h": round(sum(h["duration"] for h in seam) / 3600, 2),
                      "pos_class": Counter(h["pos_class"] for h in seam if h["side"] == "left"),
                      "kind": Counter(h["kind"] for h in seam if h["side"] == "left")}, indent=1))


# ------------------------------------------------------------------------------------------- 2. Modal GPU run
try:
    import modal
except ImportError:  # local prep / select / tests need no Modal
    modal = None

if modal is not None:
    vol = modal.Volume.from_name("parakeet-spoken-form")
    image = (modal.Image.debian_slim(python_version="3.11")
             .apt_install("ffmpeg", "libsndfile1")
             .pip_install("torch==2.4.1", "torchaudio==2.4.1", "transformers==4.44.2", "soundfile", "librosa", "numpy<2")
             .add_local_python_source("ml_cands", "ml_decide", "ml_p3_core", "ml_slav", "ml_balance", "census_common", "label_gpu",
                                      "labeller", "readings")
             .add_local_file("ml_grammar_table.json", "/root/ml_grammar_table.json"))
    app = modal.App("parakeet-en-win")

    @app.function(image=image, volumes={VOL: vol}, gpu="L4", timeout=2 * 3600, max_containers=6)
    def shard(rows: list, k: int, tag: str, win_s: float, seam_pairs: int, positions: int, seed: int) -> dict:
        from collections import Counter

        import librosa
        import soundfile as sf

        import ml_balance as B
        import ml_decide as D
        from label_gpu import W2V

        class EnCTC(D.LangCTC):
            def __init__(self):
                super().__init__("en", "cuda", model=W2V)

            def ids(self, text):
                return en_ids(text, self.vocab, self.delim)

            def greedy(self, lp):
                return greedy_norm(super().greedy(lp))

        ctc = EnCTC()
        st = Counter()
        cache = {}

        def read(p):
            if p not in cache:
                x, sr = sf.read(p, dtype="float32")
                x = x.mean(1) if x.ndim > 1 else x
                cache[p] = x if sr == 16000 else librosa.resample(x, orig_sr=sr, target_sr=16000)
            return cache[p]

        aligned = {}

        def align(si, stream):
            """Word times for one stream's rows, on first use (streams the limits never reach are never aligned);
            rows the reader cannot align or whose re-decode is off (CER > MAX_CER) leave the stream."""
            if si not in aligned:
                got = []
                for r in stream:
                    try:
                        x = read(r["audio_filepath"])
                    except Exception:  # noqa: BLE001
                        st["row_unreadable"] += 1
                        continue
                    lp = ctc.lp(x)
                    tm = ctc.word_times(lp, r["text"])
                    if any(t is None for t in tm):
                        st["row_unaligned"] += 1
                        continue
                    if C.cer(r["text"], ctc.greedy(lp)) > C.MAX_CER:
                        st["row_cer"] += 1
                        continue
                    # lay the stream out with the real audio length (manifest durations may differ slightly)
                    got.append(dict(r, duration=len(x) / 16000, word_times=[[round(a, 3), round(b, 3)] for a, b in tm]))
                st["rows_aligned"] += len(got)
                aligned[si] = got
            return aligned[si]

        out = f"{VOL}/{OUT_VOL}"
        os.makedirs(f"{out}/win_en{tag}", exist_ok=True)
        os.makedirs(f"{out}/seam_en{tag}", exist_ok=True)
        rng = random.Random(f"en-win-{seed}-{k}")
        streams = list(enumerate(C.group_streams(rows, rng, max_rows=8)))
        rng.shuffle(streams)
        wins, s_win = [], 0.0
        for si, stream in streams:
            if s_win >= win_s:
                break
            cache.clear()
            stream = align(si, stream)
            for j, w in enumerate(C.build_windows(stream, rng)):
                if s_win >= win_s:
                    break
                st["win_planned"] += 1
                if not window_ok(stream, w):
                    st["win_reading_split"] += 1
                    continue
                x = C.window_audio(stream, w["a"], w["b"], read)
                g = ctc.greedy(ctc.lp(x))
                if C.cer(w["text"], g) > WIN_MAX_CER:
                    st["win_cer"] += 1
                    continue
                if not C.edges_match(w["text"], g, w.get("cut_left"), w.get("cut_right")):
                    st["win_edge_word"] += 1
                    continue
                wid = f"w{k}_{si}_{j}"
                wav = f"{out}/win_en{tag}/{wid}.wav"
                sf.write(wav, x, 16000, subtype="PCM_16")
                wins.append({"audio_filepath": wav, "duration": round(len(x) / 16000, 3), "text": w["text"], "lang": "en",
                             "id": f"en_{wid}{tag}", "origin": "window", "num_tokens": w["num_tokens"],
                             "cut_words": w["cut_words"], "kept_cut_words": w["kept_cut_words"],
                             "edge_fracs": w["edge_fracs"], "cut_number": w["cut_number"],
                             "edge_number": w["edge_number"], "parents": [stream[i]["id"] for i in w["rows"]],
                             "src": [[stream[i]["id"], kk] for i, kk in w["src"]], "kinds": window_kinds(stream, w),
                             "part": stream[w["rows"][0]]["part"], "greedy": g, "synthetic": False})
                s_win += len(x) / 16000
        st["win_kept"], st["win_h"] = len(wins), round(s_win / 3600, 3)
        rng.shuffle(streams)
        halves, n_pairs = [], 0
        for si, stream in streams:
            if n_pairs >= seam_pairs:
                break
            cache.clear()
            stream = align(si, stream)
            for j, p in enumerate(B.seam_crops(stream, rng, positions)):
                if n_pairs >= seam_pairs:
                    break
                pc = B.pos_class(p["pos"], p["n_words"])
                st["pair_planned"] += 1
                if not pair_ok(stream, p):
                    st["pair_reading_split"] += 1
                    continue
                got, good = [], True
                for h in (p["left"], p["right"]):
                    x = C.window_audio(stream, h["a"], h["b"], read)
                    g = ctc.greedy(ctc.lp(x))
                    if C.cer(h["text"], g) > WIN_MAX_CER:
                        st[f"pair_{h['side']}_cer"] += 1
                        good = False
                        break
                    if not B.seam_edges_ok(h, g):
                        st[f"pair_{h['side']}_edge_word"] += 1
                        good = False
                        break
                    got.append((h, x, g))
                if not good:
                    continue
                pid = f"s{k}_{si}_{j}"
                for h, x, g in got:
                    hid = f"{pid}_{h['side'][0]}"
                    wav = f"{out}/seam_en{tag}/{hid}.wav"
                    sf.write(wav, x, 16000, subtype="PCM_16")
                    halves.append({"audio_filepath": wav, "duration": round(len(x) / 16000, 3), "text": h["text"],
                                   "lang": "en", "id": f"en_seam{tag}_{hid}", "origin": "seam", "pair": pid,
                                   "side": h["side"], "pos": p["pos"], "pos_class": pc, "kind": p["kind"],
                                   "reading": p["reading"], "reading_part": h["reading_part"],
                                   "num_tokens": h["num_tokens"], "cut_words": h["cut_words"],
                                   "kept_cut_words": h["kept_cut_words"], "edge_fracs": h["edge_fracs"],
                                   "cut_number": True, "edge_number": True, "greedy": g,
                                   "parents": [stream[i]["id"] for i in h["rows"]],
                                   "src": [[stream[i]["id"], kk] for i, kk in h["src"]],
                                   "part": stream[h["rows"][0]]["part"], "synthetic": False})
                n_pairs += 1
                st[f"pair_kept_{pc}"] += 1
        st["pairs_kept"] = n_pairs
        vol.commit()
        return {"stats": dict(st), "win": wins, "seam": halves}

    @app.local_entrypoint()
    def run(rows: str, out: str, tag: str = "", shards: int = 6, win_h: float = 3.0, seam_pairs: int = 900,
            positions: int = 2, seed: int = 1, limit_rows: int = 0):
        from collections import Counter

        rs = [json.loads(l) for l in open(rows) if l.strip()]
        if limit_rows:
            rs = rs[:limit_rows]
        parts = [[] for _ in range(shards)]
        for r in rs:
            parts[shard_of(r, shards)].append(r)
        args = [(p, k, tag, win_h * 3600 / shards, -(-seam_pairs // shards), positions, seed) for k, p in enumerate(parts)]
        os.makedirs(out, exist_ok=True)
        tot, win, seam = Counter(), [], []
        for res in shard.starmap(args):
            tot.update(res["stats"])
            win += res["win"]
            seam += res["seam"]
        for name, xs in (("win", win), ("seam", seam)):
            with open(f"{out}/{name}_en{tag}_all.jsonl", "w", encoding="utf-8") as f:
                for x in xs:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
        st = dict(tot)
        st["win_h"] = round(sum(w["duration"] for w in win) / 3600, 3)
        json.dump(st, open(f"{out}/stats_en{tag}.json", "w"), indent=1)
        print(json.dumps(st, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "prep":
        prep(sys.argv[2:])
    elif cmd == "select":
        select(sys.argv[2:])
    else:
        print(__doc__)
        sys.exit(2)
