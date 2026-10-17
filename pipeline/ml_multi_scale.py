"""Scale-up of approach A-ctc (ml_multi_ctc) for Polish and Romanian parliament speech (2026-10-16), cost-reduced:

  select   variety-first selection from the Granary-VoxPopuli text index (ml/multi/granvox/<lang>/index.jsonl, which
           already leaves out every VoxPopuli transcribed dev/test day): year VALUES stratified, class quotas, <= 3
           clips per masked-digit template, <= 8 per session, from the 3-4 debate years that fill the strata best
           (each language-year streams two session archives, so fewer years = fewer downloads)       (local, `select`)
  cut      the selected clips of one language-year out of both archives; resumable (a dropped stream retries and
           skips the sessions already cut; a finished language-year is saved and skipped on rerun)   (Modal CPU)
  score    CTC reader: every value-checked candidate of every span (ml_multi_ctc.expand_cands) scored in the clip,
           the decision (margin over rivals, fit to the greedy transcript), the word times of the decided label,
           and the number-bearing PIECES cut as FLAC for the external transcript                     (Modal)
  assemble the external transcript of each piece must give the caption's value at every span (digits) and agree
           with the piece label; a clip whose every span passes is kept whole, else its passing pieces are kept as
           rows of their own; rows in the label2 format -> ml/p3v2/<lang>/rows_<tag>.jsonl            (Modal CPU)

    python ml_multi_scale.py select --lang pl --index idx/pl/index.jsonl --out sel_pl.json
    modal run ml_multi_scale.py::cut --lang pl --sel sel_pl.json --tag multi1
    modal run ml_multi_scale.py::score --lang pl --items <volume path of items jsonl> --tag multi1
    modal run ml_multi_scale.py::assemble --lang pl --tag multi1 --ext <volume path of ext jsonl>
"""
import collections
import json
import os
import random
import re
import sys

try:
    import modal
except ImportError:
    modal = None

VOL = "/vol"
P3 = f"{VOL}/ml/p3v2"
HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ selection (pure, local)
CUR_RO = r"lei\b|leu\b|ron\b|euro\w*|eur\b|dolar\w*|€|\$"
DATE_WORD = {"pl": r"stycz\w*|lut\w*|mar\w*|kwie\w*|maj\w*|czerw\w*|lip\w*|sierp\w*|wrze\w*|paźdz\w*|listopad\w*|grud\w*",
             "ro": r"ianuarie|februarie|martie|aprilie|mai|iunie|iulie|august|septembrie|octombrie|noiembrie|decembrie"}
QUOTAS = {"clock": 400, "cur": 500, "dec": 350, "pct": 300, "big": 300, "range": 250, "ord": 150, "date": 250,
          "phone": 60}


def classes(text: str, lang: str) -> dict:
    """ml_multi.classes plus: number + currency word with Romanian "de" ("20 de milioane de euro") as a price, and
    dates (numeric, or day + month name)."""
    import ml_multi as MM

    c = dict(MM.classes(text, lang))
    t = text.lower()
    if lang == "ro" and not c["cur"]:
        c["cur"] = bool(re.search(r"\d[\d.,]*\s?(?:de\s)?(?:(?:milioan\w*|" + MM.SCALE + r")\s?)?(?:de\s)?(?:" + CUR_RO + ")", t))
    c["date"] = bool(re.search(r"(?<![\d.,])\d{1,2}[./]\d{1,2}[./](?:\d{2}|\d{4})(?!\d)", text)
                     or re.search(r"(?<![\d.,])([1-9]|[12]\d|3[01])\.?\s+(?:" + DATE_WORD.get(lang, "$^") + r")\b", t))
    return c


def year_wants() -> dict:
    want = {y: 4 for y in range(1900, 1990)}
    want.update({y: 40 for y in range(1990, 2031)})
    want.update({y: 10 for y in range(2031, 2061)})
    for y in (2030, 2035, 2040, 2050):
        want[y] = 60
    return want


def select(rows: list, lang: str, years: list, seed: int = 1, total_cap: int = 3000, per_template: int = 3,
           per_session: int = 8, value_cap: int = 80, quotas: dict | None = None) -> tuple:
    """Variety-first selection from `rows` (index rows) of the given debate years. Returns (selected rows, stats)."""
    quotas = quotas or QUOTAS
    rng = random.Random(f"{seed}-{lang}")
    rows = [r for r in rows if r["day"][:4] in years and 40 <= len(r["text"]) <= 450]
    rows.sort(key=lambda r: r["utt_id"])
    rng.shuffle(rows)
    cls = {r["utt_id"]: r.get("_cls") or classes(r["text"], lang) for r in rows}
    tmpl = lambda r: re.sub(r"\d+", "#", re.sub(r"[^\w\s]", "", r["text"].lower()))[:60]  # noqa: E731
    t_n, s_n, taken, sel = collections.Counter(), collections.Counter(), set(), []
    yv = collections.Counter()

    def take(r, why):
        u = r["utt_id"]
        if u in taken or t_n[tmpl(r)] >= per_template or s_n[u.rsplit("_", 1)[0]] >= per_session or len(sel) >= total_cap:
            return False
        if any(yv[y] >= value_cap for y in cls[u]["years"]):
            return False
        taken.add(u)
        t_n[tmpl(r)] += 1
        s_n[u.rsplit("_", 1)[0]] += 1
        yv.update(cls[u]["years"])
        sel.append(dict(r, why=why))
        return True

    want = year_wants()
    by_val = collections.defaultdict(list)
    for r in rows:
        for y in set(cls[r["utt_id"]]["years"]):
            by_val[y].append(r)
    for y in sorted(want, key=lambda y: len(by_val[y])):  # rarest values first
        for r in by_val[y]:
            if yv[y] >= want[y]:
                break
            take(r, f"year:{y}")
    have = collections.Counter(k for r in sel for k in quotas if cls[r["utt_id"]][k])
    for k in sorted(quotas, key=lambda k: sum(cls[r["utt_id"]][k] for r in rows)):
        for r in rows:
            if have[k] >= quotas[k]:
                break
            if cls[r["utt_id"]][k] and take(r, k):
                for kk in quotas:
                    have[kk] += bool(cls[r["utt_id"]][kk])
    tot = sum(yv.values()) or 1
    st = {"lang": lang, "years": years, "clips": len(sel), "year_spans": sum(yv.values()), "year_values": len(yv),
          "y1990_2030_ge40": sum(yv[y] >= 40 for y in range(1990, 2031)),
          "y1990_2030_ge10": sum(yv[y] >= 10 for y in range(1990, 2031)),
          "forward_2031_2060_values": sum(1 for y in yv if 2031 <= y <= 2060),
          "forward_2031_2060_ge10": sum(yv[y] >= 10 for y in range(2031, 2061)),
          "forward_spans": sum(c for y, c in yv.items() if 2031 <= y <= 2099),
          "targets": {y: yv[y] for y in (2030, 2035, 2040, 2050)},
          "old_values": sum(1 for y in yv if 1900 <= y < 1990),
          "top_value_share": round(max(yv.values(), default=0) / tot, 3),
          "top5_share": round(sum(c for _, c in yv.most_common(5)) / tot, 3), "classes": dict(have),
          "sessions": len(s_n), "max_per_session": max(s_n.values(), default=0),
          "max_per_template": max(t_n.values(), default=0), "hours_est": round(sum(min(30, len(r["text"]) / 15) for r in sel) / 3600, 1)}
    return sel, st


def score_sel(st: dict, quotas: dict | None = None) -> float:
    """How well a selection fills the strata (0..1 per stratum, years weighted like all class quotas together)."""
    quotas = quotas or QUOTAS
    yr = (st["y1990_2030_ge40"] / 41 + st["forward_2031_2060_ge10"] / 30 + sum(min(1, v / 60) for v in st["targets"].values()) / 4) / 3
    cl = sum(min(1, st["classes"].get(k, 0) / q) for k, q in quotas.items()) / len(quotas)
    return round(yr + cl, 4)


def choose_years(rows: list, lang: str, k: int = 4, seed: int = 1) -> tuple:
    """Greedy: add the debate year that most improves the strata, up to k years."""
    import ml_multi as MM  # noqa: F401  (classes cache below)

    for r in rows:
        r["_cls"] = classes(r["text"], lang)
    all_years = sorted({r["day"][:4] for r in rows})
    chosen, log = [], []
    for _ in range(k):
        best = None
        for y in all_years:
            if y in chosen:
                continue
            _sel, st = select(rows, lang, chosen + [y], seed)
            sc = score_sel(st)
            if best is None or sc > best[0]:
                best = (sc, y, st)
        chosen.append(best[1])
        log.append({"years": list(chosen), "score": best[0]})
        print(lang, "years", chosen, "score", best[0], file=sys.stderr)
    return chosen, log


# ------------------------------------------------------------------ span kinds (label2 names)
def span_kind(written: str, caption: str, a: int, lang: str) -> tuple:
    """(kind as ml_cands names it, finer class for the census) of one caption span."""
    ctx = caption[max(0, a - 12):a + len(written) + 14]
    c = classes(written, lang)
    cc = classes(ctx, lang)
    if c["date"]:
        return "date", "date"
    if c["clock"] or (cc["clock"] and re.search(r"\d[.:]\d\d", written)):
        return "clock", "clock"
    if c["phone"]:
        return "phone", "phone"
    if c["pct"] or "%" in written:
        return "pct", "pct"
    if c["cur"] or cc["cur"] and re.search(r"(?:" + CUR_RO + r"|zł|złot\w*)", written.lower()):
        return "cur", "cur"
    if c["range"]:
        return "range", "range"
    if c["years"] and re.fullmatch(r"(?:w\s|în\s|anul\s)?(?:1[89]\d\d|20\d\d)(?:\s?(?:r\.?|rok\w*|roku))?", written.strip()):
        return "int", "year"
    if re.search(r"\d\.$|\d-?(?:lea|a)\b", written) or c["ord"]:
        return "ord", "ord"
    if c["dec"]:
        return "dec", "dec"
    if c["big"]:
        return "int", "big"
    return "int", "int"


# ------------------------------------------------------------------ pieces (pure)
def render_tokens(text: str, spans: list, pick: list) -> tuple:
    """Rendered label tokens with provenance: [(token, span index or None)]; spans without a pick keep their written
    text (marked with their index, so a piece can avoid them)."""
    out, pos = [], 0
    for j, (s, k) in enumerate(zip(spans, pick)):
        for t in text[pos:s["a"]].split():
            out.append((t, None))
        words = s["cands"][k] if (k is not None and s["cands"]) else s["written"]
        # punctuation glued to the span's end in the caption stays with its last word
        tail = ""
        rest = text[s["b"]:]
        m = re.match(r"[.,;:!?)\]»\"”']+", rest)
        if m:
            tail = m.group(0)
        ws = words.split()
        for i, t in enumerate(ws):
            out.append((t + (tail if i == len(ws) - 1 else ""), j))
        pos = s["b"] + len(tail)
        # a word glued to the span ("5%-owy") is rare; its remainder starts the next token
    for t in text[pos:].split():
        out.append((t, None))
    return out


def plan_pieces(tokens: list, times: list, good: set, dead: set, dur: float, ctx: float = 1.0, max_side: float = 4.0,
                merge_gap: float = 1.0) -> list:
    """Pieces around the GOOD spans (decided), never reaching a DEAD span (no reading / undecided / leftover digits):
    each piece runs from the start of its sentence part (previous , ; : . ! ?) to the end of it, at least `ctx` s of
    context each side where the clip has it, at most `max_side` s beyond the span; pieces closer than `merge_gap`
    merge. Returns [{"i0", "i1", "t0", "t1", "spans"}] (token range [i0, i1), seconds)."""
    n = len(tokens)
    if not n or any(t is None for t in times):
        return []
    by_span = collections.defaultdict(list)
    for i, (_t, j) in enumerate(tokens):
        if j is not None:
            by_span[j].append(i)
    dead_tok = {i for j in dead for i in by_span.get(j, [])}
    dead_tok |= {i for i, (t, j) in enumerate(tokens) if j is None and re.search(r"\d", t)}
    raw = []
    for j in sorted(good):
        idx = by_span.get(j)
        if not idx:
            continue
        a, b = idx[0], idx[-1]
        t_a, t_b = times[a][0], times[b][1]
        i0 = a
        while i0 > 0 and (i0 - 1) not in dead_tok:
            prev = tokens[i0 - 1][0]
            if re.search(r"[.,;:!?]$", prev) and t_a - times[i0][0] >= ctx:
                break
            if t_a - times[i0 - 1][0] > max_side:
                break
            i0 -= 1
        i1 = b + 1
        while i1 < n and i1 not in dead_tok:
            if re.search(r"[.,;:!?]$", tokens[i1 - 1][0]) and times[i1 - 1][1] - t_b >= ctx:
                break
            if times[i1][1] - t_b > max_side:
                break
            i1 += 1
        raw.append([i0, i1, [j]])
    raw.sort()
    merged = []
    for p in raw:
        if merged and (p[0] <= merged[-1][1] or times[p[0]][0] - times[merged[-1][1] - 1][1] < merge_gap) and not any(
                i in dead_tok for i in range(merged[-1][1], p[0])):
            merged[-1][1] = max(merged[-1][1], p[1])
            merged[-1][2] += p[2]
        else:
            merged.append(p)
    out = []
    for i0, i1, js in merged:
        # cut in the middle of the silence next to each edge word (or at the clip edge)
        left = times[i0 - 1][1] if i0 > 0 else 0.0
        right = times[i1][0] if i1 < n else dur
        t0 = max(0.0, (left + times[i0][0]) / 2 if i0 > 0 else max(0.0, times[i0][0] - 0.3))
        t1 = min(dur, (times[i1 - 1][1] + right) / 2 if i1 < n else min(dur, times[i1 - 1][1] + 0.3))
        if i0 > 0:
            t0 = max(t0, times[i0][0] - 0.25)
        if i1 < n:
            t1 = min(t1, times[i1 - 1][1] + 0.25)
        out.append({"i0": i0, "i1": i1, "t0": round(t0, 3), "t1": round(t1, 3), "spans": sorted(js)})
    return out


def piece_caption(text: str, spans: list, tokens: list, i0: int, i1: int) -> tuple:
    """The caption text of a piece (written spans in place) and its spans re-based to that text."""
    out, sp, seen = [], [], set()
    for i in range(i0, i1):
        t, j = tokens[i]
        if j is None:
            out.append(t)
        elif j not in seen:
            seen.add(j)
            pre = " ".join(out) + (" " if out else "")
            w = spans[j]["written"]
            sp.append(dict(spans[j], a=len(pre), b=len(pre) + len(w), idx=j))
            out.append(w)
    return " ".join(out), sp


# ------------------------------------------------------------------ Modal side
if modal is not None:
    vol = modal.Volume.from_name("parakeet-spoken-form")
    cut_image = (modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg", "libsndfile1")
                 .pip_install("soundfile", "numpy"))
    ctc_image = (modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg", "libsndfile1")
                 .pip_install("torch==2.4.1", "torchaudio==2.4.1", "transformers==4.44.2", "soundfile", "librosa",
                              "numpy<2")
                 .add_local_python_source("ml_multi", "ml_multi_cases", "ml_multi_ctc", "ml_multi_scale", "ml_llm",
                                          "ml_multi_lexgen")
                 .add_local_file(os.path.join(HERE, "ml_multi_lex.json"), "/root/ml_multi_lex.json")
                 .add_local_file(os.path.join(HERE, "ml_multi_months.json"), "/root/ml_multi_months.json"))
    app = modal.App("parakeet-ml-multi-scale")
    cut_fn = app.function(image=cut_image, volumes={VOL: vol}, cpu=2, memory=6144, timeout=8 * 3600, max_containers=8)
    score_fn = app.function(image=ctc_image, volumes={VOL: vol}, cpu=8, memory=16384, timeout=4 * 3600,
                            max_containers=12, secrets=[modal.Secret.from_name("hf")])
    gpu_score_fn = app.function(image=ctc_image, volumes={VOL: vol}, gpu=["T4", "L4", "A10G"], cpu=4, memory=16384,
                                timeout=4 * 3600, max_containers=6, secrets=[modal.Secret.from_name("hf")])
    asm_fn = app.function(image=ctc_image, volumes={VOL: vol}, cpu=2, memory=8192, timeout=2 * 3600)
    entry = app.local_entrypoint
else:
    cut_fn = score_fn = gpu_score_fn = asm_fn = (lambda f: f)
    entry = lambda: (lambda f: f)  # noqa: E731


@cut_fn
def cut_year(lang: str, year: str, ids: list, tag: str, rows_meta: dict, tname: str = "") -> dict:
    """Cut the selected clips of one language-year from `<lang>_<year>.tar` and `<lang>_<year>_2.tar` (16 kHz wav
    -> /vol/ml/p3v2/<lang>/wav/<tag>_<id>.wav); `tname` limits the job to one of the two archives (the two run in
    parallel). Resumable: a dropped stream restarts the tar and skips the sessions already cut; the finished job is
    written to multi_parts/<tag>_<archive>.jsonl and skipped on rerun."""
    import csv
    import gzip
    import subprocess
    import tarfile
    import urllib.request

    import numpy as np
    import soundfile as sf

    tars = [tname] if tname else [f"{lang}_{year}", f"{lang}_{year}_2"]
    part = f"{P3}/{lang}/multi_parts/{tag}_{tname or year}.jsonl"
    if os.path.exists(part):
        rows = [json.loads(l) for l in open(part)]
        return {"stats": {"lang": lang, "year": year, "cut": len(rows), "resumed": True}, "rows": rows}
    want = set(ids)
    events = {u.rsplit("_", 1)[0] for u in want}
    urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/voxpopuli/annotations/unlabelled_v2.tsv.gz", "/tmp/segs.tsv.gz")
    segs = {}
    with gzip.open("/tmp/segs.tsv.gz", "rt") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["event_id"] in events:
                segs[f'{r["event_id"]}_{r["segment_no"]}'] = (float(r["start"]), float(r["end"]))
    out = f"{P3}/{lang}/wav"
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.dirname(part), exist_ok=True)
    rows, done_ev = [], set()
    st = {"lang": lang, "year": year, "wanted": len(want), "seg_found": sum(u in segs for u in want), "cut": 0,
          "too_short": 0, "stream_retries": 0, "tars": []}
    for tname in tars:
        for attempt in range(6):
            try:
                resp = urllib.request.urlopen(f"https://dl.fbaipublicfiles.com/voxpopuli/audios/{tname}.tar", timeout=300)
            except Exception as e:  # noqa: BLE001  (404: not every year has a second tar; else retry)
                if "404" in str(e) or "403" in str(e):
                    break
                st["stream_retries"] += 1
                continue
            try:
                with tarfile.open(fileobj=resp, mode="r|") as tar:
                    for m in tar:
                        if not m.isfile():
                            continue
                        ev = os.path.basename(m.name).rsplit(".", 1)[0]
                        if ev not in events or ev in done_ev:
                            continue
                        data = tar.extractfile(m).read()
                        p = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", "16000",
                                            "-f", "s16le", "pipe:1"], input=data, capture_output=True)
                        x = np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768
                        for u in sorted(u for u in want if u.rsplit("_", 1)[0] == ev and u in segs):
                            s0, e0 = segs[u]
                            clip = x[int(s0 * 16000): int(e0 * 16000)]
                            if len(clip) < 16000:
                                st["too_short"] += 1
                                continue
                            w = f"{out}/{tag}_{u}.wav"
                            sf.write(w, clip, 16000, subtype="PCM_16")
                            rows.append({"audio_filepath": w, "duration": round(len(clip) / 16000, 3),
                                         "text": rows_meta[u]["text"], "lang": lang, "src": "vox", "id": u,
                                         "origin": "new", "why": rows_meta[u].get("why"), "day": u[:8]})
                            st["cut"] += 1
                        done_ev.add(ev)
                st["tars"].append(tname)
                break
            except (tarfile.ReadError, OSError, EOFError):
                st["stream_retries"] += 1
                vol.commit()
                continue
        vol.commit()
    with open(part, "w") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    vol.commit()
    return {"stats": st, "rows": rows}


@entry()
def cut(lang: str, sel: str, tag: str = "multi1"):
    """Cut a local selection file ({"ids_by_year": {year: [utt_id]}, "rows": {utt_id: {text, why}}}) in parallel
    and write ml/p3v2/<lang>/cands_<tag>.jsonl."""
    import io

    S = json.load(open(sel))
    jobs = [(lang, y, ids, tag, {u: S["rows"][u] for u in ids}, t) for y, ids in sorted(S["ids_by_year"].items()) if ids
            for t in (f"{lang}_{y}", f"{lang}_{y}_2")]
    rows, failed = [], []
    for job, res in zip(jobs, cut_year.starmap(jobs, return_exceptions=True)):
        if isinstance(res, Exception):
            failed.append(f"{job[5]}: {type(res).__name__} {str(res)[:120]}")
            continue
        print(json.dumps(res["stats"]))
        rows += res["rows"]
    with vol.batch_upload(force=True) as b:
        b.put_file(io.BytesIO("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()),
                   f"ml/p3v2/{lang}/cands_{tag}.jsonl")
    print(lang, "rows", len(rows), "hours", round(sum(r["duration"] for r in rows) / 3600, 2))
    print("FAILED (rerun resumes):", failed)


# ------------------------------------------------------------------ CTC reader (copied from ml_decide so a concurrent
# edit there cannot change this run; same model class and rule)
READ_MARGIN = 3.0


class Reader:
    def __init__(self, lang: str):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        import ml_multi_ctc as C

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        name = C.READERS[lang]
        self.proc = Wav2Vec2Processor.from_pretrained(name)
        self.model = Wav2Vec2ForCTC.from_pretrained(name).to(self.device).eval()
        self.vocab = self.proc.tokenizer.get_vocab()
        self.blank = self.proc.tokenizer.pad_token_id
        self.delim = self.vocab.get("|")
        letters = [k for k in self.vocab if len(k) == 1 and k.isalpha()]
        self.upper = bool(letters) and sum(k.isupper() for k in letters) > len(letters) / 2

    def lp(self, x):
        with self.torch.no_grad():
            iv = self.proc(x, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
            return self.model(iv).logits.float().log_softmax(-1)[0]

    def greedy(self, lp) -> str:
        return self.proc.decode(lp.argmax(-1).tolist()).lower()

    def ids(self, text: str) -> list:
        import unicodedata

        out = []
        t = text.upper() if self.upper else text.lower()
        for w in re.sub(r"[^\w' ]+", " ", t).split():
            cs = []
            for c in w:
                if c in self.vocab:
                    cs.append(self.vocab[c])
                else:
                    b = "".join(d for d in unicodedata.normalize("NFD", c) if not unicodedata.combining(d))
                    cs += [self.vocab[d] for d in b if d in self.vocab]
            if cs:
                if out and self.delim is not None:
                    out.append(self.delim)
                out += cs
        return out

    def nll(self, lp, text: str) -> float:
        torch = self.torch
        ids = self.ids(text)
        if not ids or len(ids) > lp.shape[0]:
            return float("inf")
        return float(torch.nn.functional.ctc_loss(lp[:, None, :], torch.tensor([ids], device=lp.device),
                                                  torch.tensor([lp.shape[0]]), torch.tensor([len(ids)]),
                                                  blank=self.blank, reduction="sum", zero_infinity=True))

    def word_times(self, lp, toks: list) -> list:
        import torchaudio.functional as AF

        torch = self.torch
        ids, owner = [], []
        for ti, tok in enumerate(toks):
            cs = self.ids(tok)
            if not cs:
                continue
            if ids and self.delim is not None:
                ids.append(self.delim)
                owner.append(-1)
            ids += cs
            owner += [ti] * len(cs)
        out = [None] * len(toks)
        need = len(ids) + sum(1 for a, b in zip(ids, ids[1:]) if a == b)
        if not ids or need > lp.shape[0]:
            return out
        try:
            al, sc = AF.forced_align(lp[None].float().cpu(), torch.tensor([ids], dtype=torch.int32), blank=self.blank)
            spans = AF.merge_tokens(al[0], sc[0].exp(), blank=self.blank)
        except (RuntimeError, ValueError):
            return out
        if len(spans) != len(ids):
            return out
        sec = 320 / 16000
        for sp, ti in zip(spans, owner):
            if ti < 0:
                continue
            t0, t1 = sp.start * sec, (sp.end + 1) * sec
            out[ti] = (t0, t1) if out[ti] is None else (min(out[ti][0], t0), max(out[ti][1], t1))
        return out


def _norm(w: str) -> str:
    import unicodedata

    w = w.lower().replace("ß", "ss").replace("ł", "l")
    w = "".join(c for c in unicodedata.normalize("NFD", w) if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", w)


def rkey(reading: str) -> str:
    return "".join(_norm(w) for w in reading.split())


# a length-relative fit tolerance only where the reader's greedy text is too noisy for 1-2 edits (Romanian); Polish
# keeps ml_decide's: a relative tolerance let "dwadzieścia DO dwadzieścia pięć procent" fit a greedy text without
# "do" (pilot hand check 2026-10-16)
FIT_REL = {"ro": 0.2}
_FIT_LANG = [""]


def fits(reading: str, greedy: str) -> bool:
    """ml_decide.fits (fuzzy), with a tolerance that grows with the reading: the reading's letters are a run of
    greedy words, or (>= 6 letters) occur in the space-free greedy text within max(1 or 2, 20% of its letters) edits,
    ending at a greedy word boundary. The Romanian reader glues and drops letters ("domi dozeci și patru" for "două
    mii douăzeci și patru"), so the fixed 1-2 edits of ml_decide rejected half its readings (2026-10-16 pilot); the
    VALUE is checked by the external transcript on every span, and the CTC margin still chooses among candidates."""
    k = rkey(reading)
    ws = [w for w in (_norm(x) for x in greedy.split()) if w]
    for i in range(len(ws)):
        acc = ""
        for j in range(i, len(ws)):
            acc += ws[j]
            if acc == k:
                return True
            if len(acc) >= len(k) or not k.startswith(acc):
                break
    if len(k) < 6:
        return False
    tol = max(2 if len(k) >= 12 else 1, int(FIT_REL.get(_FIT_LANG[0], 0.0) * len(k)))
    g, ends, pos = "", set(), 0
    for w in ws:
        g += w
        pos += len(w)
        ends.add(pos)
    if not ends:
        return False
    prev = [0] * (len(g) + 1)
    for i, c in enumerate(k, 1):
        cur = [i] + [0] * len(g)
        for j, d in enumerate(g, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (c != d))
        prev = cur
    return min(prev[j] for j in ends) <= tol


def greedy_dist(reading: str, greedy: str) -> int:
    """Edit distance of the reading's letters to the closest stretch of the space-free greedy text that ends at a
    greedy word boundary (the fuzzy part of `fits`)."""
    k = rkey(reading)
    ws = [w for w in (_norm(x) for x in greedy.split()) if w]
    g, ends, pos = "", set(), 0
    for w in ws:
        g += w
        pos += len(w)
        ends.add(pos)
    if not ends or not k:
        return 10**6
    prev = [0] * (len(g) + 1)
    for i, c in enumerate(k, 1):
        cur = [i] + [0] * len(g)
        for j, d in enumerate(g, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (c != d))
        prev = cur
    return min(prev[j] for j in ends)


def single_safe(reading: str, lang: str) -> bool:
    """Every word of the reading is a number word (its words follow from the value)."""
    import ml_multi as MM

    lx = MM.lexicon(lang)
    glue = lx.glue | {"de"}  # "douăzeci ȘI cinci", "douăzeci DE mii" are part of the number
    for t in reading.split():
        k = re.sub(r"[^\w'-]", "", MM.fold(t)).strip("-'")
        kind = MM._token_kind(lx, k)
        if not (kind and kind[0] == "num") and k not in glue:
            return False  # a decimal / separator / unit word, or not a number word at all
    return True


def greedy_veto(cands: list, best: int, greedy: str) -> bool:
    """True when a rival reading (other letters) is CLOSER to the greedy transcript than the CTC choice: the two
    signals of the same reader disagree, so the ending is not trusted (2026-10-16 audit: 43 of 2,837 Polish choices,
    e.g. "czternastego trzydzieści" where the greedy text says "czternastej", "siedem minuta")."""
    db = greedy_dist(cands[best], greedy)
    bk = rkey(cands[best])
    return any(greedy_dist(c, greedy) < db for c in cands if rkey(c) != bk)


def cer(a: str, b: str) -> float:
    import difflib

    x, y = rkey(a), rkey(b)
    if not x:
        return 1.0
    sm = difflib.SequenceMatcher(None, x, y, autojunk=False)
    return 1 - sum(bl.size for bl in sm.get_matching_blocks()) / max(len(x), 1)


WIN_CTX = 1.5  # s of context either side of a span when its candidates are scored on a window of the clip


def _window(toks: list, times: list, j: int, n_frames: int):
    """(frame0, frame1, left words, right words) of the window that scores span j's candidates: the span's words in
    the current label plus whole words of >= WIN_CTX s context either side (CTC forced-alignment times), or None when
    the times are missing (the caller then scores the whole clip). Every candidate is scored on the same frames with
    the same context, so the margin compares like with like."""
    idx = [i for i, (_t, jj) in enumerate(toks) if jj == j]
    if not idx or times is None or any(times[i] is None for i in idx):
        return None
    a, b = idx[0], idx[-1]
    l = a
    while l > 0 and (times[l - 1] is not None) and times[a][0] - times[l - 1][0] <= WIN_CTX:
        l -= 1
    rr = b
    while rr + 1 < len(toks) and (times[rr + 1] is not None) and times[rr + 1][1] - times[b][1] <= WIN_CTX:
        rr += 1
    if (l > 0 and times[l - 1] is None) or (rr + 1 < len(toks) and times[rr + 1] is None):
        return None  # an untimed word (digits) next to the window: its audio would be unexplained
    # the window runs from the end of the word before it to the start of the word after it: a longer candidate
    # ("trzynastego") must find its extra sounds inside the window even when the current label ("trzynaste") was
    # aligned shorter (pilot hand check 2026-10-16: cutting at the aligned end favoured the shorter reading)
    f0 = int(times[l - 1][1] / 0.02) if l > 0 else 0
    f1 = min(n_frames, int(times[rr + 1][0] / 0.02) + 1) if rr + 1 < len(toks) else n_frames
    if f1 - f0 < 10:
        return None
    return f0, f1, [t for t, _ in toks[l:a]], [t for t, _ in toks[b + 1:rr + 1]]


@score_fn
def score_shard(lang: str, items_path: str, out_path: str, tag: str, window: bool = True) -> dict:
    return _score_impl(lang, items_path, out_path, tag, window)


@gpu_score_fn
def score_shard_gpu(lang: str, items_path: str, out_path: str, tag: str, window: bool = True) -> dict:
    return _score_impl(lang, items_path, out_path, tag, window)


def _score_impl(lang: str, items_path: str, out_path: str, tag: str, window: bool = True) -> dict:
    """Per row: CTC scores of every candidate of every span (other spans at their current best; two passes), the
    decision, greedy transcript, word times of the decided label, and the planned pieces (FLAC cut to
    /vol/ml/p3v2/<lang>/flac_<tag>/<id>_p<k>.flac). Resumable: an existing out_path is returned as is."""
    import time

    import soundfile as sf

    if os.path.exists(out_path):
        return {"resumed": True, "out": out_path}
    t_start = time.time()
    R = Reader(lang)
    _FIT_LANG[0] = lang
    t_load = time.time() - t_start
    rows = [json.loads(l) for l in open(items_path)]
    fdir = f"{P3}/{lang}/flac_{tag}"
    os.makedirs(fdir, exist_ok=True)
    res = []
    n_nll = n_win = n_full = 0
    for r in rows:
        x, sr = sf.read(r["audio_filepath"], dtype="float32")
        lp = R.lp(x)
        greedy = R.greedy(lp)
        spans = r["spans"]
        sp = [j for j, s in enumerate(spans) if s["cands"]]
        pick = [0 if s["cands"] else None for s in spans]
        dec = {}
        for _ in range(2):
            cur = render_tokens(r["text"], spans, pick)
            ctimes = R.word_times(lp, [t for t, _ in cur]) if window else None
            for j in sp:
                s = spans[j]
                win = _window(cur, ctimes, j, lp.shape[0]) if window else None
                scores = []
                for ci in range(len(s["cands"])):
                    if win:
                        f0, f1, left, right = win
                        scores.append(R.nll(lp[f0:f1], " ".join(left + [s["cands"][ci]] + right)))
                    else:
                        p = list(pick)
                        p[j] = ci
                        scores.append(R.nll(lp, " ".join(t for t, _ in render_tokens(r["text"], spans, p))))
                    n_nll += 1
                n_win += bool(win)
                best = min(range(len(scores)), key=lambda i: scores[i])
                bk = rkey(s["cands"][best])
                rivals = [i for i in range(len(scores)) if rkey(s["cands"][i]) != bk]
                margin = min((scores[i] - scores[best] for i in rivals), default=None)
                dec[j] = {"best": best, "margin": None if margin is None else round(margin, 2),
                          "fit": fits(s["cands"][best], greedy), "n": len(s["cands"]),
                          "top": [s["cands"][i] for i in sorted(range(len(scores)), key=lambda i: scores[i])[:3]],
                          "top_s": [round(v, 2) for v in sorted(scores)[:3]]}
                pick[j] = best
        if window:  # a close call on the window is scored again on the whole clip (pilot: "trzynastego" lost 0.66
            for j in sp:  # on its window, won by 23 nats on the clip)
                d, s = dec[j], spans[j]
                if d["n"] < 2 or (d["margin"] is not None and d["margin"] >= READ_MARGIN):
                    continue
                scores = []
                for ci in range(len(s["cands"])):
                    p = list(pick)
                    p[j] = ci
                    scores.append(R.nll(lp, " ".join(t for t, _ in render_tokens(r["text"], spans, p))))
                    n_nll += 1
                best = min(range(len(scores)), key=lambda i: scores[i])
                bk = rkey(s["cands"][best])
                rivals = [i for i in range(len(scores)) if rkey(s["cands"][i]) != bk]
                margin = min((scores[i] - scores[best] for i in rivals), default=None)
                dec[j] = {"best": best, "margin": None if margin is None else round(margin, 2),
                          "fit": fits(s["cands"][best], greedy), "n": len(s["cands"]), "full": True,
                          "top": [s["cands"][i] for i in sorted(range(len(scores)), key=lambda i: scores[i])[:3]],
                          "top_s": [round(v, 2) for v in sorted(scores)[:3]]}
                pick[j] = best
                n_full += 1
        good, dead = set(), set()
        for j, s in enumerate(spans):
            d = dec.get(j)
            # one candidate: its words follow from the value (Romanian cardinals do not inflect), and the external
            # transcript checks the value; several: the CTC best must fit the greedy transcript and lead by MARGIN
            ok = bool(d) and ((d["n"] == 1 and (d["fit"] or single_safe(s["cands"][0], lang))) or (
                d["fit"] and d["margin"] is not None and d["margin"] >= READ_MARGIN
                and not greedy_veto(s["cands"], d["best"], greedy)))
            (good if ok else dead).add(j)
        toks = render_tokens(r["text"], spans, [pick[j] if j in good else None for j in range(len(spans))])
        times = R.word_times(lp, [t for t, _ in toks])
        dur = len(x) / 16000
        # a token without letters (digits left in the caption) has no time: pieces may not touch it
        tfix = list(times)
        for i, t in enumerate(tfix):
            if t is None:
                prev = next((tfix[k] for k in range(i - 1, -1, -1) if tfix[k]), (0.0, 0.0))
                tfix[i] = (prev[1], prev[1])
        no_time = [i for i, t in enumerate(times) if t is None and re.search(r"[^\W\d_]", toks[i][0])]
        pieces = plan_pieces(toks, tfix, good, dead | ({-1} if no_time else set()), dur) if not no_time else []
        for k, pc in enumerate(pieces):
            fp = f"{fdir}/{r['id']}_p{k}.flac"
            sf.write(fp, x[int(pc["t0"] * 16000):int(pc["t1"] * 16000)], 16000, format="FLAC")
            pc["flac"] = fp
            pc["text"] = " ".join(t for t, _ in toks[pc["i0"]:pc["i1"]])
        res.append({"id": r["id"], "greedy": greedy, "dec": {str(j): d for j, d in dec.items()},
                    "good": sorted(good), "dead": sorted(dead), "pick": pick, "tokens": toks,
                    "times": [None if t is None else [round(t[0], 3), round(t[1], 3)] for t in times],
                    "pieces": pieces, "dur": round(dur, 3), "cer_ctc": round(cer(" ".join(t for t, _ in toks), greedy), 3)})
    with open(out_path, "w") as f:
        for x_ in res:
            f.write(json.dumps(x_, ensure_ascii=False) + "\n")
    vol.commit()
    return {"rows": len(res), "secs": round(time.time() - t_start, 1), "load_s": round(t_load, 1), "nll": n_nll,
            "windowed": n_win, "full_rescored": n_full, "out": out_path}


@entry()
def score(lang: str, items: str, tag: str = "multi1", shard: int = 250, window: bool = True, gpu: bool = False,
          prefix: str = ""):
    """items: volume path of the items jsonl (rows with "spans" from ml_multi_ctc.candidates). Splits it into shards
    of `shard` rows, scores them in parallel; outputs <lang>/ctc_<tag>/shard_<k>.jsonl."""
    import io

    lines = b"".join(vol.read_file(items)).decode().splitlines()
    base = f"ml/p3v2/{lang}/ctc_{tag}"
    jobs = []
    for k in range(0, len(lines), shard):
        p = f"{base}/items_{prefix}{k // shard:03d}.jsonl"
        with vol.batch_upload(force=True) as b:
            b.put_file(io.BytesIO(("\n".join(lines[k:k + shard]) + "\n").encode()), p)
        jobs.append((lang, f"{VOL}/{p}", f"{VOL}/{base}/shard_{prefix}{k // shard:03d}.jsonl", tag, window))
    failed = []
    fn = score_shard_gpu if gpu else score_shard
    for job, res in zip(jobs, fn.starmap(jobs, return_exceptions=True)):
        if isinstance(res, Exception):
            failed.append(f"{job[2]}: {type(res).__name__} {str(res)[:200]}")
            continue
        print(json.dumps(res))
    print("FAILED (rerun resumes):", failed)


# ------------------------------------------------------------------ assemble (pure + Modal)
WF_LABEL = re.compile(r"[0-9$€£¥¢₹₩₽%#/+°²³¹½¼¾⅓⅔&‰]")  # mix_build.WF_LABEL: never in a spoken-form label
MAX_CER = 0.25  # ml_p3.MAX_CER: the caption must fit the CTC greedy transcript of the clip
MIN_AGREE = 0.75  # word agreement of a piece's caption with its external transcript (digits on both sides)
MIN_PIECE_S = 1.5


def _edge_ok(cap_text: str, ext_words: list) -> bool:
    """The piece's first and last caption words are heard at the piece's edges (within the external transcript's
    first / last two words): the cut neither clipped an edge word nor kept a neighbour."""
    import ml_multi as MM

    c = [MM._key(t) for t in cap_text.split() if MM._key(t)]
    e = [MM._key(w["w"]) for w in ext_words if MM._key(w["w"])]
    if not c or not e:
        return False

    def near(a, bs):
        if re.search(r"\d", a):
            return True  # a number at the edge is checked by its value
        return any(a == b or (len(a) >= 5 and (a.startswith(b[:-2]) or b.startswith(a[:-2]))) for b in bs)

    return near(c[0], e[:2]) and near(c[-1], e[-2:])


def core_agreement(cap_text: str, ext_words: list, trim: int = 2) -> float:
    """Word agreement of a piece's caption with its external transcript, not counting differences that touch the
    first / last `trim` caption words (the cut edges; they matter only when the piece itself becomes a row)."""
    import difflib

    import ml_multi as MM

    a = [MM._key(t) for t in cap_text.split() if MM._key(t)]
    b = [MM._key(w["w"]) for w in ext_words if MM._key(w["w"])]
    if len(a) <= 2 * trim:
        return MM.word_agreement(cap_text, ext_words)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    errs = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or i2 <= trim or i1 >= len(a) - trim:
            continue
        errs += max(i2 - i1, j2 - j1)
    return max(0.0, 1 - errs / (len(a) - 2 * trim))


def _num_chars(toks: list, spans_ok: set) -> list:
    """[a, b) character ranges of the spoken numbers of `toks` ([(token, span index)] joined by single spaces); a
    range never holds punctuation (it is split there)."""
    out, pos, cur = [], 0, None
    for i, (t, j) in enumerate(toks):
        core = re.sub(r"[.,;:!?)\]»\"”']+$", "", t)
        lead = len(t) - len(t.lstrip("([«\"„“'"))
        a, b = pos + lead, pos + len(core)
        if j is not None and j in spans_ok and b > a:
            if cur and cur[2] == j and not re.search(r"[.,;:!?]$", toks[i - 1][0]):
                cur[1] = b
            else:
                if cur:
                    out.append(cur[:2])
                cur = [a, b, j]
        else:
            if cur:
                out.append(cur[:2])
            cur = None
        pos += len(t) + 1
    if cur:
        out.append(cur[:2])
    return out


def assemble_row(item: dict, sc: dict, ext: dict, lang: str) -> dict:
    """One clip: which spans pass (CTC decided + the external transcript of their piece gives the caption value),
    and the rows it yields: the whole clip when every span passes, else each passing piece."""
    import ml_multi as MM
    import ml_multi_ctc as C

    spans = item["spans"]
    toks = [tuple(t) for t in sc["tokens"]]
    for i, (t, j) in enumerate(toks):  # a reading that starts a sentence starts with a capital
        if j is not None and (i == 0 or re.search(r"[.!?]$", toks[i - 1][0])) and (i == 0 or toks[i - 1][1] != j):
            toks[i] = (t[:1].upper() + t[1:], j)
    good = set(sc["good"])
    vetoed = {j for j in good if spans[j]["cands"] and len(spans[j]["cands"]) > 1
              and greedy_veto(spans[j]["cands"], sc["pick"][j], sc["greedy"])}
    # a single candidate skips the CTC choice; that is safe only when its words follow from the value alone (number
    # words): a connector or separator word ("pięć PRZECINEK trzy" for "ASP 5.3", said "pięć kreska trzy") must fit
    vetoed |= {j for j in good if len(spans[j]["cands"]) == 1 and not single_safe(spans[j]["cands"][0], lang)
               and not sc["dec"].get(str(j), {}).get("fit")}
    good -= vetoed
    why = {j: ("no_reading" if not s["cands"] else "greedy_veto" if j in vetoed else "undecided")
           for j, s in enumerate(spans) if j not in good}
    span_ok, piece_res = {}, []
    for k, pc in enumerate(sc["pieces"]):
        e = ext.get(f"{item['id']}_p{k}")
        cap_text, psp = piece_caption(item["text"], spans, toks, pc["i0"], pc["i1"])
        if not e or not e.get("words"):
            for s in psp:
                why[s["idx"]] = "no_ext"
            piece_res.append({"ok": False, "check_ok": False, "why": "no_ext"})
            continue
        vals = {}
        for s in psp:
            v = C.ext_value_ok(cap_text, s, e["words"], lang)
            vals[s["idx"]] = v
            if v is True and s["idx"] in good:
                span_ok[s["idx"]] = True
            elif v is True:
                pass  # the value checks but the reading was vetoed
            else:
                why[s["idx"]] = "ext_value_false" if v is False else "ext_value_none"
        agree = MM.word_agreement(cap_text, e["words"])
        core = core_agreement(cap_text, e["words"])
        edge = _edge_ok(cap_text, e["words"])
        text = " ".join(t for t, _ in toks[pc["i0"]:pc["i1"]])
        vals_ok = all(v is True for v in vals.values()) and all(s["idx"] in good for s in psp)
        # the piece CHECKS its spans (values + the words around them, cut edges aside) ...
        check_ok = vals_ok and core >= MIN_AGREE
        # ... and is a row of its own only when its cut is clean too
        ok = check_ok and agree >= MIN_AGREE and edge and not WF_LABEL.search(text)
        piece_res.append({"ok": ok, "check_ok": check_ok, "agree": round(agree, 3), "core": round(core, 3),
                          "edge": edge, "vals": {str(a): b for a, b in vals.items()},
                          "why": "" if ok else ("value" if not vals_ok else "agree" if core < MIN_AGREE else
                                                "edge" if not edge or agree < MIN_AGREE else "symbol")})
        if not check_ok:
            for s in psp:
                if vals.get(s["idx"]) is True:
                    why[s["idx"]] = "piece_" + piece_res[-1]["why"]
                    span_ok.pop(s["idx"], None)
    text = " ".join(t for t, _ in toks)
    whole_ok = (bool(spans) and len(span_ok) == len(spans) and all(p["check_ok"] for p in piece_res) and bool(piece_res)
                and sc["cer_ctc"] <= MAX_CER and not WF_LABEL.search(text))

    def span_recs(js):
        out = []
        for j in sorted(js):
            s, d = spans[j], sc["dec"].get(str(j), {})
            kind, cls = span_kind(s["written"], item["text"], s["a"], lang)
            rd = s["cands"][sc["pick"][j]]
            out.append({"text": s["written"], "kind": kind, "cls": cls, "reading": rd, "canonical": rd,
                        "pick": sc["pick"][j], "margin": d.get("margin"), "n_cands": len(s["cands"]), "by": "ctc",
                        "ext_value": True, "top": d.get("top", [])[:3]})
        return out

    rows = []
    base = {"lang": lang, "src": "vox", "caption": item["text"], "cer_ctc": sc["cer_ctc"]}
    if whole_ok:
        rows.append(dict(base, id=item["id"], origin="new", duration=sc["dur"], text=text, spans=span_recs(span_ok),
                         num_chars=_num_chars(toks, set(span_ok)), word_times=sc["times"], t0=0.0, t1=sc["dur"],
                         ext_scope="pieces", cer_ext=round(1 - min(p["core"] for p in piece_res), 3)))
    else:
        for k, (pc, pr) in enumerate(zip(sc["pieces"], piece_res)):
            if not pr["ok"] or pc["t1"] - pc["t0"] < MIN_PIECE_S:
                continue
            ptoks = toks[pc["i0"]:pc["i1"]]
            js = {j for _t, j in ptoks if j is not None}
            if not js or not js <= set(span_ok):
                continue
            wt = [None if t is None else [round(t[0] - pc["t0"], 3), round(t[1] - pc["t0"], 3)]
                  for t in sc["times"][pc["i0"]:pc["i1"]]]
            rows.append(dict(base, id=f"{item['id']}_p{k}", origin="piece", parent=item["id"],
                             duration=round(pc["t1"] - pc["t0"], 3), text=" ".join(t for t, _ in ptoks),
                             spans=span_recs(js), num_chars=_num_chars(ptoks, js), word_times=wt, t0=pc["t0"],
                             t1=pc["t1"], ext_scope="piece", cer_ext=round(1 - pr["agree"], 3)))
    return {"rows": rows, "whole": whole_ok, "span_ok": sorted(span_ok), "why": {str(j): w for j, w in why.items()},
            "pieces": piece_res, "n_spans": len(spans)}


@asm_fn
def assemble_lang(lang: str, tag: str, items_path: str, ext_path: str, out_name: str) -> dict:
    """Assemble every clip of a pass: rows -> /vol/ml/p3v2/<lang>/<out_name>.jsonl (+ piece wavs), diag ->
    diag_<out_name>.jsonl."""
    import glob

    import soundfile as sf

    items = {json.loads(l)["id"]: json.loads(l) for l in open(items_path)}
    sc = {}
    for p in sorted(glob.glob(f"{P3}/{lang}/ctc_{tag}/shard_*.jsonl")):
        for l in open(p):
            x = json.loads(l)
            sc[x["id"]] = x
    ext = {}
    for l in open(ext_path):
        j = json.loads(l)
        if j.get("status") == "completed":
            ext[j["id"]] = j
    st = collections.Counter()
    os.makedirs(f"{P3}/{lang}/wav", exist_ok=True)
    with open(f"{P3}/{lang}/{out_name}.jsonl", "w") as rf, open(f"{P3}/{lang}/diag_{out_name}.jsonl", "w") as df:
        for rid, it in items.items():
            if rid not in sc:
                st["no_score"] += 1
                continue
            res = assemble_row(it, sc[rid], ext, lang)
            st["clips"] += 1
            st["spans"] += res["n_spans"]
            st["spans_ok"] += len(res["span_ok"])
            st["whole"] += res["whole"]
            for w in res["why"].values():
                st["why_" + w] += 1
            x = None
            for row in res["rows"]:
                if row["origin"] == "piece":
                    if x is None:
                        x, _sr = sf.read(it["audio_filepath"], dtype="float32")
                    w = f"{P3}/{lang}/wav/{tag}_{row['id']}.wav"
                    sf.write(w, x[int(row["t0"] * 16000):int(row["t1"] * 16000)], 16000, subtype="PCM_16")
                    row["audio_filepath"] = w
                else:
                    row["audio_filepath"] = it["audio_filepath"]
                st["rows_" + row["origin"]] += 1
                st["spans_in_rows"] += len(row["spans"])
                st["h_" + row["origin"]] += row["duration"] / 3600
                rf.write(json.dumps(row, ensure_ascii=False) + "\n")
            df.write(json.dumps({"id": rid, **{k: v for k, v in res.items() if k != "rows"},
                                 "rows": [r["id"] for r in res["rows"]]}, ensure_ascii=False) + "\n")
    vol.commit()
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items()}


@entry()
def assemble(lang: str, tag: str, items: str, ext: str, out_name: str):
    print(json.dumps(assemble_lang.remote(lang, tag, f"{VOL}/{items}", f"{VOL}/{ext}", out_name)))


# ------------------------------------------------------------------ CLI (local steps)
def main():
    import argparse

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("select")
    p.add_argument("--lang")
    p.add_argument("--index")
    p.add_argument("--out")
    p.add_argument("--years", type=int, default=4)
    p.add_argument("--fixed-years", default="")
    p.add_argument("--cap", type=int, default=3000)
    a = ap.parse_args()
    if a.cmd == "select":
        rows = [json.loads(l) for l in open(a.index)]
        if a.fixed_years:
            years, log = a.fixed_years.split(","), []
            for r in rows:
                r["_cls"] = classes(r["text"], a.lang)
        else:
            years, log = choose_years(rows, a.lang, a.years)
        sel, st = select(rows, a.lang, years, total_cap=a.cap)
        ids_by_year = collections.defaultdict(list)
        for r in sel:
            ids_by_year[r["day"][:4]].append(r["utt_id"])
        json.dump({"stats": st, "years_log": log, "ids_by_year": ids_by_year,
                   "rows": {r["utt_id"]: {"text": r["text"], "why": r["why"]} for r in sel}},
                  open(a.out, "w"), ensure_ascii=False)
        print(json.dumps(st, ensure_ascii=False))


if __name__ == "__main__":
    main()
