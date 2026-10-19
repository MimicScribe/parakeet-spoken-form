#!/usr/bin/env python3
"""General-speech word-error sentinel: score checkpoints on number-free FLEURS dev speech (13 languages).

usage: wer_sentinel.py RUN [RUN ...] [--control w3soup_a0.5@0] [--no-stock] [--decode] [--work DIR] [--pipeline DIR]
                       [--out FILE] [--json FILE]
  RUN       "ultra", "<run>@<step>" or "<path>.pt" (as pipeline/modal_app.py loads them)
  --decode  decode the arms that have no hyps/<run>/wer_sentinel_v1.jsonl yet (modal run wer_sentinel.py::decode, L4,
            ~$0.1 for a set of arms; existing outputs are kept) before fetching and scoring
  --control the arm every other arm is compared with (default w3soup_a0.5@0); stock "ultra" is added as a reference

Metric: WER on text with case and punctuation removed (apostrophes dropped, hyphens split), number-word morphemes a model
spaced out ("zwei und zwanzig", "vier en twintig", "venti tre") joined with compound_split.join_morphemes in de / nl / it.
Pooled WER = total word errors / total reference words. Pass bar (plan B6): pooled delta vs the control <= +0.15 points
for all 13 languages and for each group (wave-1 = en, de, es, fr, it, nl, pt; other = ru, uk, pl, ro, cs, hu), and no
language above +0.5 points. A paired bootstrap over utterances gives the 95% interval of each delta (report-only).
Bed: pipeline/wer_sentinel.py (FLEURS dev, number-free, ~200 utterances per language)."""
import argparse
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPE = HERE.parents[1] / "pipeline"
sys.path.insert(0, str(PIPE))
sys.path.insert(0, str(HERE))
import volume_io as V  # noqa: E402

TAG = "wer_sentinel_v1"
WAVE1 = ("en", "de", "es", "fr", "it", "nl", "pt")
OTHER = ("ru", "uk", "pl", "ro", "cs", "hu")
LANGS = WAVE1 + OTHER
JOIN_LANGS = ("de", "nl", "it")
BAR_POOLED, BAR_LANG = 0.15, 0.5  # points of WER (absolute)
MAX_RUN = 8


def norm_tokens(text: str) -> list[str]:
    """Lower case, punctuation and symbols removed: apostrophes deleted, hyphens and every other mark split a word."""
    s = unicodedata.normalize("NFC", text.lower())
    s = re.sub(r"['’ʼ`´]", "", s)
    s = "".join(c if unicodedata.category(c)[0] in "LNM" else " " for c in s)
    return s.split()


def join_numbers(toks: list[str], lang: str) -> list[str]:
    """Join runs of number morphemes that spell one compound number word (de / nl / it); other languages unchanged."""
    if lang not in JOIN_LANGS:
        return toks
    import compound_split as CS

    out, i = [], 0
    while i < len(toks):
        for j in range(min(len(toks), i + MAX_RUN), i + 1, -1):
            cand = CS.join_morphemes(toks[i:j], lang)
            if [p.lower() for p in CS.split_token(cand, lang)] == toks[i:j]:
                out.append(cand.lower())
                i = j
                break
        else:
            out.append(toks[i])
            i += 1
    return out


def words(text: str, lang: str) -> list[str]:
    return join_numbers(norm_tokens(text), lang)


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def score_rows(rows: list[dict]) -> dict:
    """id -> (lang, errors, ref words) for hyp rows {id, ref, hyp}."""
    out = {}
    for r in rows:
        lang = r["id"].split("_", 1)[0]
        ref, hyp = words(r["ref"], lang), words(r["hyp"], lang)
        out[r["id"]] = (lang, edit_distance(ref, hyp), len(ref))
    return out


def pooled(sc: dict, langs, ids=None) -> tuple[int, int]:
    e = n = 0
    for i, (l, ee, nn) in sc.items():
        if l in langs and (ids is None or i in ids):
            e, n = e + ee, n + nn
    return e, n


def wer(sc: dict, langs) -> float:
    e, n = pooled(sc, langs)
    return 100.0 * e / n if n else float("nan")


def bootstrap_delta(a: dict, b: dict, langs, reps: int = 2000, seed: int = 7) -> tuple[float, float, float]:
    """Paired bootstrap over utterances: (delta, lo, hi) of WER(a) - WER(b) in points; a, b = score_rows results."""
    import numpy as np

    ids = [i for i in a if i in b and a[i][0] in langs]
    ea = np.array([a[i][1] for i in ids], float)
    eb = np.array([b[i][1] for i in ids], float)
    n = np.array([a[i][2] for i in ids], float)
    if not len(ids):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(reps, len(ids)))
    d = 100.0 * (ea[idx].sum(1) - eb[idx].sum(1)) / n[idx].sum(1)
    return 100.0 * (ea.sum() - eb.sum()) / n.sum(), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def load_hyps(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def remote_has(run: str) -> bool:
    return f"{TAG}.jsonl" in V.vlist(f"hyps/{run}")


def fetch(run: str, work: Path) -> Path:
    p = work / (run.replace("/", "_") + ".jsonl")
    if not p.exists() and not V.read_shared(f"hyps/{run}/{TAG}.jsonl", p):
        raise SystemExit(f"hyps/{run}/{TAG}.jsonl is not on the volume (run with --decode)")
    return p


def decode_missing(runs: list[str], pipe: Path) -> None:
    todo = [r for r in runs if not remote_has(r)]
    if todo:
        print(f"decoding {len(todo)} arm(s): {', '.join(todo)}", flush=True)
        subprocess.run(["modal", "run", "wer_sentinel.py::decode", "--runs", ",".join(todo)], cwd=pipe, check=True)


def report(arms: dict, control: str) -> tuple[str, dict]:
    """Table text and a json-able summary. arms: name -> score_rows result."""
    names = list(arms)
    ctrl = arms[control]
    groups = [("wave-1", WAVE1), ("other", OTHER), ("all 13", LANGS)]
    w = max(12, *(len(n) + 1 for n in names))
    lines = [f"WER % on {TAG} (number-free FLEURS dev; case/punct removed). control = {control}", ""]
    lines.append(f"{'lang':8}{'words':>7}" + "".join(f"{n:>{w}}" for n in names))
    summary = {"control": control, "wer": {}, "delta": {}, "verdict": {}}
    for name, langs in [(l, (l,)) for l in LANGS] + groups:
        if name in ("wave-1", "other"):
            lines.append("")
        n = pooled(ctrl, langs)[1]
        lines.append(f"{name:8}{n:>7}" + "".join(f"{wer(arms[a], langs):>{w}.2f}" for a in names))
        for a in names:
            summary["wer"].setdefault(a, {})[name] = wer(arms[a], langs)
    lines += ["", f"delta vs control (points; 95% paired bootstrap), bar: groups <= +{BAR_POOLED}, language <= +{BAR_LANG}"]
    fails = {}
    for a in names:
        if a == control:
            continue
        lines.append(f"  {a}")
        bad = []
        for name, langs in [(l, (l,)) for l in LANGS] + groups:
            d, lo, hi = bootstrap_delta(arms[a], ctrl, langs)
            bar = BAR_LANG if len(langs) == 1 else BAR_POOLED
            over = d > bar
            summary["delta"].setdefault(a, {})[name] = {"d": d, "lo": lo, "hi": hi, "over_bar": over}
            if over:
                bad.append(name)
            if len(langs) > 1 or over or abs(d) >= 0.3:
                lines.append(f"    {name:8}{d:+7.2f}  [{lo:+.2f}, {hi:+.2f}]{'   OVER BAR' if over else ''}")
        fails[a] = bad
        summary["verdict"][a] = "PASS" if not bad else "FAIL: " + ", ".join(bad)
        lines.append(f"    -> {summary['verdict'][a]}")
    return "\n".join(lines), summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--control", default="w3soup_a0.5@0")
    ap.add_argument("--no-stock", action="store_true")
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--work", type=Path, default=Path("/tmp/wer_sentinel_work"))
    ap.add_argument("--pipeline", type=Path, default=PIPE, help="pipeline dir to launch modal from (a clean export)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    runs = list(dict.fromkeys(([] if a.no_stock else ["ultra"]) + [a.control] + a.runs))
    if a.decode:
        decode_missing(runs, a.pipeline)
    a.work.mkdir(parents=True, exist_ok=True)
    arms = {r: score_rows(load_hyps(fetch(r, a.work))) for r in runs}
    ids = set.intersection(*(set(s) for s in arms.values()))
    arms = {r: {i: v for i, v in s.items() if i in ids} for r, s in arms.items()}
    text, summary = report(arms, a.control)
    print(text)
    if a.out:
        a.out.write_text(text + "\n")
    if a.json:
        a.json.write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
