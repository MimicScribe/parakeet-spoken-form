"""English parliament clips (NVIDIA Granary VoxPopuli pseudo-labels, CC-BY-4.0; audio = VoxPopuli unlabelled, CC0).

Local, CPU only. Selects clips from the English Granary-VoxPopuli text index (ml_p3.granvox_index --langs en ->
/vol/ml/granvox/en/index.jsonl) by number VALUE and class for variety, the same way ml_p3.granvox_select_lang does for
the other languages, plus two English-only rules:
  - the English labeller must be able to label the caption (labeller.row_ok, >= 1 number span), since English clips
    are labelled by label_gpu (the D1 path), not label2;
  - held-out days: every plenary day of the VoxPopuli English dev/test split (and any extra list) is never selected.

    python vox_en.py select <index.jsonl> <heldout_days.txt> <out_dir> [--total 3200]
        -> <out_dir>/ids_by_year.json (input of ml_p3.granvox_cut_ids) + selection.jsonl + select_stats.json
    python vox_en.py manifests <cands_<tag>.jsonl> <out_dir> [--per 500]
        -> label_gpu manifests (one per 500 clips) for modal_app.s3_label --lang en_vox
"""

import collections
import json
import os
import random
import re
import sys

YEAR_TARGETS = (2030, 2035, 2040, 2050)
MONTHS = r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"
DATE_RE = re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?" + MONTHS + r"\b|\b" + MONTHS + r"\s+\d{1,2}(?:st|nd|rd|th)?\b",
                     re.I)
QUOTAS = {"clock": 450, "cur": 600, "dec": 400, "date": 300, "pct": 250, "big": 250, "range": 100}


def classes_en(text: str, gv_classes) -> dict:
    """ml_p3._gv_classes(text, "en") plus an English day-month date flag ("22nd November", "May 30th")."""
    return {**gv_classes(text, "en"), "date": bool(DATE_RE.search(text))}


def template(text: str) -> str:
    """Masked-digit sentence pattern (as granvox_select_lang): at most `per_template` clips share one."""
    return re.sub(r"\d+", "#", re.sub(r"[^\w\s]", "", text.lower()))[:60]


def session(utt_id: str) -> str:
    return utt_id.rsplit("_", 1)[0]


def est_seconds(text: str) -> float:
    return len(text) / 15


def labelable(text: str) -> bool:
    import labeller as L

    sp = L.spans(text)
    return bool(sp) and not L.row_ok(text, sp)


def select(rows: list, cls_of, held: set, seed: int = 1, per_year: int = 40, per_forward: int = 10,
           per_target: int = 50, per_old: int = 3, quotas: dict | None = None, per_template: int = 3,
           per_session: int = 8, total_cap: int = 3200, min_s: float = 3.0, max_s: float = 30.0,
           ok=labelable) -> tuple[list, dict]:
    """Year values first (rarest value first so common ones never crowd them out), then class quotas; caps per
    template / session / total; never a held-out day. Returns (selected rows with "why", stats)."""
    quotas = quotas or dict(QUOTAS)
    rows = [r for r in rows if r["day"] not in held and min_s <= est_seconds(r["text"]) <= max_s]
    rng = random.Random(f"{seed}-en")
    rng.shuffle(rows)
    cls = {r["utt_id"]: cls_of(r["text"]) for r in rows}
    t_n, s_n, taken, sel = collections.Counter(), collections.Counter(), set(), []
    ok_cache = {}

    def take(r, why):
        u = r["utt_id"]
        if u in taken or t_n[template(r["text"])] >= per_template or s_n[session(u)] >= per_session or len(sel) >= total_cap:
            return False
        if u not in ok_cache:
            ok_cache[u] = ok(r["text"])
        if not ok_cache[u]:
            return False
        taken.add(u)
        t_n[template(r["text"])] += 1
        s_n[session(u)] += 1
        sel.append(dict(r, why=why))
        return True

    want = {y: per_old if y < 1990 else per_year if y <= 2030 else per_forward for y in range(1900, 2061)}
    for y in YEAR_TARGETS:
        want[y] = max(want[y], per_target)
    by_val = collections.defaultdict(list)
    for r in rows:
        for y in set(cls[r["utt_id"]]["years"]):
            by_val[y].append(r)
    got = collections.Counter()
    for y in sorted(want, key=lambda y: len(by_val[y])):
        for r in by_val[y]:
            if got[y] >= want[y]:
                break
            if take(r, f"year:{y}"):
                for yy in set(cls[r["utt_id"]]["years"]):
                    got[yy] += 1
    have = collections.Counter(k for r in sel for k in quotas if cls[r["utt_id"]][k])
    for k in sorted(quotas, key=lambda k: sum(cls[r["utt_id"]][k] for r in rows)):
        for r in rows:
            if have[k] >= quotas[k]:
                break
            if cls[r["utt_id"]][k] and take(r, k):
                for kk in quotas:
                    have[kk] += bool(cls[r["utt_id"]][kk])
    yv = collections.Counter(y for r in sel for y in cls[r["utt_id"]]["years"])
    tot = sum(yv.values()) or 1
    st = {"clips": len(sel), "hours_est": round(sum(est_seconds(r["text"]) for r in sel) / 3600, 2),
          "year_spans": sum(yv.values()), "year_values": len(yv),
          "y1990_2030_ge40": sum(yv[y] >= 40 for y in range(1990, 2031)),
          "y1990_2030_min": min(yv[y] for y in range(1990, 2031)),
          "forward_2031_2060_values": sum(1 for y in yv if 2031 <= y <= 2060),
          "forward_spans": sum(c for y, c in yv.items() if 2031 <= y <= 2099),
          "targets": {y: yv[y] for y in YEAR_TARGETS},
          "top_value_share": round(max(yv.values()) / tot, 3) if yv else 0,
          "top5_share": round(sum(c for _, c in yv.most_common(5)) / tot, 3), "classes": dict(have),
          "sessions": len(s_n), "max_per_session": max(s_n.values()) if s_n else 0,
          "max_per_template": max(t_n.values()) if t_n else 0,
          "debate_years": dict(sorted(collections.Counter(r["day"][:4] for r in sel).items())),
          "index_avail": {k: sum(cls[r["utt_id"]][k] for r in rows) for k in quotas},
          "held_days": len(held), "rows_after_held_and_length": len(rows),
          "labelable_checked": len(ok_cache), "labelable_rate": round(sum(ok_cache.values()) / max(len(ok_cache), 1), 3)}
    return sel, st


def manifests(cands: list, per: int = 500) -> list:
    """label_gpu manifests from granvox_cut rows: the session is the group ("video") unit, kind 'digit'."""
    out = []
    for i in range(0, len(cands), per):
        part = []
        for r in cands[i:i + per]:
            part.append({"utt_id": r["id"], "duration": r["duration"], "lang": "<en>", "text": r["text"],
                         "original_audio_id": session(r["id"]), "original_audio_offset": 0.0, "kind": "digit",
                         "shard": "granary/voxpopuli/en", "audio_filepath": r["audio_filepath"]})
        out.append(part)
    return out


def _main(argv):
    if argv[0] == "select":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from ml_p3 import _gv_classes

        idx, held_path, out = argv[1:4]
        total = int(argv[argv.index("--total") + 1]) if "--total" in argv else 3200
        rows = [json.loads(l) for l in open(idx) if l.strip()]
        held = {m.group(1) for l in open(held_path) if (m := re.match(r"(?:# group vox:)?(\d{8})", l.strip()))}
        assert held, f"{held_path}: no held-out days parsed"  # an empty list would let eval days into training
        sel, st = select(rows, lambda t: classes_en(t, _gv_classes), held, total_cap=total)
        os.makedirs(out, exist_ok=True)
        by_year = collections.defaultdict(list)
        for r in sel:
            by_year[r["day"][:4]].append(r["utt_id"])
        json.dump(dict(sorted(by_year.items())), open(f"{out}/ids_by_year.json", "w"))
        with open(f"{out}/selection.jsonl", "w") as f:
            for r in sel:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        json.dump(st, open(f"{out}/select_stats.json", "w"), indent=1)
        print(json.dumps(st))
    elif argv[0] == "manifests":
        cands = [json.loads(l) for l in open(argv[1]) if l.strip()]
        out = argv[2]
        per = int(argv[argv.index("--per") + 1]) if "--per" in argv else 500
        os.makedirs(out, exist_ok=True)
        for i, part in enumerate(manifests(cands, per)):
            with open(f"{out}/vox_en_{i:03d}.jsonl", "w") as f:
                for r in part:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{len(cands)} clips -> {(len(cands) + per - 1) // per} manifests")


if __name__ == "__main__":
    _main(sys.argv[1:])
