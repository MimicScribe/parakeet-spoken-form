"""Paired comparison of two arms' per-clip verdicts (scorer v2.1). Headline OK = ok + ok_point + ok_edge.
Usage: python3 paired.py verdicts_v21.json stock N2 [--video-of rows.json]
Clip ids look like 'en109_00000415_<videoid>_<...>'; video id = third '_' field (YouTube ids may contain '_', so
the fallback key is the id minus its trailing '_<start>_<n>_<h>_<m>_<s>_w<k>' pieces); pass video_of={id: video} to override.
Bootstrap: resample videos (clusters) with replacement."""
import json, random, sys, re
from math import comb
HEAD = {"ok", "ok_point", "ok_edge"}


def video_key(cid):
    """clip id = <lang><n>_<seq>_<11-char video id>_<start..>_wK -> the 11-char video id"""
    m = re.match(r"^[A-Za-z]+\d+_\d+_(.{11})_", cid)
    return m.group(1) if m else cid


def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def paired(va, vb, video_of=None, n_boot=2000, seed=1):
    """va (baseline), vb (arm): {clip: verdict}. Returns dict; diff in points = arm - baseline."""
    ids = sorted(set(va) & set(vb))
    A = {i: va[i] in HEAD for i in ids}; B = {i: vb[i] in HEAD for i in ids}
    fixed = sum(1 for i in ids if B[i] and not A[i]); broken = sum(1 for i in ids if A[i] and not B[i])
    by = {}
    for i in ids:
        v = (video_of or {}).get(i) or video_key(i)
        s = by.setdefault(v, [0, 0]); s[0] += B[i] - A[i]; s[1] += 1
    vs = list(by); rnd = random.Random(seed); out = []
    for _ in range(n_boot):
        s = c = 0
        for _ in vs:
            x = by[rnd.choice(vs)]; s += x[0]; c += x[1]
        out.append(100 * s / c)
    out.sort()
    return dict(n=len(ids), base_ok=sum(A.values()), arm_ok=sum(B.values()), fixed=fixed, broken=broken,
                diff_pts=100 * (fixed - broken) / len(ids), mcnemar_p=mcnemar_exact(fixed, broken),
                ci95=(out[int(.025 * n_boot)], out[int(.975 * n_boot)]), videos=len(vs))


if __name__ == "__main__":
    d = json.load(open(sys.argv[1])); d = d.get("new", d)
    a, b = sys.argv[2], sys.argv[3]
    print(a, "->", b, paired(d[a], d[b]))
