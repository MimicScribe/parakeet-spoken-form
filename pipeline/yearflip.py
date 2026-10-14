"""Year-reading flips across overlapping app windows (e21_dev, 3 phases): the precursor of the app's "twenty thousand
N" blends. The seam merge splices one window's "twenty" onto another's "thousand twenty" when two windows that hear the
same year read it differently (rt2 4366893: "two thousand twenty" vs "twenty twenty" for one "2020").

Key = call + the year's position in call time + its value (window-onset years count too: R3.23, the external review
found the old 2-word prefix hid every flip at a window start, the blend geometry); a flip = one key read both as "two thousand X" and "twenty X".
"""
import collections
import re

_U = "one|two|three|four|five|six|seven|eight|nine"
_T = "ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
TENS = {"twenty": 20, "thirty": 30}
SMALL = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
YEAR = re.compile(rf"(?P<pre>(?:\S+\s+){{0,2}})(?P<y>two thousand (?:and )?(?P<a>{_T}|{_U}|twenty(?: (?:{_U}))?)|twenty (?P<b>{_T}|twenty(?: (?:{_U}))?))\b",
                  re.I)


def _val(s: str) -> int:
    w = s.lower().split()
    return sum(SMALL.get(x, TENS.get(x, 0)) for x in w)


def flips(rows, field: str, tol: float = 2.0) -> dict:
    """Occurrences placed in call time (window start + char position x window length); readings of one value within
    `tol` seconds of each other in different windows are the same spoken year."""
    occ = collections.defaultdict(list)
    for r in rows:
        t = re.sub(r"[^\w' ]", " ", r[field]).lower()
        for m in YEAR.finditer(t):
            v = 2000 + _val(m.group("a") or m.group("b"))
            at = r["start"] + r["duration"] * m.start("y") / max(len(t), 1)
            onset = len(m.group("pre").split()) <= 1  # the year opens the window: the seam-blend geometry
            occ[(r["call"], v)].append((at, "2k" if m.group("a") else "20", r["id"], onset))
    n_fl, ex, n_occ, n_on = 0, [], 0, 0
    for (call, v), xs in occ.items():
        xs.sort()
        used = [False] * len(xs)
        for i in range(len(xs)):
            if used[i]:
                continue
            grp = [j for j in range(i, len(xs)) if xs[j][0] - xs[i][0] <= tol]
            for j in grp:
                used[j] = True
            n_occ += 1
            if len({xs[j][1] for j in grp}) > 1:
                n_fl += 1
                # blend precursor: an onset window reads "two thousand X" while another window reads "twenty X"
                n_on += any(xs[j][1] == "2k" and xs[j][3] for j in grp)
                if len(ex) < 12:
                    ex.append((call, v, round(xs[i][0], 1), sorted({xs[j][1] for j in grp})))
    return {"year_occurrences": n_occ, "flips": n_fl, "onset_2k_flips": n_on, "examples": ex}


if __name__ == "__main__":
    import json
    import sys

    rows = [json.loads(line) for line in open(sys.argv[1])]
    for f in sys.argv[2].split(","):
        print(f, json.dumps(flips(rows, f)))
