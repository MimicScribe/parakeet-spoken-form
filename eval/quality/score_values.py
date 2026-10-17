"""Value-level score. Usage: score_values.py <cases_RUN.jsonl> <parse_out_RUN.jsonl>"""
import json, collections, re, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
P = {}
for l in open(sys.argv[2]):
    if l.strip(): r = json.loads(l); P[r['key']] = (r['values'], r.get('alt', []))  # greedy + no-connector reading
def num(s):
    try: return float(s)
    except: return None
def slot_ok(slot, v, vals):
    nums = [num(x) for x in vals]; nums = [x for x in nums if x is not None]
    if slot == 'PH':
        joined = "".join(x.replace('.', '') for x in vals); return v in joined
    if slot == 'P':
        a = int(v); c = round((v - a) * 100)
        return any(abs(x - v) < 1e-6 for x in nums) or (a in nums and c in nums)
    if slot == 'T':
        h, m = v.split(':'); h, m = int(h), int(m)
        return (h in nums and (m == 0 or m in nums)) or any(abs(x - (h + m / 100)) < 1e-6 for x in nums) or (h * 100 + m) in nums
    return any(abs(x - float(v)) < 1e-6 for x in nums)
def slot_any(slot, v, readings):  # a slot value found in either reading of the same text
    return any(slot_ok(slot, v, x) for x in readings)
def correct(r, side):
    vals = P.get(f"{rows.index(r)}:{side}", ([],))
    return all(slot_any(k, v, vals) for k, v in r['values'].items())
c = collections.Counter(); ex = collections.defaultdict(list)
L = collections.defaultdict(collections.Counter)  # per language, over ALL sentences
for k, r in enumerate(rows):
    lv, sv = P.get(f"{k}:long", ([],)), P.get(f"{k}:single", ([],))
    lo = all(slot_any(s, v, lv) for s, v in r['values'].items()); so = all(slot_any(s, v, sv) for s, v in r['values'].items())
    t = L[r['lang']]; t['n'] += 1; t['long_ok'] += lo; t['single_ok'] += so; t['seam_wrong'] += so and not lo; t['seam_fixed'] += lo and not so
    for s, v in r['values'].items():
        sl, ss = slot_any(s, v, lv), slot_any(s, v, sv); t['n_' + s] += 1; t['long_ok_' + s] += sl; t['seam_wrong_' + s] += ss and not sl
    if r.get('same', False): continue  # the categories below are the differing readings only, as before

    lo = all(slot_any(s, v, lv) for s, v in r['values'].items()); so = all(slot_any(s, v, sv) for s, v in r['values'].items())
    m = bool(r['merge_gaps']); grp = 'en' if r['lang'] == 'en' else 'non-en'
    cls = ('form_only' if lo and so else 'LONG_WRONG_single_right' if so else 'long_right_single_wrong' if lo else 'both_wrong')
    key = f"{grp}|{cls}|{'merge' if m else 'nomerge'}"; c[key] += 1
    if cls == 'LONG_WRONG_single_right': ex[m].append(r)
for k in sorted(c): print(k, c[k])
SL = ['L', 'PH', 'P', 'T', 'Y', 'PCT', 'N', 'M', 'A', 'B']
print("\n== per language, all sentences: long-file value-right % | single % | seam-made wrong | seam-fixed | seam-made wrong by slot ==")
for lang in sorted(L):
    t = L[lang]; n = t['n']
    print(f"{lang} n={n} long={100 * t['long_ok'] / n:.1f}% single={100 * t['single_ok'] / n:.1f}% seam_wrong={t['seam_wrong']} seam_fixed={t['seam_fixed']} "
          + " ".join(f"{s}:{t['seam_wrong_' + s]}/{t['n_' + s]}" for s in SL if t['n_' + s]))
T = collections.Counter()
for t in L.values(): T.update(t)
print("ALL", f"n={T['n']} long={100 * T['long_ok'] / T['n']:.1f}% single={100 * T['single_ok'] / T['n']:.1f}% seam_wrong={T['seam_wrong']} seam_fixed={T['seam_fixed']}",
      " ".join(f"{s}:{T['seam_wrong_' + s]}/{T['n_' + s]}(long ok {100 * T['long_ok_' + s] / T['n_' + s]:.0f}%)" for s in SL if T['n_' + s]))
print("\n== long wrong, single right, merge-touched (first 25) ==")
for r in ex[True][:25]:
    print(r['file'], r['values'], '\n  L:', r['long'], '\n  S:', r['single'], '\n  gaps:', [(g['pick'], g['A'], g['B'], g['R']) for g in r['merge_gaps']])
