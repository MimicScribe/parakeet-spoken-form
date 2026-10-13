"""Put back the fillers and repetitions the YouTube captions dropped (step-1000 attribution, 2026-10-13: caption labels
cut "uh"/"um" output by 32% vs v10pt). Only INSERTIONS from v10pt's own decode of the same crop are taken, and only
when every inserted word is a filler or repeats an adjacent label word; the caption text (and its number readings)
is otherwise untouched."""
import difflib, json, re, sys, collections

from common import NUMBER_WORDS, SPAN_CORE, SPAN_LINK
FILL = {"uh", "um", "uhm", "er", "erm", "hmm", "mm", "mhm", "uh-huh", "mm-hmm", "ah"}
k = lambda w: re.sub(r"[^\w'-]", "", w.lower())

def merge(label, hyp):
    L, V = label.split(), hyp.split()
    Lk, Vk = [k(w) for w in L], [k(w) for w in V]
    out, n_ins = [], 0
    sm = difflib.SequenceMatcher(None, Lk, Vk, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "insert" and i1 == 0:  # never at the clip start ("Uh And then"): the next word's case is unknown
            continue
        if op == "insert":
            ins = Vk[j1:j2]
            nb = set(Lk[max(0, i1 - 2):i1 + 2])
            # repeats only of non-number words: a doubled number word in v10pt's decode is as likely its error as a stutter
            if all(w in FILL or (w in nb and not NUMBER_WORDS.fullmatch(w) and w not in SPAN_CORE
                                         and w not in SPAN_LINK) for w in ins) and len(ins) <= 4:
                words = [re.sub(r"[^\w'-]", "", w).lower() if k(w) in FILL else re.sub(r"[.?!,;:]+$", "", w) for w in V[j1:j2]]
                if out and not out[-1][-1:] in ".?!" or not out:
                    words = [w.lower() if k(w) in FILL else w for w in words]
                out += words
                n_ins += len(words)
            continue
        out += L[i1:i2]
    t = " ".join(out)
    if t and t[0].islower() and label[:1].isupper():
        t = t[0].upper() + t[1:]
    return t, n_ins

def main(S):
    dec = json.load(open(f"{S}/rt/mix/v10_decode.json"))
    stats = collections.Counter()
    new_text = {}
    for part in ("d1", "d1p"):
        rows = [json.loads(l) for l in open(f"{S}/rt/mix/{part}.jsonl")]
        for x in rows:
            h = dec.get(x["audio_filepath"])
            if h is None:
                stats[part + "_nodecode"] += 1
                continue
            t, n = merge(x["text"], h)
            if n:
                stats[part + "_rows_changed"] += 1
                stats[part + "_words_inserted"] += n
                new_text[(x.get("src", x["audio_filepath"]), x["text"])] = t
                x["text"] = t
        open(f"{S}/rt/mix/{part}.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    # speed copies carry their source's text
    by_text = {}
    for (src, old), t in new_text.items():
        by_text.setdefault(old, set()).add(t)
    for part in ("d10n", "d10p"):
        rows = [json.loads(l) for l in open(f"{S}/rt/mix/{part}.jsonl")]
        for x in rows:
            key = (x.get("src", ""), x["text"])
            if key in new_text:
                x["text"] = new_text[key]; stats[part + "_changed"] += 1
            elif len(by_text.get(x["text"], ())) == 1:
                x["text"] = next(iter(by_text[x["text"]])); stats[part + "_changed"] += 1
        open(f"{S}/rt/mix/{part}.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    print(dict(stats))


if __name__ == "__main__":
    main(sys.argv[1])
