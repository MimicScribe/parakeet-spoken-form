"""An LLM writes one carrier sentence per brief (Gemini Flash-Lite via GEMINI_API_KEY, or Claude Haiku via the `claude` CLI), batched.
Every required reading must appear VERBATIM; the clip-start cell must open with its number; rejects are retried once.
Output: sentences/<tag>.jsonl rows {id, cell, said, label_v3, numbers, setting, register, form}. Hard spend cap --max-usd.
  python3 gen_sentences.py --tag sample --max-usd 0.2"""
import argparse, json, subprocess, tempfile, re, os, sys, concurrent.futures as cf
SYSTEM = ("You write natural spoken English sentences for a speech dataset. For each brief, write ONE utterance exactly as a person "
          "would say it out loud, matching the setting, register, form and length. Every required number reading must appear "
          "EXACTLY as given (same words, same order, same commas), never rewritten as digits or reworded. Write all other words "
          "plainly; do not add any other numbers unless the brief says so. Vary sentence structure, vocabulary and openings across "
          "items; never reuse a sentence frame. Make every number mean what its brief says (a year is a year, an ID is an ID), so the sentence makes sense. Letters in codes stay as capital letters separated by spaces as given. Use normal English capitalization (sentence starts, names, places) and normal punctuation, as a careful transcriber would write the speech. Every utterance must be grammatical and something a real person would say; never string words together just to include them.")
SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}}, "required": ["id", "text"]}}}, "required": ["items"]}


def brief_text(b):
    nums = "; ".join(f'"{n["reading"]}" (it is {n["kind"]}; use it with that meaning)' for n in b["numbers"]) or "none"
    return (f'<BRIEF id="{b["id"]}">\nsetting: {b["setting"]}\nregister: {b["register"]}\nform: {b["form"]}\nlength: {b["length"]}\n'
            f'required readings, verbatim: {nums}\nposition of the first number: {b["position"]}\n'
            + ("include a short hesitation (um/uh) or restart somewhere\n" if b["disfluent"] else "")
            + ("mention a person's or place's name\n" if b.get("names_nearby") else "")
            + (f'use this person\'s name: {b["name"]}\n' if b.get("name") else "") + (f'mention this place: {b["place"]}\n' if b.get("place") else "")
            + (f'extra: {b["extra"]}\n' if b["extra"] else "") + "</BRIEF>")


GEM_MODEL = "gemini-3.5-flash-lite"
TOK = {"in": 0, "out": 0}


def _key():
    """GEMINI_API_KEY from the environment, else from a .env in the data dir or next to this file"""
    if os.environ.get("GEMINI_API_KEY"): return os.environ["GEMINI_API_KEY"]
    for p in (".env", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
        if os.path.exists(p):
            for l in open(p):
                if l.startswith("GEMINI_API_KEY="): return l.split("=", 1)[1].strip().strip('"')
    raise SystemExit("GEMINI_API_KEY not set (environment or .env)")


def call_gemini(batch):
    import urllib.request, time
    prompt = "Write one utterance per brief. Return every id.\n\n" + "\n\n".join(brief_text(b) for b in batch)
    body = {"systemInstruction": {"parts": [{"text": SYSTEM}]}, "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 1.0, "maxOutputTokens": 8192, "responseMimeType": "application/json",
                                 "responseSchema": {"type": "OBJECT", "properties": {"items": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {"id": {"type": "STRING"}, "text": {"type": "STRING"}}, "required": ["id", "text"]}}}, "required": ["items"]}}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEM_MODEL}:generateContent?key={_key()}"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            d = json.load(urllib.request.urlopen(req, timeout=180))
            u = d.get("usageMetadata", {}); TOK["in"] += u.get("promptTokenCount", 0); TOK["out"] += u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
            txt = d["candidates"][0]["content"]["parts"][0]["text"]
            return {x["id"]: x["text"] for x in json.loads(txt).get("items", [])}, 0.0
        except Exception:
            time.sleep(2 + 3 * attempt)
    return {}, 0.0


def call(batch):
    prompt = "Write one utterance per brief. Return every id.\n\n" + "\n\n".join(brief_text(b) for b in batch)
    with tempfile.TemporaryDirectory() as cwd:
        r = subprocess.run(["claude", "-p", "--model", "haiku", "--output-format", "json", "--json-schema", json.dumps(SCHEMA), "--system-prompt", SYSTEM,
                            "--strict-mcp-config", "--tools", "", "--disable-slash-commands"], input=prompt, capture_output=True, text=True, timeout=600, cwd=cwd)
    if r.returncode: return {}, 0.0
    try:
        env = json.loads(r.stdout); ans = env.get("structured_output") or json.loads(env.get("result", ""))
        return {x["id"]: x["text"] for x in ans.get("items", [])}, float(env.get("total_cost_usd") or 0)
    except Exception:
        return {}, 0.0


NUMW = set("zero oh one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen "
           "nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand double triple".split())


def _whole(reading, t):
    return list(re.finditer(r"(?<![A-Za-z'])" + re.escape(reading) + r"(?![A-Za-z'])", t, re.I))


def check(b, t):
    low = t.lower()
    for n in b["numbers"]:
        hits = _whole(n["reading"], t)
        if not hits: return False
        if n.get("label") and n["label"] != n["reading"]:  # converted reading: exactly once, no stray number word touching it
            if len(hits) != 1: return False
            a, z = hits[0].span(); bw = re.findall(r"[A-Za-z']+", t[:a])[-1:]; aw = re.findall(r"[A-Za-z']+", t[z:])[:1]
            if (bw and bw[0].lower() in NUMW) or (aw and aw[0].lower() in NUMW): return False
    pk = b.get("pos_key")
    if pk == "start" and b["numbers"] and not low.lstrip("\"' ").startswith(b["numbers"][0]["reading"].lower()): return False
    if pk == "end" and b["numbers"] and not low.rstrip(" .!?\"')").endswith(b["numbers"][-1]["reading"].lower()): return False
    if b.get("pos_key") == "none" and re.search(r"\b(" + "|".join(NUMW - {"oh"}) + r")\b", low): return False
    if b["cell"] == "clip_start" and b["numbers"] and not low.lstrip("\"' ").startswith(b["numbers"][0]["reading"].lower()): return False
    if b["cell"] == "clip_end" and b["numbers"] and not low.rstrip(" .!?\"')").endswith(b["numbers"][-1]["reading"].lower()): return False
    if b["cell"] == "no_number" and re.search(r"\d", t): return False
    if re.search(r"\d", t): return False  # the voice reads words; digits only appear in the label
    return True


def label(b, t):
    lab = t
    for n in b["numbers"]:
        if n.get("label") and n["label"] != n["reading"]:
            h = _whole(n["reading"], lab)
            if h: a, z = h[0].span(); lab = lab[:a] + n["label"] + lab[z:]
    return lab


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--max-usd", type=float, required=True); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--engine", default="gemini", choices=["gemini", "haiku"])
    ap.add_argument("--avoid", help="file with one over-used opening per line; the writer is told not to start with them")
    ap.add_argument("--root", default=".", help="data directory holding briefs/ (sentences/ is written here)")
    a = ap.parse_args(); os.chdir(a.root); fn = call_gemini if a.engine == "gemini" else call
    global SYSTEM
    if a.avoid:
        av = [l.strip() for l in open(a.avoid) if l.strip()]
        SYSTEM = SYSTEM + " Never start an utterance with any of these over-used openings: " + "; ".join(f'"{x}"' for x in av) + "."
    briefs = [json.loads(l) for l in open(f"briefs/{a.tag}.jsonl")]
    os.makedirs("sentences", exist_ok=True); out, spent, todo = {}, 0.0, briefs
    for attempt in range(2):
        batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]; bad = []
        with cf.ThreadPoolExecutor(a.workers) as ex:
            for batch, (got, cost) in zip(batches, ex.map(fn, batches)):
                spent += cost
                for b in batch:
                    t = got.get(b["id"])
                    if t: t = re.sub(r"[\[\]()]", "", t).strip(); t = t[:1].upper() + t[1:]
                    if t and check(b, t): out[b["id"]] = dict(id=b["id"], cell=b["cell"], said=t.strip(), label_v3=label(b, t.strip()), numbers=b["numbers"], setting=b["setting"], register=b["register"], form=b["form"])
                    else: bad.append(b)
                if spent > a.max_usd: print("STOP at cap", spent); break
        todo = bad
        if not todo or spent > a.max_usd: break
    open(f"sentences/{a.tag}.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out.values()))
    print(f"ok {len(out)} / {len(briefs)}  rejected {len(todo)}  spent ${spent:.3f}  gemini tokens in {TOK['in']} out {TOK['out']}")


if __name__ == "__main__": main()
