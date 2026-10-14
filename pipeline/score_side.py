# per-side year-reading accuracy over existing hyps (review sim; $0)
import json,re,collections,sys,os
U="one|two|three|four|five|six|seven|eight|nine"
T=rf"(ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty(?: (?:{U}))?|thirty(?: (?:{U}))?|oh (?:{U})|{U})"
END=rf"(?!(?:{U}|hundred|thousand|million|billion|point|and)\b)"
def norm(t): return " "+" ".join(re.sub(r"[^a-z' ]"," ",t.lower().replace("-"," ")).split())+" "
def spans(t):
    out=[]
    for m in re.finditer(rf" (two thousand(?: and)?|twenty) {T} {END}",norm(t)):
        cls="2k" if m.group(1).startswith("two") else "tw"; tail=m.group(2)
        if (tail in U.split("|") or tail.startswith("oh")): continue
        out.append((cls,tail))
    return out
root,runs,beds=sys.argv[1],sys.argv[2].split(","),sys.argv[3].split(",")
for bed in beds:
  for run in runs:
    p=f"{root}/{run}/{bed}.jsonl"
    if not os.path.exists(p): continue
    c=collections.Counter()
    for l in open(p):
        r=json.loads(l)
        if bed=="adv" and "__en_" not in r["id"]: continue
        hs=norm(r['hyp'])
        for cls,tail in spans(r['ref'].split('\t')[0] if bed!="fast" else r['ref'].split('\t')[1] if r['ref'].count('\t')>=1 and not r['ref'].split('\t')[1].isdigit() else r['ref'].split('\t')[0]):
            right=(rf" two thousand (?:and )?{tail} {END}" if cls=="2k" else rf" twenty {tail} {END}")
            other=(rf" twenty {tail} {END}" if cls=="2k" else rf" two thousand (?:and )?{tail} {END}")
            k="R" if re.search(right,hs) else "O" if re.search(other,hs) else "N"
            c[cls+k]+=1
    tot2=c['2kR']+c['2kO']+c['2kN']; totw=c['twR']+c['twO']+c['twN']
    print(f"{bed:10s} {run:24s} 2k R/O/N {c['2kR']}/{c['2kO']}/{c['2kN']} of {tot2}   tw R/O/N {c['twR']}/{c['twO']}/{c['twN']} of {totw}")
