"""Dry run for an app-side number re-check: does the acoustic model prefer a close alternative reading?

For each clip: greedy-decode with the candidate model, generate one-edit alternatives of every spoken number
(teen<->ty, insert/delete "point", garbled number word -> nearest number word, "eine"/"ein" -> "eleven"/"one"), and
score the whole clip transcript under the TDT loss (sum of -log p, sigma 0) for the candidate AND stock decoder+joint
(the encoder is shared: frozen in the fine-tune). Choosing a reading is done offline from the returned scores.
"""

import json
import os
import re



NUMS = ("zero oh one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
        "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million "
        "billion trillion").split()
SCALE_OR_UNIT = set("hundred thousand million billion trillion percent dollars dollar euros euro cents point".split())
SWAP = {"thirteen": "thirty", "fourteen": "forty", "fifteen": "fifty", "sixteen": "sixty", "seventeen": "seventy",
        "eighteen": "eighty", "nineteen": "ninety"}
SWAP.update({v: k for k, v in SWAP.items()})
FOREIGN = {"eine": ["eleven", "one"], "ein": ["one", "eleven"], "eins": ["one"]}


def _edit(a: str, b: str) -> int:
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def _core(w: str) -> str:
    return re.sub(r"[^a-z]", "", w.lower())


def _recase(orig: str, new: str) -> str:
    m = re.match(r"^([^A-Za-z]*)([A-Za-z\-']+)([^A-Za-z]*)$", orig)
    if not m:
        return new
    pre, word, post = m.groups()
    if word[:1].isupper():
        new = new[:1].upper() + new[1:]
    return pre + new + post


def alternatives(text: str) -> list[dict]:
    words = text.split()
    cores = [_core(w) for w in words]
    numeric = [c in NUMS or c in SCALE_OR_UNIT for c in cores]
    out = []

    def near_number(i):
        return any(0 <= k < len(words) and numeric[k] for k in (i - 1, i + 1))

    for i, c in enumerate(cores):
        if c in SWAP:
            out.append({"i": i, "kind": "teen_ty", "new": SWAP[c]})
        if c in FOREIGN and near_number(i):
            for n in FOREIGN[c]:
                out.append({"i": i, "kind": "foreign", "new": n})
        if c and c not in NUMS and c not in SCALE_OR_UNIT and c not in FOREIGN and len(c) >= 4 and near_number(i):
            for n in NUMS:
                if abs(len(n) - len(c)) <= 2 and _edit(c, n) <= min(2, len(c) * 0.4):
                    out.append({"i": i, "kind": "garble", "new": n})
        # a spoken number starts here (not continuing one): try a "point" before it
        if c in NUMS and c not in ("hundred", "thousand", "million", "billion", "trillion") and \
                (i == 0 or not numeric[i - 1]):
            out.append({"i": i, "kind": "point_ins", "new": "point " + c})
        if c == "point":
            out.append({"i": i, "kind": "point_del", "new": ""})
    for a in out:
        w = list(words)
        w[a["i"]] = _recase(words[a["i"]], a["new"]) if a["new"] else ""
        a["text"] = " ".join(x for x in w if x)
        a["old"] = words[a["i"]]
    return out


def run(vol: str, cand_run: str, ultra: str, clips_dir: str, out_path: str) -> None:
    import torch
    import librosa
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.losses.rnnt import RNNTLoss

    import ft_eval

    cand = ft_eval.load_model(vol, cand_run, ultra)
    stock = nemo_asr.models.ASRModel.restore_from(ultra, map_location="cpu").cuda().eval()
    durations = list(cand.cfg.model_defaults.tdt_durations)
    loss_fn = RNNTLoss(num_classes=cand.joint.num_classes_with_blank - 1, reduction="sum", loss_name="tdt",
                       loss_kwargs={"durations": durations, "sigma": 0.0, "omega": 0.0})
    paths = sorted(os.path.join(clips_dir, f) for f in os.listdir(clips_dir) if f.endswith(".wav"))
    base = ft_eval.decode(cand, paths, batch_size=16)
    res = {}
    with torch.no_grad():
        for p, b in zip(paths, base):
            x, _ = librosa.load(p, sr=16000)
            sig = torch.tensor(x, device="cuda")[None]
            sig_len = torch.tensor([sig.shape[1]], device="cuda")
            feats, feats_len = cand.preprocessor(input_signal=sig, length=sig_len)
            enc, enc_len = cand.encoder(audio_signal=feats, length=feats_len)
            f = enc.transpose(1, 2)

            def nll(m, text):
                ids = m.tokenizer.text_to_ids(text)
                if not ids:
                    return None
                tok = torch.tensor([ids], device="cuda")
                tl = torch.tensor([len(ids)], device="cuda")
                g, _, _ = m.decoder(targets=tok, target_length=tl)
                logits = m.joint.joint(f, g.transpose(1, 2)).float()
                return float(loss_fn(log_probs=logits, targets=tok, input_lengths=enc_len, target_lengths=tl))

            alts = alternatives(b["text"])
            row = {"base": b["text"], "base_cand": nll(cand, b["text"]), "base_stock": nll(stock, b["text"]), "alts": []}
            for a in alts:
                a["cand"], a["stock"] = nll(cand, a["text"]), nll(stock, a["text"])
                row["alts"].append(a)
            res[os.path.basename(p)[:-4]] = row
    json.dump(res, open(out_path, "w"), indent=1)
    print(f"{len(res)} clips, {sum(len(r['alts']) for r in res.values())} alternatives")
