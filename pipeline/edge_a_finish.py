"""Edge pack, tier A: from checked clips (edge_a_ctc rows) and the blind-transcript verdicts to the set's final rows.

A clip is kept when (1) the CTC check passed (CER, every reading heard, every probe on every edge phrase) and (2) at least one
of the two blind transcripts says the edge phrase as labelled; a clip the CTC rejects but BOTH transcripts accept is listed
apart (`second_look`, never trained on). Rows whose construction or voice group failed the pilot gate are dropped by the
caller (`drop_constructions`, `drop_groups`). Output rows carry the fields of the other synthetic sets plus the edge fields
(`edge`, `edge_spans`, `ambig`, `construction`, `values`, `half`, `group`).
"""
import collections
import json

KEEP_FIELDS = ("audio_filepath", "duration", "text", "lang", "src", "id", "origin", "caption", "kind", "cls", "spans", "num_chars",
               "word_times", "cer_ctc", "fit", "edge_quiet", "split", "voice", "voice_license", "speed", "engine", "engine_src", "ckey",
               "register", "slots", "gclasses", "template", "edge", "edge_spans", "ambig", "construction", "constructions", "values", "half",
               "family", "group", "accent", "locale", "side", "label_style")


def finalize(rows: list, verdicts: dict, drop_constructions=(), drop_groups=(), set_tag: str = "ttsEC1") -> tuple:
    """rows: edge_a_ctc output rows; verdicts: {id: {"a": bool|None, "b": bool|None}} (blind transcript A / B said the phrase).
    Returns (final rows, second-look rows, report)."""
    st = collections.Counter()
    keep, second = [], []
    for r in rows:
        st["in"] += 1
        if r["construction"] in drop_constructions or r.get("group") in drop_groups:
            st["dropped_by_pilot_gate"] += 1
            continue
        v = verdicts.get(r["id"], {})
        va, vb = v.get("a"), v.get("b")
        said_any = bool(va) or bool(vb)
        said_both = bool(va) and bool(vb)
        if r.get("ctc_ok") and said_any:
            x = {k: r[k] for k in KEEP_FIELDS if k in r}
            x.update(origin=f"{set_tag}/{r.get('voice')}", src="tts", blind=[bool(va), bool(vb)])
            keep.append(x)
            st["kept"] += 1
        elif not r.get("ctc_ok") and said_both and r.get("basic_ok"):
            second.append(dict(r, blind=[True, True]))
            st["second_look"] += 1
        else:
            st["ctc_fail" if not r.get("ctc_ok") else "blind_fail"] += 1
    return keep, second, dict(st)


def tally(rows: list) -> dict:
    """Rows / hours per language x family x construction and per voice group (for the report)."""
    out = {"by_family": collections.Counter(), "by_construction": collections.Counter(), "by_group": collections.Counter(),
           "hours": round(sum(r["duration"] for r in rows) / 3600, 3), "rows": len(rows)}
    for r in rows:
        out["by_family"][r["family"]] += 1
        out["by_construction"][r["construction"]] += 1
        out["by_group"][r.get("group")] += 1
    return {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in out.items()}


def marker(set_tag: str, lang: str, rows: int, judges: str) -> dict:
    """The judge marker a set needs beside its rows (the format of ml_tts_pack.marker, with the checks of this pack named)."""
    return {"lang": lang, "set": set_tag, "rows": rows, "judge": judges,
            "chunks": [set_tag + "c"], "date": "2026-10-19"}
