# Audio tools: clip edges and copy-count gate

Libraries with tests (plain `python3 -I`, no pytest); no audio files and no network.

| File | What it does |
|---|---|
| `boundary_cut.py` | One edge-placement rule for clips cut out of longer audio, given word times. Every edge is cut at a word gap; a word the nominal edge would clip leaves both the audio and the label; a number cut mid-word excludes the clip. `place_edge`, `cut_window(words, total, a_nom, b_nom, search=...)`, `seam_pair`, `edge_status` (audit stored edges), `forced_share`. The window length is the caller's (`a_nom` / `b_nom`); `search` (seconds, default 2.0) is how far an edge may move inward. Constants (`MIN_GAP` 0.10 s, `MIN_PAD` 0.06 s, `FORCED_SILENCE` 0.08 s) are module defaults and keyword arguments. Edge kinds: `bound`, `forced`, `valley`, `dip`. Exclusion reasons: `number_mid_word`, `no_gap_in_range`, `no_dip`, `rejected`, `too_short`, `too_few_words`. |
| `edge_rule.py` | The valley rule of a windowed streaming decoder, in numpy: `Track(x, vad, frame=1280, vad_chunk=4096, sr=16000)` holds per-frame energy and VAD chunk probabilities; `snap(track, lo_s, hi_s)` picks the lowest-energy frame of the widest VAD valley; `snap_with_dip` applies the minimum-dip rule (cut inside continuous speech only if the cut is at least `DIP_DB` below the local speech level, else exclude). `vad_chunk_probs(x)` needs `torch` and the `silero-vad` package. With a `track`, `boundary_cut` uses the same rule to place a cut inside a word gap and to rescue clips with no gap. |
| `copycount_gate.py` | A gate for synthetic speech that catches a collapsed or added repeat copy. `variants()` builds the one-copy-fewer / one-copy-more texts of every repeat unit (`double five` <-> `five` / `triple five`; a period `four two` <-> `four two four two`); `probe(nll, words, units, ...)` fails a slot when a variant is not at least `FLOOR` (3.0) CTC nats worse than the voiced text and marks it uncertain between the floor and the calibrated margin; `calibrate_margin(contrasts)`, `copy_durations_ok`, `onset_between` (energy dip between two plain copies). The scorer is any callable `nll(text) -> nats`. |

Collapse policy the gate is built for: never drop a collapsed slot silently and never relabel it to the fewer-copies reading; re-voice it, and if it still fails keep it as a flagged stratum that is scored apart. Only "more copies or different digits than the script" counts as a listener dispute (ASR systems dedupe repeats, so they never vote on a copy count).

```
python3 -I copycount_gate.py                          # selftest with a fake scorer
python3 -I copycount_gate.py calibrate contrasts.json --pct 5    # json list of NLL(false variant) - NLL(true text) on ordinary chunks of one engine
sh run_tests.sh                                       # all four test files
```

Environment (names only): `SILERO_PYLIB` (optional directory holding the `silero_vad` package, if it is not installed).
