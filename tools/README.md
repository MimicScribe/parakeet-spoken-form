# Tools

Generic data, label, audio and scoring tools. They hold no keys, no machine paths and no data; configuration is by flag or environment variable (names are listed in each README).
Every test runs with plain `python3 -I` (no pytest): `sh tools/run_tests.sh` runs them all.

| Directory | What it is |
|---|---|
| [`numbers/`](numbers/README.md) | Spoken-number briefs, carrier sentences, near-dup filter, census, label fixes, repeat markers, form-blind number scorer, exact repeat-count scorer, paired comparison. |
| [`numread/`](numread/README.md) | The pinned number reader and normaliser (sha256 pins, 239 golden pairs). |
| [`labels/`](labels/README.md) | Never-reject CTC labelling: number spans and candidate readings, whole-row scoring, margin calibration, tiering, label ledger. |
| [`audio/`](audio/README.md) | Clip-edge rule (word gaps, valley / dip selection) and the copy-count gate for synthetic speech. |
| [`quality/`](quality/README.md) | Near-dup and leakage check, variety gates, semantic paraphrase check. |

Optional third-party dependencies: `torch`, `transformers`, `soundfile`, `torchaudio` (`labels/label_local.py`), `torch` + `silero-vad` (`audio/edge_rule.vad_chunk_probs`), `numpy`
(`audio/`, `quality/`), `scipy` (resampling in `audio/edge_rule.load_audio`). Everything else is the standard library.
