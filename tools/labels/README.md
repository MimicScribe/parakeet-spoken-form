# Label tools: never-reject CTC labelling

Choose, for each number in a training label, the spoken reading the audio supports, without ever dropping a row. Every input row comes back with a status
(`scored`, `undecided`, `cer_high`, `align_fail`, `error`); below the calibrated margin the current label is kept and marked `unconfirmed`.

| File | What it does |
|---|---|
| `spanfind.py` | Words to number spans and candidate readings, no digit caption needed. `find_spans(label)` finds runs of number words (+ adjacent units / letter codes); `candidates(span)` lists spoken readings of the same value (cardinal, digit-by-digit with `oh` / `zero`, pairs, `double` / `triple`, years, `a hundred`) plus +-1 repeat-count variants. The current reading is always first; no commas, digits or hyphens. |
| `labeller_core.py` | The scoring logic, no torch. `score_row(row, loss_fn, free_hyp, align_fn)`: per span the CTC loss of every candidate in the whole-row context (two passes when a row has several spans), top-2 losses, margin over ALL candidates including same-key alternates (`oh` / `zero`), tie flag, current-reading rank, row-level alternative labels, CER gate, alignment failure as a status. Bit-vector Levenshtein. |
| `w2v_torch.py` | torch helpers: batched CTC losses of whole-row labels, greedy transcript, forced word alignment (torchaudio), onset. |
| `label_runner.py` | Threaded pipeline (load, forward, score, write in input order) and a GPU / CPU sampler line. Stdlib at import time. |
| `label_local.py` | Command-line driver: manifest in, scored jsonl out, resumable. Needs torch, transformers, soundfile, torchaudio and a wav2vec2 960h-family model. |
| `calibrate_margin.py` | `curve`: accuracy / coverage / Wilson bound vs margin on human-transcribed rows (where the current reading is the known answer), per alternative type, and the smallest threshold meeting the bars. `decide`: the fallback rule (accept above the threshold, else `unconfirmed`), optionally written to the ledger. |
| `tiering.py` | Which rows need a paid referee pass: tier 0 human transcripts, tier 1 free evidence suffices, tier 2 referee (undecided, low margin, tie, value mismatch, high-risk class), with an hours / cost estimate from the ledger. |
| `ledger.py` | The label ledger, one SQLite file: `rows`, `spans`, `candidates`, append-only `evidence` and `decisions` (triggers refuse update / delete), append-only `gold`, listening `queues`, views `latest_decision`, `row_state`, `review_page`. `export_mix` rewrites each part from the stored raw line plus the latest decision (byte-identical when none applies). |
| `load_listing.py` | Fill a ledger from listing jsonl files; part names, source classes and datasets come from a rules file (regex on the part name), never from the code. |
| `manifest.py` | Ledger rows to labeller manifest rows (spans recomputed on the exact label scored). |

Flow: `load_listing.py` (with `--spans`) -> `manifest.py` -> `label_local.py` -> `calibrate_margin.py curve` on human-transcribed rows -> `calibrate_margin.py decide --db ... --threshold T` -> `ledger.py export-mix`.

```
python3 load_listing.py --db ledger.db --listing 'listings/*.jsonl' --rules rules.json --spans
python3 manifest.py --db ledger.db --out manifest.jsonl --limit 200
python3 label_local.py --manifest manifest.jsonl --out scored.jsonl --threads 2
python3 calibrate_margin.py curve scored.jsonl --gold manifest_with_gold_form.jsonl
python3 calibrate_margin.py decide scored.jsonl --threshold 1.0 --db ledger.db
python3 tiering.py --db ledger.db --usd-per-min 0.004
python3 ledger.py --db ledger.db stats
```

Environment (names only): `LABELS_DB` (default for `--db`), `HF_HOME` (model cache). The model vocabulary must equal `labeller_core.VOCAB` (checked at start of `label_local.py`).
Models credited in the top-level README (wav2vec 2.0, Meta).

Tests (plain `python3 -I`, no pytest; `test_w2v_torch.py` skips without torch): `sh run_tests.sh`, or one file at a time, e.g. `python3 -I test_labeller_core.py`.
