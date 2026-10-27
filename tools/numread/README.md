# numread: one number reader for scoring

Scores a hypothesis against a reference by meaning: digits vs words do not matter, a dropped, mangled or changed number does.

| File | What it does |
|---|---|
| `numread.py` | `reads(text, lang)`, `want_from(reading, text, lang)`, `classify(want, hyp, lang, refcount)` → `ok / dup / split / embedded / trunc / wrong / lost / unscorable`, `repeat_count(text, unit, lang)`, `normalize(text, lang)`, `stamp_line()`. `install()` verifies the pinned files, then makes `import number_outcomes` / `score_values` resolve to them. |
| `fair_reader.py` | The normaliser: thousands marks per language, spoken decimals, digit + scale words, spoken digit runs, `twenty o five`. Strict by default; `lenient=True` adds colloquial readings (`one fifty eight` → 158), always reported next to the strict result. |
| `reader/` | The base reader tree (`eval/quality/number_outcomes.py`, `score_values.py`, `pipeline/group_read.py`). Together with `pipeline/ml_*` (word-form tables for the non-core languages) this is the reader the repo's scorers use; it is vendored here at the exact version the pins name. |
| `numread_pins.json` | sha256 of every file above plus the `pipeline/ml_*` tables and `fair_reader.py`. A changed file refuses to load (`PinError`). Change a pin only after re-running the golden file and showing old vs new totals. |
| `numread_golden.jsonl`, `numread_golden.py` | 239 golden pairs: `reads` (text → values), `acc` (key vs hypothesis → outcome, 55 negatives such as `in twenty five` ≠ 2005, `5` ≠ `50` ≠ `500`), `rep` (exact repeat count). Must pass 100 %. |

The accept set comes from the spoken words, never from a digit string alone, and has no trailing-zero tolerance.
`split` means adjacent reads (same clause, no word between) whose digits concatenate to the wanted value.

Environment (names only): `NUMREAD_BASE` (reader tree; default `./reader`), `SPOKEN_FORM_PIPELINE` (the `ml_*` tables; default `../../pipeline`).

```
python3 -I numread_golden.py [-v]     # golden pairs, prints the reader stamp, exit 1 on any miss
python3 -I test_numread.py            # pins, install, golden, mutations that must turn the golden file red, edge cases
python3 -I test_fair_reader.py        # the normaliser
python3 numread.py --lang en "since twenty o five"
```

Licence: all files are this repo's own code (Apache-2.0).
