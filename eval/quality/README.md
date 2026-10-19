# eval/quality

Data-quality and scoring tools. Python 3, stdlib only for the tests (the number-word oracle test needs `num2words` and is skipped without it; labelling and upload steps need the Modal CLI and the volume).

| file | role |
|---|---|
| `data_gate.py` | census and bar per language x class (`--tier`) |
| `carve_heldout.py` | carve rare-class held-out lists by whole session or video |
| `score_heldout.py` | four-outcome held-out scoring |
| `classread.py` | per language x class readout of one or more arms with paired cluster-bootstrap differences. Default: the symmetric table W (written in words), V (value right in either form), U_w / U_d (spoken unit or decimal word missing, among spans written in words / digits); `--legacy` prints the old asymmetric table. Needs `numpy` |
| `symscore.py` | the symmetric value / unit / decimal scorer behind `classread.py` and `score_heldout.py` (digits and words read by one reader and graded by one procedure; `score_heldout.py --legacy` leaves it out) |
| `number_outcomes.py` | number reader and scorer (CORE block identical to `pipeline/score_retrain.py`) |
| `judge_reject.py`, `judge_plro.py` | row judges against a second transcript |
| `wordfix_ext.py` | interior word repair from agreeing sources |
| `seam_gate.py` | seam veto vs a baseline: offset score over decoder dumps (fails on a missing scope, language or slot); proxy gate and lost-word rate are report-only |
| `offset_score.py`, `score_values.py` | seam scorers over decoder dumps (see CONCEPTS.md) |
| `false_conversion.py` | false-conversion gate: spurious digits / number words per 1,000 words on number-free speech, paired vs a base arm (stock) |

## Tests

Stdlib only, except `test_number_outcomes.py` (an oracle test that needs `num2words`; skipped without it). The tests import `pipeline/` modules and `ml_grammar_table.json`; no Modal, GPU or network. Conventions are mixed, so run each file the way it is written:

```bash
cd eval/quality
for t in test_*.py; do [ "$t" = test_judge_plro.py ] || python3 -m unittest "${t%.py}"; done
python3 test_judge_plro.py        # a plain script that runs judge_plro.py on a temp pass
```

`python3 -m pytest eval/quality` also works if pytest is installed (not needed otherwise).

## Environment overrides

| Variable | Read by | Meaning |
|---|---|---|
| `SPOKEN_FORM_PIPELINE` | `number_outcomes.py`, `judge_reject.py` | Path to `pipeline/` (default: the repo's own) |
| `SPOKEN_FORM_JUDGE_ROOT` | `judge_plro.py` | Judge work root, `<root>/<lang>/<pass>/` (default `/tmp/spoken_form_judge`) |
| `JUDGE_WORK` | `wordfix_ext.py` | Work dir (default `/tmp/spoken_form_judge`) |
| `PROBE_WORK` | `score_heldout.py` | Default for `--work` (default `/tmp/spoken_form_probe`) |
| `ML_CANDS_OFF`, `ML_CANDS_ON`, `ML_DECIDE_OFF` | `pipeline/ml_cands.py`, `ml_decide.py` (imported by the judges and scorers) | Comma lists that switch individual candidate or decision rules off or on, for ablations |

`judge_plro.py --upload` also calls the `modal` CLI to read and write the volume; nothing else here needs credentials.

## Seam-scorer files: `cases_<arm>.jsonl` and `parse_out_<arm>.jsonl`

`offset_score.py <workdir> <arm>...` and `score_values.py <cases> <parse_out>` read two files per arm, produced by the streaming harness (not in this repo).

`cases_<arm>.jsonl`, one row per sentence per window offset:

| field | meaning |
|---|---|
| `id` | sentence id, `<lang>_<n>`; the same id at each offset (the part before `_` is the language) |
| `lang` | language code |
| `values` | expected values by slot, `{slot: value}` (a string) |
| `long` | text decoded from the long (stitched) file at this offset |
| `single` | text decoded from the sentence alone |
| `same` | true if `long` and `single` are identical (skipped by `score_values.py`'s category counts) |
| `merge_gaps` | list of `{pick, A, B, R}`, the places where the two windows disagreed and the merge chose; empty means the break is not merge-touched |
| `file` | the stitched file the row came from |

`parse_out_<arm>.jsonl`, one row per decoded text: `key` is `"<row index in cases>:long"` or `"<row index>:single"`; `values` is the list of number strings the parser read from that text (greedy reading); `alt` is an optional second list for the same text (the no-connector reading). `parse_in_<arm>.jsonl` next to it (`key`, `lang`, `text`: the parser's input) adds a third reading, the public reader `number_outcomes.extract_numbers` on the same text; without it a warning is printed and only the two parser readings count. A slot counts as right if it matches in any reading.

The parse step turns decoded text into numbers with a number grammar that is **not in this repo**. Supply your own parser that writes this format.

**Slot matching (strict).** Slots are matched in order, and each parsed number is used once. A phone (`PH`) must be an exact digit run. Clock (`T`) and price (`P`) parts must be adjacent. Other slots match a parsed number equal to the expected value, or 2-4 adjacent parts split by one joining word that add up to it ("cincizeci și cinci" = 50 + 5). The headline is **total wrong** (wrong slot readings summed over offsets); seam-broken and always-wrong split it. Compare arms with a paired bootstrap over files (90% CI) and a sign test over slots. It is still a screen, not a proof.

Invented words and split-form numbers (R8 measurement):
- `invented_words.py` counts words that are in no lexicon of the language (references, stock/base decodes, training text, number morphemes): sentences with an invented token, invented tokens per converted number, invented rate by token length. Method in its docstring; `--help` for the parity-decode and app-path inputs.
- `number_outcomes.form_outcomes` calls a number `converted` only when it is written as words AND the value is right; a wrong or broken words chain is `words_wrong`, and the old count stays in `form_old` / `converted_old`.
