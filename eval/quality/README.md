# eval/quality

Data-quality and scoring tools. Python 3, stdlib only for the tests (the number-word oracle test needs `num2words` and is skipped without it; labelling and upload steps need the Modal CLI and the volume).

| file | role |
|---|---|
| `data_gate.py` | census and bar per language x class (`--tier`) |
| `carve_heldout.py` | carve rare-class held-out lists by whole session or video |
| `score_heldout.py` | four-outcome held-out scoring |
| `number_outcomes.py` | number reader and scorer (CORE block identical to `pipeline/score_retrain.py`) |
| `judge_reject.py`, `judge_plro.py` | row judges against a second transcript |
| `wordfix_ext.py` | interior word repair from agreeing sources |
| `offset_score.py`, `score_values.py` | seam scorers over decoder dumps (see CONCEPTS.md) |

Run all tests: `cd eval/quality && for t in test_*.py; do python3 -m unittest $t; done` (`test_judge_plro.py` is a script: `python3 test_judge_plro.py`).
Tools find `pipeline/` relative to the repo; override with `SPOKEN_FORM_PIPELINE`, `SPOKEN_FORM_JUDGE_ROOT`, `JUDGE_WORK`.
