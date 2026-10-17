# Requirements

## Python

Python 3.10 or newer. The Modal images in `pipeline/modal_app.py` use 3.12, so 3.12 matches what the training and labelling jobs run on.

## Tests

The tests use mixed conventions: plain scripts (top-level asserts, run with `python3 test_x.py`), `unittest` files, and pytest-style files. Run `pipeline/` tests from `pipeline/` (they import their modules by bare name) and `eval/quality/` tests from there.

| Where | Needs | How to run |
|---|---|---|
| `eval/quality/test_*.py` | stdlib only; `num2words` optional (one test is skipped without it) | `python3 -m unittest test_x` per file; `test_judge_plro.py` is a script: `python3 test_judge_plro.py` |
| `pipeline/test_*.py`, most files | stdlib only | `python3 test_x.py`; the unittest files (`test_ml_multi`, `test_ml_multi_scale`, `test_ml_tight`, `test_ml_wordfix`, `test_ml_tts_cloud_class_n`) also run as `python3 -m unittest test_x` |
| `test_ml_p3`, `test_ml_p3_yield`, `test_ml_parl`, `test_mix_one_per_id` | `numpy`; `test_mix_one_per_id` also `soundfile` | `python3 test_x.py` |
| `test_phonefmt_listing`, `test_run2_mix`, `test_vox_en`, `test_vox_heldout` | the `modal` package (they import `modal_app` or `ml_p3`); no Modal login needed. `test_run2_mix` also `numpy` | `python3 test_x.py` in an environment with `modal` installed |
| `test_r3.py`, `test_r4_losses.py` | `pytest`; `test_r4_losses` also `torch`, `lightning`, `omegaconf`, `numpy<2` (`soundfile` for one test) | `python -m pytest test_r3.py -q`; `python -m pytest test_r4_losses.py -q` |

`librosa` and `safetensors` are not needed by the tests. `librosa` is used by the Modal jobs that cut and score audio, and `safetensors` (with `numpy`) by `swap_weights.py`.

## Credentials and accounts (only for the steps that use them)

| What | Used for |
|---|---|
| Hugging Face token, as the Modal secret `hf` | Downloading gated or rate-limited datasets and the base checkpoint; uploading weights |
| Modal login, and a Modal volume named `parakeet-spoken-form` | All GPU and large-data steps (`modal run` from `pipeline/`) |
| `GEMINI_API_KEY` | `pipeline/ml_llm.py`: LLM-written readings, candidate readings and lexicon drafts for the 16 languages of `ml_multi`. Optional `LLM_LOG` file logs token use |
| Google Cloud login (`gcloud auth print-access-token`, project set in `gcloud config`) | `pipeline/ml_tts_cloud.py` and the other cloud-voice generators (Chirp 3 HD). Billed per character |
| A second-transcript service | The fetcher is not in this repo; the judges only read its output (`ext.jsonl`, see RECIPE section 2) |

You must follow the terms of each provider (Google Cloud Text-to-Speech, the Gemini API, Hugging Face datasets, Modal). No synthetic audio is distributed with this repo.

## Not needed for reading the data tools

`eval/quality/data_gate.py`, `carve_heldout.py`, `score_heldout.py`, `number_outcomes.py` and the judges run on local copies of the volume files with stdlib Python.
