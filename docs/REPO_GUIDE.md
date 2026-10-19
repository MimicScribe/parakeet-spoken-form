# Repo guide

A short map of the code by stage. The data recipe is [RECIPE.md](RECIPE.md); the state and run commands are in [../RESTART.md](../RESTART.md); dependencies and credentials are in [REQUIREMENTS.md](REQUIREMENTS.md). Run every `modal run` from `pipeline/`: the Modal images add local files by path relative to it.

## Lineage

Stock Parakeet Ultra (NeMo checkpoint; see the README for the pinned download) -> `v10pt` (TTS plus replay, 1,500 steps) -> `r3_SBd0` (real-speech retrain-3, 4,000 steps) -> soup `A` -> `r3_W3` (1,000 steps, mix v3) -> soup and alpha 0.5 blend -> **model B**. The launch scripts for each step are in [`../runs/`](../runs/). Run 2 (the current data phase) has no trained model yet.

## `pipeline/` by stage

**Orchestration**
- `modal_app.py`: the Modal app. Entry points for data prep, TTS, mixes (`build_mix*`, `retrain3_mix_v*`), training (`run`, `retrain3`), soups (`soup3`), evaluation (`eval_multi`, `probe_heldout`, `probe_seam`) and export (`export_weights`).
- `common.py`, `preflight.py`: shared constants; pre-run checks.
- `../scripts/launch_run.sh`, `run_manifest.py`, `run_watch.py`: the run launcher and its cost guard. The launcher refuses a run name that has no line in the approvals file you pass, refuses an existing `exp/<run>`, launches `retrain3_80_main` (6 h cap per attempt) from a clean `git archive` of one commit and logs the commit and the sha256 of every mounted module; `run_watch.py` stops the app when no new `step<N>.pt.done` appears and keeps the app logs. Header of the script has the usage.

**English data**
- `readings.py`, `gen.py`, `carriers*.txt`: spoken readings of numbers and carrier sentences, expanded into synthetic rows.
- `labeller.py`, `label_gpu.py`, `fetch.py`, `filter3.py`, `recrop.py`, `restore_fillers.py`, `restore_d1p.py`: English real speech: caption numbers to spoken readings, audio fetch, quiet-gap crops, filters, filler repair.
- `s0_scan.py`, `harvest.py`, `en_hits.py`, `en_win.py`, `vox_en.py`: caption scans, non-English replay harvest, rare-shape hunt, English window and seam crops, English parliament clips.
- `data_replay.py`, `data_fleurs.py`, `data_icsi.py`, `data_e22.py`: replay (LibriSpeech-PC, FLEURS, ICSI) and Earnings-22 data.

**Multilingual real speech (labelling, "P3")**
- `ml_cands.py`, `ml_slav*.py`, `ml_uk_words.py`, `ml_ukau.py`, `ml_grammar_table.json`: number spans and every plausible spoken reading per language; the grammar table.
- `ml_decide.py`, `ml_p3_core.py`, `ml_p3.py`, `ml_modal.py`: CTC-based choice among readings, row building, and the Modal jobs (`scan`, `label2`, `windows2`, `seam_crops`).
- `ml_numword*.py`, `ml_variety*.py`, `ml_wordfix.py`, `ml_fleurs_fix.py`, `cv22_uk.py`, `vox_fill.py`, `vox_heldout.py`: number-word captions, look-alike and business-shape sets, word repair, FLEURS fix, Ukrainian Common Voice rows, targeted and held-out parliament pulls.
- `ml_balance*.py`, `ml_multi_scale.py`: class balance and caps; the Polish / Romanian scale-up.

**The 16 languages of `ml_multi`**
- `ml_multi.py`, `ml_multi_cases.py`, `ml_multi_ctc.py`, `ml_multi_tight.py`, `ml_multi_vox.py`, `ml_multi_lexgen.py`, `ml_multi_lex.json`, `ml_multi_months.json`: number readers, case expansion, CTC labelling, row tightening, parliament source, lexicon. Parts of the lexicon and some candidate readings are LLM-drafted and parse-checked (RECIPE section 8).
- `ml_llm.py`: the LLM client used by the files above and by `ml_tts_multi.py`.

**Synthetic speech**
- `ml_tts.py`, `ml_tts_templates.py`, `ml_tts_uk.py`, `ml_tts_ru.py`, `ml_tts_cloud.py`, `ml_tts_fill.py`, `ml_tts_multi.py`: planning (code-built readings for most languages), carrier text, cloud voicing and top-up. No synthetic audio is distributed.
- `grammar_rules.py`, `grammar_scan.py`: deterministic grammar checks for synthetic readings, and the scan that writes the append-only `grammar_reject_ids*.txt` lists the build excludes (RECIPE section 8).
- `phonefmt.py`: phone-number layout in digit-labelled rows.

**Mix, training, export**
- `mix_build.py`, `mix_report.py`: mix builders (`build3`, `build4`) and the coverage report. `census_common.py` holds the census classes, bars and the shared synthetic rule (fill-to-bar); `mix_build.py` holds the held-out voices and the final post-build check (RECIPE sections 9, 11).
- `ft_train.py`, `ft_eval.py`: training (decoder and joint only, KL anchors) and decoding for evaluation.
- `swap_weights.py`: writes the trained decoder and joint into a CoreML bundle (byte-exact patch; `--check` round-trips stock tensors).

**Scoring**
- `score_retrain.py`, `score_mp.py`, `score_gold.py`, `score_side.py`, `e44_bar.py`, `rescore.py`, `beds.py`: checkpoint scoring and selection, minimal pairs, gold clips, the Earnings-21 number bar, evaluation beds.

**Shell helpers:** `run_p3_*.sh` launch the labelling scan and label jobs.

## `eval/quality/`

Census, held-out, judge and seam scorers; see its [README](../eval/quality/README.md).

## Tests

About 40 test files next to the code. Recent ones: `test_fill_review.py` (fill-to-bar, rare-class floor), `test_grammar_fix.py` and `test_grammar_review.py` (grammar rules, reject lists, held-out voices), `test_syn_seam.py` (synthetic seam pairs), `test_ro_gender.py`. Conventions and dependencies are in [REQUIREMENTS.md](REQUIREMENTS.md).

## Not in this repo

- The Modal volume (checkpoints, mixes, beds, caches) and the `hf` secret.
- The second-transcript fetcher and the judge driver; the number-grammar dumper that regenerates `ml_grammar_table.json`; the step that assembles the English parts into `p3_en.jsonl`.
- The streaming-decoder harness (evaluation beds, merge replay) and the builders for the adversarial and fast beds.
- Corpora (ICSI, Earnings-21 audio) and all generated audio.
