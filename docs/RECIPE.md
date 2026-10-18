# Recipe: from sources to a gated training run

How a spoken-form training row is made, and what to check before any run. The code is in `pipeline/` (data, labelling, mix, training) and `eval/quality/` (census, held-out, judges, scorers). Ideas behind the choices are in [CONCEPTS.md](CONCEPTS.md).

Goal: every number and symbol is written as the words the speaker said, in all 25 languages, with no lost words, no invented words and no changed values. Long numbers (phones, millions, prices with cents, dates, years) matter most.

## Languages and tiers

Run 2 trains 17 non-English languages plus English; the other languages are anchors that stay at stock behaviour (digits). Tiers set the census bar (`data_gate.py` `TIER_LANGS`; `--tier` overrides):

| Tier | Languages | Bar |
|---|---|---|
| full | en de es fr it nl pt | the first, strictest bar |
| next | ru pl ro | same checks, about 55% of the counts |
| moderate | uk sv cs hu fi sk bg hr | lower counts; bg is held to no regression against stock behaviour only |
| light / anchors | da et el lv lt mt sl | not trained; FLEURS digit rows keep them at stock (`--anchor18`) |

The 17 codes in `build_mix4 --langs`: de es fr it nl pt ru uk pl ro sv cs hu fi sk bg hr (`RUN2_LANGS` in `modal_app.py`, which lists uk with ru, pl, ro; the census tier for uk is moderate). English is built by the English parts.

## 1. Source real speech (public data only)

- Parliament speech with machine transcripts (Granary VoxPopuli) plus the matching unlabelled audio.
- YouTube speech with machine transcripts (YODAS-Granary), scanned for number-dense clips.
- English: parliament, YouTube top-up, and caption-hit rows (`en_hits.py`: scan captions as text for phones, clock times and rare classes, then transcribe only the hits).
- Gaps filled from Common Voice (Ukrainian).

Select by class quota (millions, billions, dates, clock times), skip every held-out day and video, and skip ids already used.

## 2. Get a second transcript

A second recognizer, number formatting off, gives independent evidence per clip. For some languages it writes words; for others (nl, pl, ro, ru, uk) it writes digits, which is value-only evidence. The transcript is cached per clip and per pass, so nothing is paid for twice.

The fetcher for this transcript is **not included** in this repo. Any independent recognizer with number formatting off will do. The judges read its output from `ext.jsonl`, one JSON object per clip, and use only: `id` (the row id), `status` (`"completed"`; `judge_reject.py` treats anything else as no judge), `text` (the transcript), and optionally `words` (`[{"w": ...}, ...]`, preferred over `text` when present; only the `w` field is read). Timestamps are not needed (`judge_reject.py` `judge_tokens` and `judge_row_nl`, `wordfix_ext.py`). `judge_plro.py` reads only `id` and a non-empty `text`.

## 3. Build the grammar table

`pipeline/ml_grammar_table.json` is checked in and usable as is. It maps a key to the spoken words of one reading (a list of words, or `["null"]` for "nothing is said here"). A key has seven `|`-separated fields, `lang|kind|n|f|s|h|m`: language, kind (`card`, `dec`, `ord`, `year`, `cur`, `pct`, `clock`), the number, a form flag (for example `01` for cents), a currency symbol, and hour / minute parts for clock times (the request fields in `ml_cands.build_table`). Example: `es|year|1998|||0|0` -> `mil novecientos noventa y ocho`. It has about 45,900 keys for de, es, fr, it, nl, pt, ru and uk. English and the other 16 languages use their own readers (`readings.py`, `ml_multi.py`), not this table.

To extend it, `ml_cands.build_table` asks a grammar dumper for every key the texts need and merges the answers. The dumper is not published, so supply your own that reads `requests.json` and writes `answers.json` in the same order (see `build_table`). Check that every key the labeller will ask for exists; labelling refuses to start when keys are missing.

## 4. Label

`ml_p3.py::label2`. For each number span in the caption:

1. List candidate readings (`ml_cands.spans`).
2. Decide (`ml_decide.decide`): CTC forced alignment with a per-language reader scores each reading; the second transcript can confirm, pick, or veto one.
3. Build the row (`ml_p3_core.make_row`). Spans that stay undecided leave a partial row: the verified pieces are kept, the rest is not trained on.

Do not filter data away to hide label problems. Fix or mask the label instead.

## 5. Cut multi-language rows

For the 16 languages with less data, `ml_multi.py` then `ml_multi_tight.py` cut each row to the runs the second transcript confirms. Numbers are never cut.

## 6. Judge

One transcript check per row against the second transcript (`eval/quality/judge_reject.py` for de/es/fr/it/nl/pt, `judge_plro.py` for pl/ro and the next-tier languages). The driver that fetches `ext.jsonl` and runs the judges over all passes is not included.

The mix builder (`mix_build.build4`) refuses a pass without `ml/p3v2/<lang>/judged_<pass>.json` and a language without `ml/p3v2/<lang>/reject_ids.txt`. Only `judge_plro.py --upload` writes both; for `judge_reject.py` see the manual step under Troubleshooting.

## 7. Repair instead of dropping

`wordfix_ext.py` repairs interior words when two of three sources agree and the CTC margin is large. Negations are repaired the same way. Edge trim drops unheard edge words and cuts the audio at a quiet pause.

## 8. Synthetic speech

Two routes, by language:

- **Code-built readings** (en, de, es, fr, it, nl, pt, ru, uk): the spoken form comes from the grammar table and the language's reading rules, which can be unit-tested for gender, case and cents. The phone, date and long-number sets use these. Generators: `ml_tts.py`, `ml_tts_cloud.py`, `ml_tts_fill.py`, `gen.py`.
- **LLM-written readings** (the 16 languages of `ml_multi`): `ml_tts_multi.py` has an LLM (Gemini Flash-Lite, public text only) adapt a real caption sentence to carry a number fragment and write its spoken form with the right case and gender. The same models propose candidate readings when labelling real speech (`ml_multi_ctc.py`) and draft the numeral lexicon (`ml_multi_lexgen.py`). An LLM reading is only used if it parses back to the written value and the rest of the sentence is unchanged; it is then voiced, transcribed and judged like any other row.

Long numbers have their own sets (big numbers, prices with cents, phones, placed mid or late in the sentence), plus year sets for the next and moderate tiers (several years per sentence), small-count sets for pl / ro, and top-off sets for the thinnest long-number cells. A held-out-voice set per language (step 9) is generated the same way. Planned sets target the census gaps by value, not a flat quota; `mix_build.TTS_SETS` lists the tags.

**Grammar checks.** `pipeline/grammar_rules.py` holds deterministic checks for known generator slips in synthetic readings (feminine clock hours and ordinal dates, the "et" in French clock times, hour nouns in the wrong number, a cents word that belongs to another currency, unit-less German / Dutch prices, Polish genitive and clock-minute forms, Romanian linker and gender, and others). The generators call them and reject what they find. `pipeline/grammar_scan.py` runs the same checks over existing synthetic rows and writes the ids of bad rows to `ml/p3v2/<lang>/grammar_reject_ids.txt`; a later scan adds a new `grammar_reject_ids_<n>.txt` (a list is never rewritten) and `build4` reads every list. The audio speaks the wrong form, so such rows are excluded, never relabelled, and replaced by regenerated sets. The Romanian gender repair applies to synthetic readings only; labels of real speech stay as spoken ("doi mii").

Each clip is verified (CTC reader or second transcript), then judged. The clips are voiced with Kokoro, Piper or Google Cloud TTS (Chirp 3 HD); see the README credits. Use normal speaking rates, never one clean voice for years, and keep synthetic within the rule in step 11. No synthetic audio is distributed with this repo.

## 9. Held-out sets

`eval/quality/carve_heldout.py` carves rare-class held-out lists from labelled real rows, whole sessions or videos at a time. Validation-only passes (`val_*`) never enter training.

**Held-out voice (2026-10-17).** Each trained language has one cloud TTS persona that never trains (`HELDOUT_VOICES`, `is_heldout_voice`, `check_heldout_voices` in `mix_build.py`). The persona is matched across the language's locales (es-ES and es-US, fr-FR and fr-CA, nl-NL and nl-BE). Its rows form the language's synthetic held-out set, written by the build as `tts_dev_<lang>.jsonl`; other voices' dev rows go to `tts_dev_other_<lang>.jsonl`. The build refuses a language that has neither a persona nor an exemption (en has its own held-out route; bg stays at stock).

## 10. Windows and seam pairs

Build window crops and seam pairs (`ml_p3.windows2`, `seam_crops`; English: `en_win.py`). Seam pairs are cut inside long numbers and labelled only with what each side heard, under the 0.9 number-word edge rule.

The synthetic long-number and year sets also yield seam pairs (and optionally windows) from their train rows only (`ml_p3.py::syn_seam_crops`; a set must carry its judge marker, and the held-out voice and dev rows never enter). The build keeps a synthetic pair only when all its parent rows were chosen under the synthetic rule in step 11. Top-up seam files (`seam_long_*`) add more long-number pairs of real speech.

## 11. Mix

`modal_app.py::build_mix4` calls `mix_build.build4`. Always on: one row per id, held-out and rejected rows out, the second-recognizer share cap per class, the per-value year cap (3%), the synthetic rule below, one voice per language kept out of training (`HELDOUT_VOICES`), and 60% of synthetic degraded (75% for the long-number sets, `TTS_AUG_FRAC_LONG`).

**Synthetic rule (fill-to-bar, 2026-10-17; `census_common.syn_allowed`).** Synthetic share caps are 30% per class, 50% for clock, price, phone, date, and 50% for every class in en / es / fr. A class short of its census bar may be filled with synthetic spans up to the bar (allowance = the larger of the share cap and bar minus real spans). Thin values (each year, each 0-100 count, the hundreds, distinct big numbers) may be filled up to their per-value floors on top of that; where real speech already meets the class bar, the class's synthetic share stays at or under 50% (`VALUE_FILL_MAX_SHARE`) or its share cap, whichever is larger. Real rows always count first and are never displaced: when synthetic is over the rule, synthetic rows are trimmed. bg keeps the plain caps. A class with no bar and almost no real speech (19 real spans or fewer) may hold up to 0.15 h of synthetic rows (`rare_floor_ok`). The build (`syn_room`) and `eval/quality/data_gate.py` call the same functions, so a build that passes its own check passes the gate.

The final build runs `post_build_check` (the gate's cap checks, recomputed on the built language) and fails on any failure except two documented exceptions: rows over the template cap and videos over the group cap (rare-class and thin-value rows that `balance_caps` keeps).

Off unless asked for:

- `--dedupe`: 8-gram dedupe against evaluation text (the older `build()` path);
- `--balance`: template cap (3) and session cap (13; 30 for the English YouTube top-up);
- `--en-bal`: the same caps and a trim of the English top-up into the 35-40% number-share band;
- `--seam-h`, `--win-h`, `--plain-share`, `--seam-long-h`: hours of seam pairs, windows, number-free rows and long-number seam pairs.

The run-2 build (run from `pipeline/`, 17 languages):

```bash
modal run modal_app.py::build_mix4 --langs <17 comma-separated codes> --dedupe --seam-h 5 --win-h 2.9 \
  --plain-share 0.6 --balance --en-bal --win-repeat 2 --anchor18 2 --en-plain-h 91 --en-free-repeat 2 \
  --mix-dir <dir> --final --seam-long-h 6
```

Run-2 order: `ml_p3.py::label2` -> judge (step 6) -> `ml_p3.py::windows2` and `seam_crops` (`--min-words 3` for the long-number set) -> `build_mix4` -> `eval/quality/data_gate.py` -> `retrain3 --mix v4 --mix-dir <dir>` -> `probe_heldout` -> `eval/quality/score_heldout.py`.

## 12. Census, then gate

`python3 eval/quality/data_gate.py <mix dir> --langs ... [--tier ...]` (reads `p3_<lang>.jsonl`, `tts_<lang>`, `win_<lang>`, `heldout_<lang>` from that directory) checks counts and value spread per language and class: years by value (every 1990-2030 value, forward years 2031-2060), 0-100, clock, price, phone, decimals, long classes, plus the caps above. In de and nl a sentence-final cardinal that looks like an ordinal ("Punkt 10." read "zehn") is counted by its value, not as an ordinal; counts only, the label is untouched. Map each failure to the source that could fix it.

English is not censused by this repo: `data_gate.py` needs a `p3_en.jsonl`, and the English parts live under `d1/en4_ok/` on the volume (listed in `mix_build.EN_REAL_PARTS`), assembled by a step that is not included. Census English with your own assembly of those parts, or treat it as not covered.

No training run starts until the census passes its bar. Thin or skewed data has been the repeated failure.

## 13. Run, then gate the candidate

Train on the mix, soup or pick checkpoints, then gate each candidate:

- `score_heldout.py`: four outcomes per number (words right, digits right, wrong value, dropped);
- no-damage rule per language against the baseline: value-right within 2 pp, wrong plus dropped within +2 pp, WER non-inferior;
- the seam test (CONCEPTS.md, first section), paired bootstrap, 90% CI;
- conversion completeness (share of numbers written as words) among candidates that pass the damage gate.

A language that fails its gate stays on the baseline.

## Pitfalls

1. **Shared audio paths.** Passes that write `wav/<id>.wav` overwrite each other and labels land on the wrong audio. Scope audio by pass (`wav/<pass>/`) and audit audio length against row duration.
2. **Duplicate ids across passes.** Recovery passes re-cut the same clips. Keep one row per id.
3. **Never restart a finished labelling pass.** It reopens its rows file for writing and truncates it. If a stats file exists, the pass is done.
4. **A sleeping laptop kills remote clients.** Keep the machine awake, run one detached job per language, and read completion from the volume.
5. **Check running processes after changing a constant.** Jobs started with the old value keep it.
6. **Audit judges before trusting a reject rate.** Early rejects came from letter names, synthetic phrasing habits, digit-only transcripts, missing month names, trailing punctuation and cents gaps.
7. **Judges can also be too loose.** Accepting a code as a substring of all digits, or a month from any number, passes coincidences. Require the value parts to be adjacent.
8. **Negations.** A word-error guard that ignores short words lets lost "no / não / niet" through. Give negations their own reject reason and repair path.
9. **Caps need every tag.** A rule tagged differently escaped the second-recognizer cap. Count every tag that came from the second transcript.
10. **Census gaps look like data gaps.** Scale words, spoken dates, clock times, prices and suffix languages were mis-measured, not missing. Fix the counter before sourcing more data.
11. **Trailing punctuation in span text** breaks character counts. Strip it.
12. **Generator bugs.** Gender ("duas mil", "cincuenta y una"), case after prepositions and accusative scale words. Add grammar tests and reject bad rows.
13. **Dev split by row leaks templates.** Split synthetic dev by carrier template; hold out voices too.
14. **Held-out by whole session or video**, never by row. Without speaker ids, a span-level CTC check still masks unheard numbers.
15. **Pilot every paid step.** Stop if yield is under 30% or a spot check is under 95%.
16. **Disk floors and rate limits.** Keep free disk, and use few readers with backoff against hub 429s.
17. **Quote heredoc delimiters** in shell scripts.

## Troubleshooting

- **Run `modal run` from `pipeline/`.** The Modal images add local files by relative path. Booleans are `--flag` / `--no-flag`; underscores in parameter names become dashes.
- **`never judged` or `reject_ids.txt missing` from `build4`.** After `judge_reject.py`, copy its `judge_stats.json` to `ml/p3v2/<lang>/judged_<pass>.json`, append its `reject_ids.txt` (one id per line) to `ml/p3v2/<lang>/reject_ids.txt`, and put both on the volume. Synthetic sets need their own `judged_<set>.json`.
- **Scoring with `score_retrain.py`.** See RESTART.md section 6.A: `--base` is a flag, not a value, unless `--select` is used.
