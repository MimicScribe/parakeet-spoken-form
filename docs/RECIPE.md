# Recipe: from sources to a gated training run

How a spoken-form training row is made, and what to check before any run. The code is in `pipeline/` (data, labelling, mix, training) and `eval/quality/` (census, held-out, judges, scorers). Ideas behind the choices are in [CONCEPTS.md](CONCEPTS.md).

Goal: every number and symbol is written as the words the speaker said, in all 25 languages, with no lost words, no invented words and no changed values. Long numbers (phones, millions, prices with cents, dates, years) matter most.

## 1. Source real speech (public data only)

- Parliament speech with machine transcripts (Granary VoxPopuli) plus the matching unlabelled audio.
- YouTube speech with machine transcripts (YODAS-Granary), scanned for number-dense clips.
- English: parliament, YouTube top-up, and caption-hit rows (`en_hits.py`: scan captions as text for phones, clock times and rare classes, then transcribe only the hits).
- Gaps filled from Common Voice (Ukrainian).

Select by class quota (millions, billions, dates, clock times), skip every held-out day and video, and skip ids already used.

## 2. Get a second transcript

A second recognizer, number formatting off, gives independent evidence per clip. For some languages it writes words; for others (nl, pl, ro, ru, uk) it writes digits, which is value-only evidence. The transcript is cached per clip and per pass, so nothing is paid for twice.

## 3. Build the grammar table

`pipeline/ml_grammar_table.json` maps `lang|kind|value` to its accepted spoken readings, including gender, case and cents rules. Build it from a dumper, then check that every key the labeller will ask for exists. Labelling refuses to start when keys are missing.

## 4. Label

`ml_p3.py::label2`. For each number span in the caption:

1. List candidate readings (`ml_cands.spans`).
2. Decide (`ml_decide.decide`): CTC forced alignment with a per-language reader scores each reading; the second transcript can confirm, pick, or veto one.
3. Build the row (`ml_p3_core.make_row`). Spans that stay undecided leave a partial row: the verified pieces are kept, the rest is not trained on.

Do not filter data away to hide label problems. Fix or mask the label instead.

## 5. Cut multi-language rows

For the 16 languages with less data, `ml_multi.py` then `ml_multi_tight.py` cut each row to the runs the second transcript confirms. Numbers are never cut.

## 6. Judge

One acoustic plus transcript check per row (`eval/quality/judge_reject.py`, `judge_plro.py`). Each judge writes a per-pass stats file; the mix builder refuses a pass that was never judged. Rejects go into a per-language reject list.

## 7. Repair instead of dropping

`wordfix_ext.py` repairs interior words when two of three sources agree and the CTC margin is large. Negations are repaired the same way. Edge trim drops unheard edge words and cuts the audio at a quiet pause.

## 8. Synthetic speech

Readings are built in code from the grammar table, never by an LLM. Each clip is verified (CTC reader or second transcript), then judged. Generators: `ml_tts_cloud.py` (a cloud TTS voice), `ml_tts_fill.py`, `gen.py`. Use normal speaking rates, never one clean voice for years, and keep synthetic within the caps in step 11.

## 9. Held-out sets

`eval/quality/carve_heldout.py` carves rare-class held-out lists from labelled real rows, whole sessions or videos at a time. Validation-only passes (`val_*`) never enter training.

## 10. Windows and seam pairs

Build window crops and seam pairs (`ml_p3.windows2`, `seam_crops`; English: `en_win.py`). Seam pairs are cut inside long numbers and labelled only with what each side heard, under the 0.9 number-word edge rule.

## 11. Mix

`modal_app.py::build_mix4` calls `mix_build.build4`:

- one row per id, held-out and rejected rows out, 8-gram dedupe against evaluation text;
- second-recognizer share cap per class;
- per-value year cap (3%);
- template cap (3) and session cap (13; 30 for the English YouTube top-up);
- synthetic share caps: 30% per class; 50% for clock, price, phone, date; 50% for any class in en / es / fr;
- 60% of synthetic degraded; window, seam and plain parts added.

## 12. Census, then gate

`python3 eval/quality/data_gate.py <mix> --langs ... [--tier ...]` checks counts and value spread per language and class: years by value (every 1990-2030 value, forward years 2031-2060), 0-100, clock, price, phone, decimals, long classes, plus the caps above. Map each failure to the source that could fix it.

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
