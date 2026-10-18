# Concepts worth borrowing

The ideas from this project that we think transfer, with the numbers that backed them. Use them to derive your own work; the code is in `pipeline/` and `eval/quality/`, the step-by-step is in [RECIPE.md](RECIPE.md).

## 1. The adversarial measurement loop

Reference transcripts are a weak test for number seams, because most references hold few long numbers and say nothing about where a window edge fell. Instead:

1. Render number-dense synthetic speech, and stitch held-out real speech.
2. Run it through a real streaming decoder: overlapping windows, then the production merge. Do this at 5 shifted window offsets, so each sentence meets 5 different edge positions.
3. Compare the offsets. A value right at one offset and wrong at another is a **seam break**. It needs no reference, because the edge position is the only thing that changed. A value wrong at every offset is a model error, not a seam error. `eval/quality/offset_score.py` computes both.
4. Diagnose the mechanism of each break class (next two sections).
5. Build training data aimed at that mechanism.
6. Gate every candidate on the same test, with a paired bootstrap over sentence slots (90% CI), so a candidate is judged against the baseline on the same items.

We do **not** train on the harness outputs. The harness only measures. Training data comes from real and separately synthesised speech, so the test stays independent of the training set.

Measured breakdown: on the synthetic bank, about 70% of model B's long-number seam breaks were merge-made splices (94 of 130); the rest were edge garbles inside one window. On real held-out speech about 50% of breaks were merge-touched (RESULTS sections 1 and 2).

## 2. Half-conversion causes seam breaks

A model that writes words for some numbers and digits for others, followed by a downstream digit-to-words step, is the worst of both. Two overlapping windows decode the same audio; one writes the number as words, the other as digits, and the later step reads the digits in its own grammar. The two forms differ, the merge has nothing to align on, and the stitch splices them.

- Digits-only baseline: the windows agree on the form; seam breaks were not tabulated for it here.
- English: 0 seam breaks in both B and R5-b on the synthetic bank, so English does not separate them.
- Model B has 209 seam breaks on the 22-language synthetic bank, mostly outside English.
- R5-b, a blended (soup) checkpoint at alpha 0.5, converted 38-48% of numbers on real held-out speech in 6 languages (Portuguese 19%) and left 50-59% as digits (Portuguese 80%). On the 22-language synthetic bank its seam breaks rose from 209 to 256 in a streaming-decoder test. On real speech the difference (92 to 101 lenient, 116 to 130 strict) is within noise. These are three different populations; do not add them up.
- Numbers that mix digits and words inside one number were only 0.3-2.2% at alpha 0.5 (RESULTS section 4), so the split is between numbers, not within one.

Hypothesis (one run per arm, not yet tested at alpha 1.0): the half-conversion is what raised the breaks.

Lesson: pick candidates by **conversion completeness under a damage gate**. Among candidates that stay inside the no-damage limits, take the most converted. Raw accuracy alone will choose a half-converted model.

## 3. Merge picks cannot fix long numbers

The obvious fix is in the merge: choose the better of two window readings. It barely helps. The earlier window clips the number at its edge, the next window starts in the middle of the number and **regroups** the digits ("uno nueve ocho" becomes "un millon novecientos ochenta y"), so no window holds the whole number.

Measured: a number-aware merge pick and an edge-trim policy moved errors about 1-2%, and edge trim lost real English words (5-6 words on 7 of 10 real files). Both were dropped. The merge replay was slightly unfaithful on the real bed (7 of 71 files differed from the recorded run), and the live re-decode was a single synthetic-bank test. A further re-decode variant is being tested.

Our approach for run 2 (data built, training pending): **seam pairs cut inside long numbers**, each labelled with only the continuation that side heard, so the next window continues the reading instead of regrouping.

The **0.9 number-word edge rule** makes those labels honest: a number word cut by an edge is labelled only if at least 90% of it is inside. Without it, in compound-number languages about 75% of in-number cuts labelled partly heard words as whole (es pilot: 1,231 of 1,673 pairs). With it, in-word halves fell from 775 / 2,678 / 694 / 755 to 34 / 10 / 23 / 48 (es / fr / it / pt). These are label-quality counts on the training data, not a model result.

## 4. Labels by agreement

No single source is trusted. A reading becomes a label when independent evidence agrees:

- a CTC forced-alignment fit of each candidate reading against the audio, with a per-language reader;
- a second transcript from a separate recognizer, used to confirm, pick or veto;
- value checks (does the reading parse to the caption's value?);
- partial rows: keep verified pieces around undecided spans rather than dropping the clip;
- judge audits on samples, and on every reject class, before a reject rate is believed.

Where a reading can be built in code from the grammar table, it is (English and the larger European languages, and the phone, date and long-number sets). An LLM-written phone reading was the top reject cause: it dropped, merged or re-grouped digits. Code can be tested for gender, case and cents rules; an LLM cannot be. For the 16 languages without a tested reader, an LLM writes some spoken readings (and proposes candidate readings for real captions), but a reading is used only if it parses back to the written value and leaves the rest of the sentence unchanged; the CTC fit, second transcript and judges then apply as usual.

## 5. Census before training

A run is only as good as its thinnest cell. Before training, count per language x class:

- years by value (each value in 1990-2030 at a minimum count), forward years (2031-2060), numbers 0-100;
- clock times, prices, phones, decimals, millions and billions;
- template and session caps, so one carrier sentence or one video cannot dominate;
- synthetic share caps per class (30%, or 50% for shapes real speech rarely has);
- a cap on the share of rows decided by the second recognizer, per class.

Counts alone are not enough; check the spread of values. Many repeated failures were data that had the right count and the wrong shape (one clean voice for years, one common price, skewed phone lengths). `eval/quality/data_gate.py` is that check. Fix or mask labels rather than filtering data to pass it.

## 6. Form conventions follow the audio

Do not canonicalise how a number is read. People say phone numbers in pairs, in triples or digit by digit; "nineteen ninety" and "one thousand nine hundred ninety"; "thirteen hundred" and "one thousand three hundred". A label that forces one convention teaches the model to ignore what was said, and reintroduces the guessing that spoken form was meant to remove. Take the reading the evidence supports, keep several readings per value in the grammar table, and census the forms rather than normalising them.

## Glossary

- **Wave 1 / run 2:** wave 1 is the first six languages (de, es, fr, it, nl, pt). Run 2 is the data phase for 17 languages plus English, with long numbers across window seams.
- **mix4 / mix4final:** the run-2 training mix built by `build_mix4`; `mix4final` is the candidate build (`--final`) written to its own directory.
- **P3:** the real-speech labelling pipeline (`ml_p3.py`): caption numbers relabelled with the reading heard in the audio.
- **ext:** the second transcript from a separate recognizer, used as independent evidence (`ext.jsonl`).
- **B / R5-b:** B is the published model; R5-b is the previous spoken-form run on the wave-1 data (a soup at alpha 0.5 back toward B).
- **Window:** one overlapping chunk of audio the streaming decoder transcribes; a training window crop mimics that geometry.
- **Seam pair:** two adjacent window crops cut inside a number, each labelled only with what its side heard.
- **seam_long:** seam pairs restricted to cuts inside readings of 3 or more words (`seam_crops --min-words 3`).
- **Anchor:** FLEURS digit rows of the languages a run does not train, listed `--anchor18` times and held at stock behaviour (KL to stock), so those languages keep writing digits.
- **Census:** the per-language, per-class count of spans and value spread in a mix, checked before any run (`data_gate.py`).
