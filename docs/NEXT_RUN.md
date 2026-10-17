# Next training run: proposal (2026-10-14, proposal only)

**Maintainer decision 2026-10-14: the goal is spoken-form numbers in ALL 25 languages, eventually** ("I want everything in spoken form"). Digits from the model become the exception. A downstream application's number reader stays as the backstop for any digits the model still writes.

Builds on [`OBSERVATIONS.md`](../OBSERVATIONS.md) and [`RESTART.md`](../RESTART.md). Init from model B (`w3soup_a0.5@0`). Same method as B: decoder + joint only, KL to stock, `dur_weight 0`, soup + α blend.

## Goal: numbers per language, written the way they were said

Today:

- **English:** the model writes numbers as spoken words.
- **The other 24 languages:** the model writes digits (about 75% of numbers), and a downstream per-language grammar turns them into words afterwards. That works, but three things go wrong:
  - **The reading is guessed.** "100" becomes "einhundert" even when the speaker said "hundert".
  - **Seam mismatches.** Two overlapping windows can write the same number differently (digits in one, words in the other), and the merge has nothing to align on. This is the English problem the project started from.
  - **Wrong values.** Mixed word + digit output ("десет 000") and English number words inside non-English sentences come out of the model, where no downstream step can repair them.

## What the run contains

### 1. Spoken-form numbers for the six largest languages (new)

- **Languages:** German, Spanish, French, Italian, Dutch, Portuguese.
- **Data: already built, never trained** (`ml_p3.py`, on the volume at `/vol/ml/p3/<lang>/rows.jsonl`).
  - Real speech: VoxPopuli TRAIN (CC0) plus the D2 harvest rows.
  - Each number is labelled with the reading heard in the audio. A per-language CTC model picks between the readings a per-language number grammar can produce, so labels always use one grammar's wording.
  - Hours: de 2.38, es 2.36, fr 2.30, it 2.42, nl 2.23, pt 1.26. Decide rate 75–82%, pt 49%. 0 wrong on the gold check.
- **Caveats:**
  - **French:** the gold check decided only 56% of French clips (short readings), below the 70% bar. Train French only if its held-out exactness passes; otherwise keep French on digits.
  - **Portuguese:** only Brazilian readings; about 1.3 h.
- **Loss change:** these rows train spoken form (like English "span" rows). The KL-to-stock anchor still covers their non-number tokens, and all other non-English rows stay anchored to stock (digits).

### 2. The other 18 languages stay on digits

- **No training data that would be safe.** Kokoro TTS covers only en/es/fr/it/pt. Piper voice licences vary. macOS voices are test-only.
- **Two targeted fixes do ride along:**
  - **Greek final ς:** `restore_unk_nonen` puts the Greek rows the filter dropped back in (plain 0.22 h + digit 1.11 h). Target: Greek back to stock.
  - **Mixed word + digit labels:** normalize them in the non-English rows ("десет 000" → all digits).
- **Possible:** non-English real-speech rows with a year followed by a quantity ("In 2019 we had 345 …"). This is where the model writes English number words today (10 of B's 12 cases). Only if FLEURS/VoxPopuli TRAIN has enough of them.

### 3. English fixes

- **Short-phrase drops:** decide data vs loss with the step-0a census first (see OBSERVATIONS 2.1; about 1 h of private-harness runs, no GPU).
  - **If data:** drop or relabel the caption rows that omit backchannels, or make them anchor-only.
  - **If loss:** anchor non-number tokens on span / `anchor_num` rows, and use a softer blank gate on human-verified rows.
  - Either way, add ICSI hand-off crops (`by_turns` only).
- **Keep everything B won:** year minimal pairs ×2, number share 35–40%, stock as KL teacher.

## Evidence for the all-languages goal (OBSERVATIONS 2.10, 2026-10-14)

A synthetic seam test ran model B on the meeting path: 660 number-dense sentences in 21 languages plus English, at 5 window-phase offsets.
- **English:** a number read in a long file never differed from decoding the sentence alone (0 of 150).
- **Non-English:** the readings differed 38% of the time, and 4.1% came out as merge-made wrong values. The cause is digits that each window groups differently.

That test is the per-language "seam disputes" gate below; its B baseline is the table in OBSERVATIONS 2.10.

**Label caution for waves 2–3:** the number grammars write nominative cardinals, while Slavic and Baltic speech uses inflected forms. Labels taken from the grammar would teach the wrong case; the CTC reading check must be allowed to choose inflected readings, or those spans stay undecided.

## Gates (B is the baseline; same evaluation build)

1. **Six spoken-form languages:**
   - a held-out spoken-form set per language: number readings exact ≥ 90%, 0 wrong values;
   - FLEURS WER no worse than B (number-neutral basis, paired CI);
   - seam number disputes in a long-file test of each language: fewer than B.
2. **All 25 languages, FLEURS:**
   - no language measurably worse than B;
   - Greek back to stock or better.
3. **English:**
   - the 44-call number bar no worse;
   - years (blind set + minimal pairs) no worse;
   - Levenshtein content drops and insertions on E21-44 + AMI-16;
   - a private live-path drop check on real meetings: fewer drops than B.
4. **Before measuring:** fix the number scorer's false positives, then re-baseline B.
5. **Last:** live invariants, then one sealed-holdout look.

## Cost and time (estimate)

| Item | Estimate |
|---|---|
| GPU: mix build, one rehearsal, one main run, soups, NeMo evals | ≈ $10–20 |
| Data | ≈ $0: the six languages are already labelled. Greek and the mixed-label cleanup are mix-build steps |
| Maintainer time | one listening round on the six languages' held-out readings, ≈ 30 min |
| Wall clock | about a day: mix + run + NeMo evals ≈ 6 h; private-harness gates overnight |

## Road to all 25 languages

The six-language step reuses the P3 pipeline (`ml_p3.py`): real speech with digit captions, numbers relabelled by a per-language CTC reader choosing among the number grammar's readings. The same pipeline extends to the rest. What each language needs:
- real speech with numbers in its captions, under a training-safe licence;
- a per-language CTC reader;
- a gold check that passes (≥ 70% decided, 0 wrong).

| Wave | Languages | Real-speech source | Status |
|---|---|---|---|
| 1 | de, es, fr, it, nl, pt | VoxPopuli TRAIN (CC0), YODAS-Granary for pt, D2 rows | labelled, untrained (this run) |
| 2 | pl, ro, hu, cs, fi, hr, sk, sl, et, lt | VoxPopuli TRAIN (CC0) has transcribed speech for all ten | P3 scan + label to run; per-language CTC readers to pick and gold-check |
| 3 | bg, da, el, lv, mt, ru, sv, uk | not in VoxPopuli: Common Voice (CC0) and FLEURS TRAIN (CC BY 4.0), likely few numbers per hour | measure number yield first; Greek also needs its ς rows; Maltese may lack a usable CTC reader |

Notes:
- English stays as it is; the label convention for every language is one number grammar's wording, applied the same way to every label.
- The volume of number-bearing speech per language is the main risk for waves 2–3. If a language yields under about 1 h, it waits, or uses licence-checked TTS (Piper voices, checked per voice).
- Each wave gets the same gates as wave 1: readings ≥ 90% exact, 0 wrong values, FLEURS no worse, seam disputes down.

## Open decisions for the maintainer

1. ~~Spoken form in these six languages?~~ Decided: spoken form everywhere, eventually (above).
2. French: include only if it passes its gate, or leave it on digits now?
3. Run the step-0a drop census first (cheap, decides the English fix), or fold both drop fixes in blind?
