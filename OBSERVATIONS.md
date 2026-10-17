# Observations and ideas for improvement

A running log of what we have measured about the spoken-form Parakeet Ultra fine-tunes, what we think causes it, and what we would try next. Newest first inside each section. Numbers come from the measurement runs named beside them; nothing here is a plan of record until a run is justified (see "When a new run is justified").

Models referred to:
- **stock**: Parakeet Ultra (`FluidInference/parakeet-ultra-coreml@95eaa59a`), the last publicly released model.
- **v10pt**: an earlier fine-tune, pinned for testing, never released.
- **B**: retrain-3 `w3soup_a0.5` (`MimicScribe/parakeet-ultra-spoken-coreml@5edfc817`), the model for the next release. Decoder + joint trained, encoder frozen, KL to stock, L2-SP to the init.

## 0. 2026-10-17: seam breaks, conversion and label agreement

Short version of what the multilingual data phase taught; details and numbers in [docs/CONCEPTS.md](docs/CONCEPTS.md), the step-by-step in [docs/RECIPE.md](docs/RECIPE.md).

- A seam break (value right at one window offset, wrong at another) needs no reference transcript; about 70% of long-number breaks were merge-made splices.
- Half-converted checkpoints (alpha 0.5 soups wrote words for 40-48% of numbers) made seams worse than digits-only; fully converted English had about 0 breaks. Select by conversion completeness under a damage gate.
- Merge-side picks and edge trimming moved long-number errors by 1-2%; the fix is training on seam pairs cut inside numbers, under the 0.9 number-word edge rule.
- Labels need agreement between CTC fit, a second transcript and value checks; synthetic readings come from code, not an LLM.

## 1. What B does well (keep these when training anything new)

- **English numbers as spoken.** 44 Earnings-21 calls (10,071 reference numbers): exact 9,606 vs 9,594 (v10pt), corrupted 176 vs 185, high-severity 20 vs 24, year blends ("twenty thousand nineteen") 0 vs 2.
- **Years in both readings.** "twenty nineteen" and "two thousand nineteen" are written the way the speaker said them; a blind listening set of real earnings-call years agrees.
- **English word accuracy is flat** vs v10pt: Earnings-21 44 calls +0.08 pp (not significant), AMI-16 +0.03 pp; a sealed holdout +0.11 pp raw WER.
- **Fewer invented spans** in a span census (B-only junk 120 vs v10pt-only 489), though the full insertion census is a wash.
- **Non-English is level with stock.** FLEURS, all 25 languages, 60–65 test sentences each (2026-10-14): number-neutral mean WER 10.25% (stock) vs 10.27% (B). No language measurably worse; Maltese better (−1.13 pp, CI [−2.13, −0.20]). Model-level FLEURS over 24 languages (15,808 utterances): 10.78% vs base 10.77%.
- **Non-English digits stay digits** when the model is run raw: of 274 numbers in a synthetic 21-language bank, about 75% exact digits, 18% native words, 5% English words, about 2% wrong value or dropped.

## 2. Known weaknesses of B

### 2.1 Short phrases dropped (the main cost)

- **What:** B drops more short phrases than v10pt: hand-offs, greetings, short overlapped replies. In a blind listening test of the largest content drops, 23 of B's 38 were words really said (v10pt: 5 of 12). The candidate pool was B 219 vs v10pt 101 spans over Earnings-21-44 + AMI-16.
- **Size:** low. Of B's 219 drops only 16 have 3+ content words and 2 have 5+. Of the extra drops, about 50 are pure gaps (voiced audio, nothing written), roughly one per hour of audio, almost all 1–2 words.
- **On public audio,** non-repeat content drops were Earnings-21-44 18 vs 8 and AMI-16 27 vs 12.
- **Where it started:** candidate A (`sbd0soup`, before W3) drops like B (non-repeat content drops A 18 / B 18 vs v10pt 7–8). So the habit arose in the SBd0 stage (retrain-3 data), not in the final W3 step.
- **Not quiet speech:** 89% of B's drops are over 10 dB above the file's noise floor (median +1.8 dB vs the file's typical speech level). They are overlapped or hurried speech, not mumbling.
- **Not a calibration offset.** A blank-bias sweep (subtract a constant from the blank logit at decode) recovers words but invents about as many:

  | Blank penalty | WER change | Deletions | Insertions | Number bar |
  |---|---|---|---|---|
  | 0.5 | +0.05 pp | −0.48 pp | +0.28 pp | slightly worse |
  | 1.0 | +0.34 pp | −0.94 pp | +0.70 pp | worse |
  | 2.0 | +1.35 pp | −1.77 pp | +1.74 pp | high-severity 5 → 16 |

- **Two candidate causes, both untested:**
  - **Data (likelier):** cross-entropy on labels that omit backchannels. The YODAS caption rows (d1p/d10p) only drop a caption when it is missing a word stock hears (`mix_build.py`, mix3b filter), so captions that leave out short replies still train "silence" there. About 35 h of read LibriSpeech speech adds no hand-offs at all.
  - **Loss:** anchoring gaps in `ft_train.py` that let cross-entropy push short or quiet speech to blank:
  - span rows anchor the blank KL only on the prefix before the first number word;
  - `anchor_num` rows that contain any number word get no anchor at all;
  - the blank KL protects only frames where stock's P(non-blank) > 0.9.
- **Cheap test that separates them (step 0a):** a Levenshtein drop census along the checkpoint chain v10pt@1500 → SBd0@500/@2000/@4000 → A → W3@1000 → B. Run a fast 11-call evaluation (~8 min each) in a streaming evaluation harness (not in this repo), with bundles from `export_weights` + `swap_weights.py`. If the drops jump where the caption rows enter, the fix is a DATA change: drop or relabel those rows, or make them anchor-only. Otherwise try the loss changes below.
- **Ideas:**
  1. Anchor the blank KL on every non-number token of span rows (prefix and suffix).
  2. Anchor the non-number tokens of `anchor_num` rows instead of skipping the row.
  3. Lower the teacher gate to P(non-blank) > 0.5 on human-verified anchor rows (ICSI, LibriSpeech, FLEURS).
  4. Add ICSI hand-off crops (`data_icsi.py --overlap`, `by_turns` mode, overlap 2–20%). Never `dominant` mode: it labels the second speaker's backchannels as silence, which teaches exactly this drop habit.
  5. Lower L2-SP to the init (≈ 0.3) so it doesn't lock B's blank offset in.

### 2.2 Greek

- **What:** FLEURS Greek is about +0.8 pp vs stock (CI includes 0) and +1.7 pp vs v10pt; number-free sentences show it too. B often drops the final ς marker ("της" → "τη").
- **Cause (found):** the vocabulary has no ς piece, so the decoder emits `<unk>` for it. `mix_build.py` drops every row whose label contains ⁇ (the unknown marker), which removes nearly every Greek sentence; `restore_unk_nonen` refilled only digit rows, never plain rows.
- **Fix (committed, not yet trained):** `restore_unk_nonen` now restores plain rows too, and any language that lost more than 20% of its rows to the ⁇ filter gets all its audited rows back (only Greek qualifies: plain 165/272 rows, 0.22 h; digit 337/373, 1.11 h).
- **Before using Greek FLEURS train rows:** check train vs test prompt overlap (normalized-text dedupe).

### 2.3 Acronyms that look like words

- "GAAP" → "gap". Accepted as is: rare, and the meaning is usually clear from context.

### 2.4 Mixed word + digit output

- The model sometimes writes half a number in words and half in digits:
  - Bulgarian "10 000" → "десет 000" (FLEURS bg_bg_001701_2, 1 of 1,512 sentences);
  - "dvanásť,50", "sedem,5", "seitsemän, 5".
- A downstream digit-to-words step then reads the digit part on its own ("десет нула нула нула", "ten zero zero zero"), which is a wrong value.
- **Ideas:**
  1. Check the training labels for mixed forms and normalize them to all digits or all words.
  2. Add minimal pairs where a number word is followed by a digit group.

### 2.5 Decimals and currency in smaller languages

- Seen in every model, not only B, so these are stock priors rather than fine-tune damage:
  - bg "7,5" → "7 за 5,5";
  - lt "2,4" → "dviejų ir 40";
  - hu "7,5" → "hét tized";
  - "12,50 евро/ευρώ" written as cents (ru, el);
  - fi "euroa" → "senttiä", hu "euró" → "cent".
- One B-specific regression: on 3 currency clips B writes ru/el "12,50 евро/ευρώ" as cents where v10pt was right. Small, but it is B's own.
- Training on these (TTS currency/decimal minimal pairs) was cut from the retrain-4 plan: stock already matches B on these cases, and the TTS readings would teach a TTS prior.
- Revisit only with real-speech data.
- **Licence constraint on non-English synthetic data:** Kokoro covers en/es/fr/it/pt only. ru/el/fi/hu/bg/lt/sk would need Piper (licences vary per voice) or real FLEURS/Common Voice TRAIN text. macOS `say` voices are test-only (licence unclear).
- **Coverage caveat:** all non-English number evidence is TTS + FLEURS read speech. There are no real non-English meetings and no non-native English speakers in any eval.

### 2.6 Rare glued blends

- "twentyhousand nineteen", "twive": 3 in 10,071 Earnings-21 numbers.
- Fast-speech year pairs (rate 1.2–1.4×) in the minimal-pair generator would target them, if a run happens anyway.

### 2.7 Number habits seen through a digit-to-words reader (2026-10-14 review)

From an external adversarial review of a downstream digit-to-words reader. Every claim was checked against what the model actually writes (raw output, app number rewrites off). Measured on the 148-clip, 21-language synthetic bank and FLEURS (11 languages × 60).

- **English number words inside non-English sentences.** Raw output; counts are clips with English number words the reference doesn't have:

  | Model | Clips of 148 |
  |---|---|
  | stock | 15 |
  | v10pt | 14 |
  | B | 12 |

  - 10 of B's 12 are one sentence shape: a 3-digit count right after a year ("In 2019 we had 345 employees" in de/it/pt/sv/sk/ro/lt/nl/pl/hr).
  - Sometimes it is also read digit by digit: pl "mieliśmy three four five pracowników", nl "three four five", hr "three forty five".
  - **Stock prior, not fine-tune damage** (B is slightly better). No downstream digit reader can repair it: no digits are left to read.
  - **Training idea:** non-English rows where a year is followed by a quantity. Prefer real speech; TTS readings teach a TTS prior (see 2.5). Gate on this bank's English-word count per language.
- **Non-English figures come out unspaced, and that is good.** The model writes "1200 osób" / "1200 personnes".
  - Space-grouped figures ("1 200") appeared 0 times in 174 real FLEURS digit groups and once in the synthetic bank (fi).
  - "250 500" is ambiguous: a figure, or a code said in two groups.
  - **Keep labels unspaced;** never introduce a thousands-separator style the model doesn't already use.
- **Small numbers as native words.** Seen in raw output: fr "50 %" → "cinquante pour cent", pl "o 14" → "o czternaście", "3 magazyny" → "trzy magazyny" (≈18% of non-English numbers, §1).
  - Value-correct.
  - The non-English label convention is still undecided. If a run labels non-English numbers, pick one convention and audit labels against it before training.
- **Vocabulary gaps.** No training fix exists without a tokenizer change:
  - no `ς` (Greek, 2.2);
  - no `ё`: Russian "объём" decodes as "объем".
  - no `▁ô` word-start piece: Slovak "ôsmy" (8th) can't be spelled, so ordinals ending in 8 fall back to digit-by-digit;
  - no `+` piece at all: a spoken "+49" can only come out as words.

  A future vocabulary extension should add these.

### 2.8 Year reading not stable across a window edge

- In streaming decode, two overlapping windows sometimes choose different readings of the same year ("two thousand X" vs "twenty X"). The merge of the two window outputs then has to pick one, or splices both.
- Seam disputes over different values went 53 → 67 with B; 10 of the extra 13 are seams cutting a "two thousand X" / "twenty twenty X" year. The retrain-3 screen failed for the same reason (cross-window year-reading flips spliced at seams).
- Fast speech drops the "twenty" tail of "two thousand twenty X" (stock never does).
- The " two" / " twe" near-tie at the first year piece is a greedy SEARCH error: forced alignment prefers the full year by 10–15 nats.
- **Training note:** the reading must not depend on where a window starts. Crops that start mid-year or just before one may help.
- **Downstream:** repair of such disputes is handled outside the model. Seams also turn "seventy five" into "seven five": B's long-file value misses were 7 vs 4 of ~216.
- Tried and not shipped:
  - v12 fast-year TTS fixed site clips 6 → 18 of 20, but broke Earnings-22 "twenty twenty two" → "two thousand two" on all 4 models.
  - A decode-time year fork (outside this repo) isn't shippable as is (a frame-base bug, ~5 s per eval).

### 2.10 Non-English numbers break at window seams; English does not (2026-10-14 synthetic seam test)

- **Test:** 660 number-dense sentences (21 languages + English; phone, price, percent, large figure, year, count, clock, range), spoken without pauses by macOS `say` (2 voices or rates per language). Joined into 44 continuous files, each run at 5 start offsets so window edges cut every number: 3,300 sentence readings through a streaming decoder with overlapping windows (14.96 s windows, 9.44 s stride), model B, language pinned.
- **Scoring:** each sentence's long-file reading vs the same sentence decoded alone. Values are parsed back from the words with per-language number grammars and checked against the written digits.

| | English | Other 21 languages |
|---|---|---|
| Reads differently than when decoded alone | 0 / 150 | 1,124 / 2,940 (38%) |
| Wrong value only in the long file, at a merge decision | 0 | 120 (4.1%) |
| Wrong value only in the long file, no merge decision | 0 | 77 (2.6%) |

- **Why:** B writes English numbers as words, so two overlapping windows agree. In the other languages it writes digits, and each window groups them differently ("29,40 €" vs "29 eur 40", phone numbers split differently). A downstream digit-to-words reader then produces two different word sequences, and the window merge stitches pieces of both.
  - About half of the merge cases are mid-word splices: the windows align on a shared piece inside different number words (bg "четиресет" vs "четири нула" → "четири нула хиляди").
  - The rest are whole-word picks of the wrong reading.
  - Classes: phone numbers 46, prices 22, large figures 21, clock times 18.
- **Training implication:** this is direct evidence for spoken-form output in every language. The model writing the words itself removed the seam problem entirely in English on the same test.
- **Gate for any run:** this test is the per-language seam gate NEXT_RUN.md asks for. Per language: share of sentences that read differently from decoded-alone, and merge-made wrong values, both no worse than B.
- **Seam-gate baseline for B: 92 merge-made wrong values** (after downstream merge changes, English unchanged). Downstream changes cannot remove the rest: the model must write the same spoken form for a figure wherever the window edge falls.
- **Label caution for non-English spoken-form data:** number grammars can turn digit transcripts into word labels, but they write nominative cardinals. Slavic/Baltic speakers say inflected forms (years as inflected ordinals, case after prepositions). Labels must match what was said, e.g. choose among grammar variants by forced-alignment score, or the model learns a reading nobody uses.

### 2.9 To check (not yet a finding)

- **Dictation sometimes loses the final punctuation.** Seen on stock: the raw ASR already lacks it on 6 of 36 utterances over 40 words, vs 0 of 34 under 40 words. Each had a pause or "um" before the last words. Not yet measured on B; no audio captured.

## 3. Lessons (do not repeat)

- **Duration KL destroys year values.** Keep `dur_weight 0`.
- **Data before recipe.** Measure where the failure is (model, downstream language choice, or the number grammar) before changing training.
- **Fair baseline.** Compare against the right model on the same evaluation build. For public numbers, that is the last released model, not an intermediate fine-tune.
- **Test through the real caller.** NeMo greedy decoding is not the streaming product path; windows, seams and language detection change what users see.
- **Gate every language.** A change aimed at English moved Greek through a data filter nobody looked at.
- **The KL teacher is stock** in every retrain-3 run: `add_r3_losses` copies the model before `init_from` loads. A plan that assumed otherwise was wrong twice.
- **Don't downsample a mix to save compute.** Steps stream batches, so downsampling saves nothing and breaks the 35–40% English number-row share.
- **Raw-model measurements** must disable every downstream number rewrite, not just the main one. One earlier arm left a secondary English digit rewrite on and miscounted.
- **Number-fidelity scorers have false positives** (adjacent spoken numbers summed, "$0.39" vs "thirty nine cents", entity-code stripping); about 45% of "corrupted" Earnings-21 items were scorer artefacts. Fix the scorer before trusting small deltas.
- **No AMI in any training mix:** its meeting series overlap the AMI evaluation sets. The leak check works by AMI SERIES, not by file.
- **When scaling a mix, recount the English number-row share.** One plan scaled 16 h to 10 h and the share fell from 39% to 24%.
- **Maintainer listening is ground truth.** An external review's listening statistics didn't match the committed answer files; cite the committed files.
- **Check decoder exposure before acting on an adversarial number report.** A text-harness report on a downstream number reader (2026-10-14) listed about ten value-change classes; the model wrote almost none of those shapes in real speech (space groups 0/174, spaced % 1 synthetic clip). Count how often the model emits a shape before investing training or downstream work on it.

## 4. When a new run is justified

A measured, model-side problem (English or multi-language) that a few-hundred-step corrective run can plausibly move. Today none qualifies on its own:
- the short-phrase drops are minor and not a calibration offset;
- Greek alone does not justify a run.

Everything in section 2 rides along if a run happens for another reason.

Before re-measuring B as the baseline, land the number-scorer fixes (adjacent numbers summed, "$0.39" vs "thirty nine cents", entity-code strip).

**Gates for any new arm** (B as baseline, same evaluation build):
- the 44-call Earnings-21 number bar;
- years (blind listening set + held-out minimal pairs);
- reference-aligned content drops AND insertions on Earnings-21-44 + AMI-16, with Levenshtein alignment (not SequenceMatcher);
- a private live-path drop check on real meetings: new ≤ B;
- FLEURS for all 25 languages, scored in a streaming evaluation harness that is not part of this repo;
- synthetic number banks;
- live-path invariants;
- one look at a sealed holdout.
