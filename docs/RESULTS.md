# Results (2026-10-17)

Benchmarks behind [CONCEPTS.md](CONCEPTS.md). Each table names its bed, its size and its scorer.

**Caveats that apply to everything here:**
- Every number comes from one run per arm. None of the seam beds below is a sealed holdout: both can be scored again and again. The sealed beds that RESTART.md and OBSERVATIONS.md mention are a different thing: the VoxPopuli, Earnings-21 and FLEURS sets decoded on the NeMo side with `eval_multi --sealed` (`ft_eval.SEALED3_SETS`), looked at once for the final candidate. They measure word accuracy of the English and multilingual model, and no table in this file uses them.
- The seam beds run through a production streaming decoder that is not part of this repo. It uses ~15 s overlapping windows, a token-level merge, and a digit verbalizer for digits the model leaves. The scorers are in [`eval/quality/`](../eval/quality).

**Models:**
- **B:** the current published model (see README).
- **R5-b:** the previous spoken-form run on the wave-1 data, a soup at α 0.5 back toward B.

## 1. Seam breaks, synthetic adversarial bank
- **Bed:** 22 languages × 30 number-dense sentences (2 voices / rates), rendered with system TTS voices and joined into continuous files.
- **Runs:** each file is decoded at 5 start offsets (0–7.6 s), so window edges fall at different places in each number.
- **Scoring (strict):** decoded text is parsed into numbers by an external number grammar (it produces `parse_out_*.jsonl`; not in this repo). Slots are matched in order, each parsed number is used once, a phone must be an exact digit run, and clock and price parts must be adjacent. It is still a screen, not a proof.
- **Total wrong (headline):** slot readings that are wrong, summed over offsets.
- **Seam-broken (`offset_score.py`):** a slot that is wrong at one offset and right at another. This needs no reference transcript.
- **Always-wrong:** a slot wrong at every offset.
- **Size:** 1,100 slots, 5,500 readings per arm.

| | B | R5-b |
|---|---|---|
| **total wrong** | **767** | **856** |
| seam-broken | 212 | 251 |
| always-wrong | 555 | 605 |

R5-b is +89 total wrong (90% CI +52 to +128, file bootstrap; 70 slots worse, 27 better; sign test p < 0.001). Window-offset timing (LAG ±0.1 s) changes nothing. Boundary-guarded (942 slots): +69 (CI +34 to +105). A repeat run of B: +0 (CI -2 to +2).

Total wrong by language (11 of the 22 languages are shown here; the totals above cover all 22):

| Lang | B | R5-b |
|---|---|---|
| en | 0 | 0 |
| es | 31 | 39 |
| fr | 14 | 18 |
| de | 1 | 12 |
| it | 15 | 31 |
| nl | 18 | 28 |
| pt | 17 | 29 |
| ro | 71 | 75 |
| ru | 28 | 30 |
| hu | 47 | 42 |
| fi | 87 | 100 |

Only de is significant on its own; the es and fr CIs include 0.

- **By slot (B):** big numbers 34, phone 63, price 33 of the seam-broken readings. 94 of the 130 long-number breaks (big numbers, phone, price) are merge-touched, meaning the two windows read the number differently and the merge spliced them.
- **Known limits:**
  - The TTS voices group phone numbers their own way.
  - The bank has one or two voices per language.

## 2. Seam breaks, real held-out speech

- **Bed:** held-out rows only, stitched by session or video into files of up to 85 s.
- **Size:** 11 languages, 71 files, 75 min, 687 number slots (repeated slot types numbered; phone fragments joined only when contiguous).
- **Runs:** 5 offsets per file.
- **Readings:** 3,435 per arm.
- **Scoring:** strict is the headline; lenient is a sensitivity check.

| Scorer | Total wrong B | Total wrong R5-b | Seam-broken B / R5-b | Always-wrong B / R5-b |
|---|---|---|---|---|
| strict | 466 (13.6%) | 509 (14.8%) | 116 / 144 | 350 / 365 |
| lenient | 402 | 440 | 112 / 135 | 290 / 305 |

- **Direction:** R5-b +43 total wrong (90% CI +9 to +81, sign p = 0.07). Wave 1 alone: +24 (CI -4 to +55, not significant). English 79 to 77. 35 of the +43 sit within 1.5 s of a merge gap.
- **Shares:** 49% of B's seam breaks are merge-touched. Phone numbers (non-English) are the worst class: seam-broken 8% (B) and 14% (R5-b), always-wrong 23% and 16%.
- **Known limits:**
  - Clock minutes are undercounted, because the grammar drops minutes after hour words.
  - Real phone and price slots are thin in several languages.

## 3. Merge-side fixes (negative results)
Both fixes were tested by replaying the recorded window decodes through alternative merge policies. Replays of the production policy reproduce the recorded runs exactly on the synthetic bed.

**The "real" columns are not the section 2 totals.** The merge prototype replayed its own run of the real bed, with slightly different totals (production 410 and 418 here, against strict totals of 466 for B and 509 for R5-b in section 2; the synthetic columns likewise come from the replay's own scoring pass, not the 767 / 856 of section 1). Its replay was also not fully faithful on the real bed: 7 files differed from the recorded run by 0 to 4 tokens. Compare the policies with each other within a column, not with section 2.

**Wrong readings (seam-broken + always-wrong):**

| Policy | Synthetic B | Synthetic R5-b | Real B | Real R5-b |
|---|---|---|---|---|
| production | 789 | 871 | 410 | 418 |
| number-whole (take a number from one window) | 780 | 859 | 401 | 412 |
| edge-trim, 5 frames (prefer the other window near an edge) | 788 | 871 | 408 | 412 |

- **Edge-trim** lost real English words (5–6 words in 7 of 10 real English files).
- **A number-gated live re-decode** of the seam (one extra decode centred on the number) changed the synthetic-bank totals by −2 to +7.
- **Why they fail:** the earlier window clips the number and the next window starts mid-number and regroups it, so no window holds the whole number to pick from.

## 4. Conversion completeness (held-out real speech, R5-b candidates)
Share of numbers written as words / left as digits / mixed inside one number, in %:

| Run | de | es | fr | it | nl | pt |
|---|---|---|---|---|---|---|
| stock | 26 / 70 / 1.7 | 12 / 87 / 0.6 | 26 / 73 / 0.3 | 30 / 70 / 0.3 | 22 / 77 / 0.5 | 3 / 97 / 0 |
| α 0.5 | 48 / 50 / 0.6 | 41 / 59 / 0.3 | 42 / 55 / 2.2 | 44 / 55 / 0.5 | 38 / 57 / 0.5 | 19 / 80 / 0.5 |
| α 1.0 | 71 / 25 / 0.6 | 77 / 22 / 0.3 | 75 / 22 / 1.9 | 70 / 22 / 0.8 | 65 / 25 / 1.9 | 63 / 36 / 0.5 |

- **Mixing inside one number is rare.** The half-conversion is between numbers, so two windows can each pick a different form for the same number.
- **Size:** 210–383 spans per language.
