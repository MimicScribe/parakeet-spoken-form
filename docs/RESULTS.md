# Results (2026-10-17)

Benchmarks behind [CONCEPTS.md](CONCEPTS.md). Each table names its bed, its size and its scorer.

**Caveats that apply to everything here:**
- Every number comes from one run per arm. There is no sealed holdout.
- The seam beds run through a production streaming decoder that is not part of this repo. It uses ~15 s overlapping windows, a token-level merge, and a digit verbalizer for digits the model leaves. The scorers are in [`eval/quality/`](../eval/quality).

**Models:**
- **B:** the current published model (see README).
- **R5-b:** the previous spoken-form run on the wave-1 data, a soup at α 0.5 back toward B.

## 1. Seam breaks, synthetic adversarial bank
- **Bed:** 22 languages × 30 number-dense sentences (2 voices / rates), rendered with system TTS voices and joined into continuous files.
- **Runs:** each file is decoded at 5 start offsets (0–7.6 s), so window edges fall at different places in each number.
- **Scoring:** values are parsed back with a number grammar (`score_values.py`).
- **Seam-broken (`offset_score.py`):** a slot that is wrong at one offset and right at another. This needs no reference transcript.
- **Always-wrong:** a slot wrong at every offset.
- **Size:** 5,500 slot readings per arm.

| | B | R5-b |
|---|---|---|
| seam-broken | 209 | 256 |
| always-wrong | 580 | 615 |

Seam-broken by language:

| Lang | B | R5-b |
|---|---|---|
| en | 0 | 0 |
| es | 1 | 14 |
| fr | 4 | 14 |
| de | 1 | 12 |
| it | 9 | 13 |
| nl | 8 | 13 |
| pt | 17 | 18 |
| ro | 16 | 25 |
| ru | 13 | 15 |
| hu | 17 | 17 |
| fi | 32 | 25 |

- **By slot (B):** big numbers 34, phone 64, price 32 of the 209. About 70% of the long-number breaks are merge-touched, meaning the two windows read the number differently and the merge spliced them.
- **Known limits:**
  - The TTS voices group phone numbers their own way.
  - The bank has one or two voices per language.

## 2. Seam breaks, real held-out speech
- **Bed:** held-out rows only, stitched by session or video into files of up to 85 s.
- **Size:** 11 languages, 71 files, 75 min, 670 number slots.
- **Runs:** 5 offsets per file.
- **Readings:** 3,350 per arm.

| Scorer | Seam-broken B | Seam-broken R5-b | Always-wrong B | Always-wrong R5-b |
|---|---|---|---|---|
| lenient | 92 (2.7%) | 101 (3.0%) | 240 (7.2%) | 260 (7.8%) |
| strict | 116 (3.5%) | 130 (3.9%) | 340 (10.1%) | 365 (10.9%) |

- **Shares:** about 50% of the breaks are merge-touched. Phone numbers are the worst class: 9–11% seam-broken and about 20% always-wrong.
- **Direction:** the same as the synthetic bank, but the R5-b gap is noise-level here.
- **Known limits:**
  - Clock minutes are undercounted, because the grammar drops minutes after hour words.
  - Real phone and price slots are thin in several languages.

## 3. Merge-side fixes (negative results)
Both fixes were tested by replaying the recorded window decodes through alternative merge policies. Replays of the production policy reproduce the recorded runs exactly on the synthetic bed.

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
