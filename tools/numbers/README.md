# Number data and scoring tools

Tools for building and checking synthetic spoken-number speech data, and for scoring number accuracy. Configuration
comes from the environment or a `.env` (see `.env.example`); nothing here holds keys or machine paths. Every script takes
`--root DIR`, the data directory it reads from and writes to.

| Tool | What it does |
|---|---|
| `generators.py` | Writes briefs from an explicit coverage spec: each number's value, exact spoken reading, training label, kind, repeat structure and position, plus context. `--tag v3 --root DIR` → `briefs/v3.jsonl`, `spec_v3.json`. |
| `sentences.py` | An LLM writes one carrier sentence per brief; any sentence that changes a reading, misplaces a start/end number or touches a converted reading with a stray number word is rejected and re-briefed. `--avoid FILE` bans over-used openings. Needs `GEMINI_API_KEY` (or `--engine haiku` with the `claude` CLI). |
| `neardup.py` | Drops sentences whose number-masked 4-gram Jaccard with any indexed pack (`--index GLOB`) is ≥ 0.6. |
| `census.py` | Counts and value spread per number kind with pass/fail flags: years (2010–2035 share, future share), clock (off-grid minutes, styles), money (every magnitude, cents), decimals, percents, counts, dates, digit strings (repeat-free share, 5-digit share, natural phone grouping, dash-free labels), positions, distinct readings, duplicates, name/opening concentration. Run on briefs (`--briefs`) before any spend, on sentences before synthesis, and with `--kept FILE` after alignment. |
| `fix_labels.py` | Label conventions: digit strings without dashes; word+number codes as words; drops rows whose converted reading is ambiguous. |
| `marker.py` | Repeat marker for training labels: `¡` between identical neighbours inside a number (`5¡5`, `two¡ two`), or counted (`¡` copy 1, `¿` copy 2, `¡¡` copy 3+); `strip_marker`, `analyze` (why a pair was not marked). `python3 -I marker.py` runs its self-tests. |
| `scorer/numscore.py` | Form-blind number scorer: does the hypothesis carry every number value of the reference, in order (digits vs words irrelevant)? `scorer/test_numscore.py` runs the cases. |
| `scorer/paired.py` | Paired arm comparison on per-clip verdicts: fixed/broken counts, exact McNemar p, video-cluster bootstrap CI. |
| `scorer/repeatcount.py` | Exact repeat-count scorer: is every repeated digit or period of the reference still there? Classifies failures (`repeat_drop`, `repeat_add`, `repeat_mixed`, `period_drop`, `period_add`, `other_value`, `slot_dropped`); form-blind (`double two` = `two two` = `22`); locates a slot in the hypothesis by alignment; self-corrections, letter-digit codes (`alnum` mode), times / ranges / powers / lakh-crore (`score_other`); counted markers are stripped before reading and reported apart. `python3 -I scorer/test_repeatcount.py` runs 231 golden pairs, negatives included. |

A pinned number reader and normaliser for scoring (golden pairs, sha256 pins) lives in [`../numread/`](../numread/README.md).

Typical order: `generators.py` → `census.py --briefs` → `sentences.py` → `neardup.py` → `census.py` → synthesize → `census.py --kept`.
