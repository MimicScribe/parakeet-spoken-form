# Data-quality checks for synthetic sentence packs

Gates to run on a new sentence pack before any audio is synthesized or any training run starts. Each prints PASS / FAIL lines with the measured number and can fail the exit code.
Rows are JSONL; the spoken sentence is `said` (falls back to `text` for old rows). Number-masked shingles come from [`../numbers/neardup.py`](../numbers/neardup.py).

| File | What it does |
|---|---|
| `dupcheck.py` | Near-duplicate and leakage check. Max number-masked 4-gram Jaccard of every new row against existing TRAIN rows and held-out EVAL texts (histogram, quantiles, hits >= 0.5 with the matched old text), within-pack sequential similarity, exact duplicate sentences, units folded to one token (`5%` = `five percent`). Leakage of held-out content: digit strings of >= 5 digits (written or spoken digit by digit) and spoken number runs of >= 4 words that occur verbatim in an eval text. `--gate`: exit 1 on any leakage or exact duplicate. |
| `variety.py` | Variety gates. Text: distinct 3-grams of the carriers, sentence-frame patterns per row and top-50 share, opening 3-gram / ending 2-gram concentration, question share, length spread. Config-driven: `caps` (no value of a field above a share), `shares` (min / max share of a value), slot-value uniqueness with closed-space cells that may repeat, topic repetition by `scene_id`, and an audio-plan section (tempo range, gender band, rows per voice, engine / degradation shares). Gates marked `[n]` are relaxed by a sampling-noise margin below 1,000 rows. `--sem` adds the semantic gate. |
| `semantic.py`, `minilm_np.py` | Semantic paraphrase check: all-MiniLM-L6-v2 in pure numpy (Apache-2.0 weights from a local snapshot) on number-masked frames. `index` embeds a text collection (resumable), `check` reports nearest-neighbour cosine vs each index and within the pack (flag >= 0.80, hard >= 0.90), `calibrate` compares paraphrase pairs' cosine with their lexical Jaccard (`example_pairs.jsonl` is a format example). A weak second detector: calibrate on your own pairs. |

For the per-number-type census (value spread, years, clock times, prices, phone numbers, decimals) use [`../numbers/census.py`](../numbers/census.py).

```
python3 -I dupcheck.py --rows new.jsonl --train 'train/*.jsonl' --eval 'eval/*.jsonl' --out dupcheck.txt --gate
python3 -I semantic.py index --texts 'train/*.jsonl' --name train --out cache/
python3 -I semantic.py index --texts 'eval/*.jsonl' --name eval --out cache/
python3 -I semantic.py check --rows new.jsonl --index cache/ --out semantic.json
python3 -I variety.py --rows new.jsonl --sem semantic.json --config gates.json --out variety.txt
sh run_tests.sh
```

Environment (names only): `MINILM_DIR` (snapshot of sentence-transformers/all-MiniLM-L6-v2 with `vocab.txt` and `model.safetensors`), `HF_HOME` (cache searched when `MINILM_DIR` is unset), `VECLIB_MAXIMUM_THREADS` (keep the numpy embedder light).
