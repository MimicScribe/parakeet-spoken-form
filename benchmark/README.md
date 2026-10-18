# benchmark

A public number benchmark for spoken-form ASR: real held-out speech in 15 languages plus English (Earnings-22), a stitched seam bed, and the baselines. Stdlib Python except the decode step (NeMo on [Modal](https://modal.com)). The only number parser is the public reader in [`eval/quality/`](../eval/quality/), imported unchanged.

| file | role |
|---|---|
| `modal_bench.py` | GPU decodes: `heldout` (one decode per row), `seam` (15 s windows every 10 s, 5 start offsets, plus each sentence alone) |
| `value_accuracy.py` | value-right and form-match on labelled spans; cluster-bootstrap CIs; paired difference vs a base arm |
| `seam_harness.py` | midpoint-of-overlap word merge, sentence assignment, reader-only parse, then `offset_score.py` |
| `oracle_recall.py` | the reader's recall on the spoken-form labels themselves (gate: 95% per language) |
| `invented_words.py` | invented tokens against a public lexicon (FLEURS train text, optional `wordfreq`, number morphemes); no model decode in the lexicon |
| `verbalizer.py` | baseline: stock digits rewritten with the pipeline's canonical grammar readings (`pipeline/ml_cands.py`, `readings.py`) |
| `stitch_bed.py`, `export_labels.py` | build a seam bed from label rows; write the publishable label files |

## Run

```bash
modal run benchmark/modal_bench.py::heldout --arm stock --refs LABELS --out hyps/stock
modal run benchmark/modal_bench.py::heldout --arm ckpt:<decoder_joint.pt on the volume> --refs LABELS --out hyps/B
python3 benchmark/verbalizer.py hyps/stock hyps/stock_verb
python3 benchmark/value_accuracy.py --refs LABELS --arm stock=hyps/stock --arm B=hyps/B --arm stock+verb=hyps/stock_verb --arm oracle
python3 benchmark/oracle_recall.py --refs LABELS --bed bed/manifest.json
modal run benchmark/modal_bench.py::seam --arm stock --manifest bed/manifest.json --bed-dir <bed on the volume> --out dec_stock.json
python3 benchmark/seam_harness.py build --manifest bed/manifest.json --decoded dec_stock.json --arm stock --work seam/
python3 benchmark/seam_harness.py score seam/ stock B
cd benchmark && python3 -m unittest test_benchmark
```

## Metrics

- **Value-right** (headline): every caption number of a span is matched by the hypothesis with the same value, in digits or words (`number_outcomes`, read in context; digit runs and clocks through the `score_values` slot rules). Reported on all spans, on *scorable* spans (the label reading itself scores right), and on *non-canonical* spans (said differently from the grammar's canonical reading: where a verbalizer has to guess).
- **Form-match | right**: of the value-right spans, the share whose label reading appears word for word in the hypothesis; `join` ignores spaces and hyphens (split vs fused compounds).
- **Seam total wrong** (headline of the bed): wrong slot readings summed over 5 offsets, split into seam (right at some offset) and always-wrong; paired file-bootstrap 90% CI and sign test (`offset_score.py`). `scorable/` keeps only slots the scorer reads right on both the spoken label and the written caption.
- CIs resample clusters: parliament day, video, FLEURS sentence, or earnings call (`common.cluster_of`).

## Data and licences

| source | used for | licence |
|---|---|---|
| [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) | held-out rows, bed | CC0 (European Parliament legal notice applies to the raw data) |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) | held-out rows, invented-word lexicon (train split) | CC BY 4.0 |
| [YODAS-Granary](https://huggingface.co/datasets/espnet/yodas-granary) | held-out rows, bed | CC BY 3.0 |
| Common Voice (uk) | held-out rows | CC0; the Mozilla Data Collective terms also forbid re-hosting the dataset |
| [Earnings-22](https://github.com/revdotcom/speech-datasets/tree/main/earnings22) | English numbers (`--e22`) | transcripts CC BY-SA 4.0; the audio has no stated licence and is never redistributed |

Published: span labels and source ids (`export_labels.py`: caption, spoken-form label, spans with kind / reading / canonical reading, crop times) and the bed plan. Not published: any audio. The benchmark has no synthetic speech; TTS audio used elsewhere in this repo is never redistributed. Labels derived from Earnings transcripts carry CC BY-SA 4.0.
