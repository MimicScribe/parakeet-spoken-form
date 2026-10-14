# Launch scripts as run

The shell scripts that launched the real runs, copied from session scratch on 2026-10-14 so the exact flags survive. Paths are rewritten: `$LOGS` (default `./logs`) replaces the old scratch folder, `$HOME` the user folder. Each script runs Modal commands from `pipeline/`. App-side evaluation steps are removed; that harness is private.

## `model_b/`: the lineage of model B (2026-10-13), in order

| Script | What it did |
|---|---|
| `launch_r3.sh` | retrain-3 screening arms S-A..S-E, 4,000 steps each from `v10pt@1500` |
| `resume_sb.sh` | resumed arm S-B |
| `launch_d0.sh` | **`r3_SBd0`**: arm S-B with `--dur-weight 0` (duration KL off), 4,000 steps |
| `final_d0.sh` | `soup3` of SBd0's last checkpoints → **candidate A** `sbd0soup_a1.0@0`; `eval_multi` |
| `mp_tts.sh`, `mp2_tts.sh` | year minimal-pair TTS sets `mp` and `mp2` (`make_tts_mp` → `filter_tts` → `mp_balance`) |
| `ws.sh`, `ws2.sh` | warm-start experiments W1/W2 from A (mixes `mp`, `v2`, `v2v`) |
| `ws3.sh` | **`r3_W3`** (mix v3, 1,000 steps, lr 3e-5, `--anchor-init`) and `soup3 --alphas 1.0,0.5 --base sbd0soup_a1.0@0` → **model B** `w3soup_a0.5@0` |
| `post3.sh` | score all candidates (`score_retrain.select`), one sealed look, `export_weights`, `swap_weights.py` into a CoreML bundle, then app-side evaluation (private) |
| `sealed2.sh` | cache the sealed beds' encoder outputs |
| `prebundle.sh` | export + bundle helper |
| `step1000.sh`, `eval2000.sh`, `evald0.sh`, `e21eval.sh`, `win_eval.sh` | evaluation passes at checkpoints |

## `retrain12/`: retrain-1 and retrain-2 (2026-10-12/13)

Arms via `modal_app.py::retrain` (`launch_arms.sh`, `launch_arms2.sh`, `launch_ablnonen.sh`), the E2 rehearsal (`e2.sh`), the `biz` TTS set (`biz.sh`), checkpoint watchers, and the YODAS shard/row lists the labeller and recrop steps used (`label_shards.txt`, `rows3_all.txt`, `recrop_batch*.txt`, `labelled_33.txt`).

## `v12/`: fast-year TTS experiment (2026-10-11), not adopted

`train.sh` holds the **exact v10pt recipe flags** (v12 = v10pt recipe + `v12year` fast-speech year TTS):
`--lspc-h 20 --no-spec-augment --icsi-h 99 --batch-duration 400 --max-steps 1500 --kl-weight 10 --kl-replay-only --kl-punct 30 --kl-punct-rows replay --kl-terms blank,gnorm`.
Result: fast-speech year site clips 6 → 18 of 20, but Earnings-22 "twenty twenty two" became "two thousand two" on all four models, so it was not adopted.
