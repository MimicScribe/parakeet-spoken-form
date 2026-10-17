# Restart here

**Asked for "another run"?** The default scope (maintainer, 2026-10-14) is in [`docs/NEXT_RUN.md`](docs/NEXT_RUN.md): spoken-form numbers in every language, in waves (multilingual first), plus the remaining English issues in [`OBSERVATIONS.md`](OBSERVATIONS.md) (2.1 short-phrase drops, 2.8 year readings across window edges, 2.6 glued blends). Start from model B. Streaming evaluation runs in a harness outside this repo; its scorers are in eval/quality/.

Cliff notes for picking up spoken-form Parakeet training. They cover where things stand, what worked, what failed, and the exact path to the next run. Details:

- The run-by-run history of both campaigns (v1–v10, retrain-1 to -4) is not in this repo; the sections below carry what a restart needs.
- [`docs/REPO_GUIDE.md`](docs/REPO_GUIDE.md): a short map of the modules by stage. [`docs/RECIPE.md`](docs/RECIPE.md) is the data recipe; [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md) lists dependencies and credentials.
- [`OBSERVATIONS.md`](OBSERVATIONS.md): B's measured strengths and weaknesses, likely causes, and ideas for the next run.
- [`runs/`](runs/): the launch scripts actually used (model B lineage, retrain-1/2, v12).
- [`docs/NEXT_RUN.md`](docs/NEXT_RUN.md): the proposed next run (spoken-form numbers in de/es/fr/it/nl/pt, Greek, English drop fix).

## 0. Current data phase (run 2, 2026-10-17)

The state above (section 1) is the model-B state. Since then the work is data only: spoken-form rows for 17 languages plus English, with long numbers cut across window seams. No training run starts before the census passes its bar and the maintainer approves that run. Order, all `modal run` commands from `pipeline/` (details in [`docs/RECIPE.md`](docs/RECIPE.md)):

1. `ml_p3.py::label2` per language: label real speech with the heard reading.
2. Judge each pass against the second transcript (`eval/quality/judge_reject.py`, `judge_plro.py`), then write the `judged_<pass>.json` marker and merge the rejects (manual unless `judge_plro.py --upload`).
3. `ml_p3.py::windows2` and `ml_p3.py::seam_crops` (`--min-words 3` for the long-number set `seam_long`).
4. `modal_app.py::build_mix4` with the run-2 flags (RECIPE section 11).
5. `eval/quality/data_gate.py <mix dir>` (census; English needs a `p3_en.jsonl` that this repo does not build).
6. If it passes and the maintainer approves: `modal_app.py::retrain3 --mix v4 --mix-dir <dir> ...`.
7. `modal_app.py::probe_heldout`, then `eval/quality/score_heldout.py` on the pulled hyps.

## 1. Where things stand (model B, 2026-10-14)

- **Released to users:** stock Parakeet Ultra (`FluidInference/parakeet-ultra-coreml@95eaa59a`), in the current public release. No fine-tune has been released yet.
- **Next release:** **model B** = `w3soup_a0.5@0`, published at `MimicScribe/parakeet-ultra-spoken-coreml@5edfc817ce69dcdf4ca7839bff5ad738e9d390b8` and queued for the next app release.
  - Checkpoint: `/vol/exp/w3soup_a0.5/step0.pt`. A local backup with SHA-256s exists outside this repo.
- **Earlier fine-tunes, pinned on main but never released:** v9pr30 (HF `06d73144`), then v10pt (HF `ccdb31c9`).
- **No corrective run from B is justified right now.** See section 5 for what would justify one; run 2 (section 0) is a separate, data-first effort.

## 2. Model B in one paragraph

Parakeet Ultra with only the prediction network and joint trained (18.1M of 627M parameters, encoder frozen). It writes English numbers as the words spoken, in the speaker's own reading ("twenty nineteen" vs "two thousand nineteen"). The other 24 languages keep their behaviour: digits stay digits, held there by a KL anchor to stock.

Lineage:

| Step | Checkpoint | How it was made |
|---|---|---|
| 0 | stock | starting point |
| 1 | `v10pt@1500` | TTS + LibriSpeech-PC + FLEURS + ICSI replay, gated blank KL, punctuation KL, no duration KL |
| 2 | `r3_SBd0` | real-speech retrain-3, 4,000 steps, `dur_weight 0` |
| 3 | `sbd0soup_a1.0@0` (candidate A) | soup of steps 3500/3750/4000 |
| 4 | `r3_W3` | 1,000 steps from A: mix v3 (minimal-pair years ×2, verbatim filter), lr 3e-5, L2-SP to the init |
| 5 | `w3soup_a0.5@0` (B) | soup of steps 500/750/1000, then α 0.5 blended back toward A |

## 3. What worked (keep)

1. **Freeze the encoder and train decoder + joint only.** The export is then a byte-exact patch of three CoreML `weight.bin` files, so users download about 36 MB, not 632 MB.
2. **Data:** Kokoro TTS of number sentences with several spoken readings per value, filtered so stock hears the carrier words. Plus real speech: LibriSpeech-PC, FLEURS train, ICSI meetings with fillers kept, and CTC-relabelled YODAS captions.
3. **SpecAugment off.**
4. **KL to STOCK on real-speech rows only:**
   - a gated blank term (`gnorm`: only where stock P(non-blank) > 0.9, normalised by the points that pass);
   - a punctuation term (weight 30, replay rows only).

   The KL teacher is stock even in warm-started runs (`add_r3_losses` copies the model before `init_from` loads).
5. **No duration KL** (`--dur-weight 0`). The duration anchor caused year and repeated-word collapse.
6. **Year minimal pairs.** Same voice, both readings ("twenty X" / "two thousand X"), repeated ×2, 15% "two thousand and X". This fixed the left-context prior that read every year as "twenty X".
7. **Soup the last checkpoints, then blend α 0.5 toward the previous winner.** This kept the year-reading gain without the full recipe's year-value losses.
8. **Keep English number-bearing rows at 35–40% of training hours.** `retrain3` asserts this; B was at 39.2%.

## 4. What failed (don't repeat)

| Tried | Result |
|---|---|
| Global (all-rows) blank/duration KL | Can't keep both spoken form and word recall. |
| Two-stage consolidation | Spoken form washed out within 100 steps. |
| Stock↔fine-tune weight soups | No α satisfied both spoken form and word recall. |
| Duration-only KL, and any duration KL | Duration-only failed outright. Any duration KL destroys year values. |
| ICSI cross-talk crops in `dominant` mode | They label the second speaker as silence, which teaches drops. |
| SpecAugment on | Conversational WER and filler regression. |
| Decode-time fixes for the year near-tie (" two" vs " twe" at about 0.50): beam, blank penalty, rescoring, duration cap | None worked. It is a greedy search error. |
| Blank-bias calibration at decode for B's drops (penalties 0.5/1/2) | Recovers words but invents about as many. |
| Downsampling a mix to save compute | Saves nothing (steps stream batches) and breaks the number share. |
| Any AMI audio in training | Its meeting series overlap the AMI evaluation sets. |

Measurement traps that produced false results:
- An evaluation that silently ran the previously installed model instead of the candidate. One "win" (v11) was this: always confirm which weights an evaluation loaded.
- TTS dev sets differed between arms, so they couldn't be compared.
- SequenceMatcher diff counts hid drops; use Levenshtein.
- A raw-model arm left a secondary English digit rewrite on.
- Number-scorer false positives: about 45% of "corrupted" items were scorer artefacts.

## 5. When to run again, and what to fold in

**A run is justified by:** a measured, model-side problem (English or multi-language) that a few-hundred-step corrective run can plausibly move. Greek alone does not qualify (maintainer decision).

**Fold in if a run happens** (details in [`OBSERVATIONS.md`](OBSERVATIONS.md) §2):
1. **Greek final ς rows restored:** `mix_build.restore_unk_nonen`, run by `patch_mix3`. Committed, not yet trained.
2. **Short-phrase drops.** Two hypotheses; test which one holds before changing anything:
   - **Data:** CE on caption labels that omit backchannels.
   - **Loss:** anchoring gaps in `ft_train.py`.

   The cheap separating test is a Levenshtein drop census along the checkpoint chain v10pt → SBd0@500/2000/4000 → A → W3@1000 → B.
3. **ICSI hand-off crops:** `data_icsi.py --overlap`, `by_turns` mode only.
4. **Mixed word+digit labels:** normalize them ("десет 000").
5. **Full v3 mix, never downsampled.**

**Gates for any new arm** (B as baseline, same evaluation build):
- the 44-call Earnings-21 number bar (`e44_bar.py` compares two runs);
- years: the blind listening set + held-out minimal pairs;
- Levenshtein content drops AND insertions on E21-44 + AMI-16, plus the live dual-channel drop census;
- FLEURS for all 25 languages, scored in a streaming evaluation harness that is not part of this repo;
- synthetic number banks;
- live invariants;
- one look at the sealed holdout.

## 6. How to restart

**Prerequisites:**
- Modal with volume `parakeet-spoken-form` and secret `hf`.
- The volume must still hold `exp/v10pt/step1500.pt`, `exp/sbd0soup_a1.0/step0.pt` and `exp/w3soup_a0.5/step0.pt`. Nothing else holds the training checkpoints: the Hugging Face repo has only the exported CoreML weights.
- Run every command from `pipeline/`.

**A. A new corrective run from B** (the likely next step):
```bash
cd pipeline
modal run modal_app.py::patch_mix3                            # runs mix_build.restore_unk_nonen (Greek ς rows back into mix3/d2, d2p)
modal run modal_app.py::build_mix3c --mp mp2 --mp-repeat 2   # minimal pairs x2 + verbatim-filtered caption rows (mix v3 inputs)
modal run modal_app.py::share_v3                              # number share must print 0.35–0.40
modal run --detach modal_app.py::retrain3 --run r4_X --mix v3 --arm S-B --dur-weight 0 \
  --init-from w3soup_a0.5@0 --anchor-init --lr 3e-5 --warmup 20 --max-steps 1000 --save-every 250
modal run modal_app.py::soup3 --run r4_X --steps 500,750,1000 --alphas 1.0,0.5 --base w3soup_a0.5@0 --name r4soup
modal run modal_app.py::eval_multi --runs "ultra,w3soup_a0.5@0,r4soup_a1.0@0,r4soup_a0.5@0" --tag _c
# pull hyps/ from the volume, then (run names get the tag appended; stock "ultra" scores as ultra_rt_c):
python score_retrain.py <pulled hyps> "ultra_rt_c,w3soup_a0.5@0_c,r4soup_a1.0@0_c,r4soup_a0.5@0_c" --base --json scores.json
# to pick a checkpoint against B, add: --select --base w3soup_a0.5@0_c
```
In `score_retrain.py`, `--base` alone turns on the FLEURS comparison against stock (`ultra_rt_c`). Only with `--select` is the value after `--base` read, as the baseline for the selection. `ultra` must be in the `eval_multi` run list: the stock default for `_c` runs is `ultra_rt_c`.
`retrain3` defaults do NOT match B's recipe: always pass `--dur-weight 0`, `--mix`, `--init-from`, and `--base` on `soup3`. The exact B commands are in [`runs/model_b/ws3.sh`](runs/model_b/ws3.sh).

**B. Rebuild the warm-start root from stock** (only if the volume is lost):
```bash
modal run modal_app.py::make_tts --name v10all --n-rows 13500 --seed 10 && modal run modal_app.py::filter_tts --name v10all
# + the 289-row sub-one supplement, merged as tts/v10pt (merge_tts); exact supplement args are only partly recorded
modal run modal_app.py::run --run v10pt --tts v10pt --lspc-h 20 --save-every 500 --no-spec-augment --icsi-h 99 \
  --batch-duration 400 --max-steps 1500 --kl-weight 10 --kl-replay-only --kl-punct 30 --kl-punct-rows replay \
  --kl-terms blank,gnorm --seed 1
```
These flags come from [`runs/v12/train.sh`](runs/v12/train.sh) (v12 = the v10pt recipe + fast-year TTS). After that, follow `runs/model_b/` in order:
- `launch_d0.sh` (SBd0)
- `final_d0.sh` (soup → A)
- `mp2_tts.sh` (minimal pairs)
- `ws2.sh` (mix v3 build)
- `ws3.sh` (W3 + soup → B)
- `post3.sh` (select, sealed look, export, bundle)

**C. Export and ship:**
1. `export_weights --run <run> --step 0`.
2. `swap_weights.py <stock bundle> decoder_joint.safetensors <out bundle>`. Use `--check` on stock tensors first: it must round-trip byte-identical.
3. Upload the three weight files to `MimicScribe/parakeet-ultra-spoken-coreml`.
4. Pin the new revision and file hashes in the downstream application that consumes the bundle.
5. The maintainer approves every upload and push.

**D. Evaluation in a downstream application** is not part of this repo. Rule that matters here: compare candidates on one frozen build, and confirm which weights each run loaded.

## 7. Not in git (needed for a restart)

- The Modal volume: checkpoints, mixes, TTS sets, beds, encoder caches.
- The `hf` secret.
- The ICSI corpus and its meeting list.
- Earnings-21 audio (for `e21local`).
- The builders for the adversarial/fast beds and `tts/gold62`.
- The grammar dumper that regenerates `pipeline/ml_grammar_table.json` (private, not in this repo).
