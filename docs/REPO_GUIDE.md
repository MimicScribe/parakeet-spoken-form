> **Read first (2026-10-14).** Extracted from the project's private plan documents; numbers and file:line cites are as those plans wrote them. Corrections that apply throughout:
> - **"Shipped" means released to users.** No fine-tune has reached users yet: the last release uses stock Parakeet Ultra. v9pr30, v10pt and model B were uploaded but never released. Model B is the one queued for the next release.
> - `$S/...` paths are scratch that no longer exists. The launch scripts that mattered are preserved in [`runs/`](../runs/).
> - Start at [`RESTART.md`](../RESTART.md).

# Restarting spoken-form Parakeet training from the repo alone

Repo audited (read-only): `./` (PUBLIC, 86 commits, branch `main`, local-only backup branch `backup/pre-strip-2026-10-13`).
Reconstructed commands are marked **[RECONSTRUCTED]**: the repo has the function signatures but not the shell history that launched them (that lived in scratch `$S/*.sh`, now gone).

---------------------------------------------------------------------------------------------------

## 0. Lineage in one screen (what "restart" means)

```
stock Ultra (ncannings/parakeet-ultra-nemo ultra.nemo @68d18c58f3d5...)   <- NeMo side; CoreML twin = FluidInference/parakeet-ultra-coreml@95eaa59a
  -> v9pr30 -> v10a/v10as2 -> v10pt@1500   (2026-10-07..10; exp/v10pt/step1500.pt; run() entry point; TTS-only number data + replay;
                                              `--kl-terms blank,gnorm`, 1500 steps, seed 1, ~$3-4/seed; EXACT flags not in repo)
  -> r3_SBd0   retrain3(arm=S-B, dur_weight=0, init v10pt@1500, mix3, 4000 steps)      ~$10 for the whole 4-arm screen + SBd0
  -> soup3 (3500,3750,4000 x alpha 1.0 ...)  => exp/sbd0soup_a1.0/step0.pt   ("A" in the docs; bundle $MODELS/parakeet-ultra-sbd0soup)
  -> r3_W3     retrain3(arm=S-B, init sbd0soup_a1.0@0, anchor_init, lr 3e-5, warmup 20, 1000 steps, dur_weight 0, mix v3)   ~$4
  -> soup3 (500,750,1000 x alpha {1.0,0.5}, base sbd0soup_a1.0@0, name w3soup) => exp/w3soup_a0.5/step0.pt  = MODEL B
  -> export_weights --step 0 -> swap_weights.py -> CoreML bundle -> HF MimicScribe/parakeet-ultra-spoken-coreml@5edfc817
```
Model B SHA-256s (bundle): Decoder weight.bin `21af365597f00e2f24e4f7249c3cf7ac4448a60257d044cbfc3a75a7ae9fe5c8` (23,604,992 B); JointDecision(v3) weight.bin `0400f7bacd38b46af51e6117c2750ba48a464a5c0cdf9519855b12087bcfbcab` (12,642,764 B). Encoder SHA-256 `315ba01f...` unchanged.

Critical consequence: **both retrain-3 warm-starts depend on `exp/v10pt/step1500.pt` on the Modal volume**, which cannot be rebuilt from this repo with certainty (see Gaps). Preserve the volume or a local backup of that checkpoint.

---------------------------------------------------------------------------------------------------

## 1. Repo map

Layout: `README.md`, `OBSERVATIONS.md`, `LICENSE` (Apache-2.0), `.gitignore`, `eval/probe_items.tsv`, `run_p3_pt.sh`, `pipeline/*`, `pipeline/gold/*`. Untracked clutter at repo root: `p3_*.log` (11 Modal run logs for retrain-4 P3), `.serena/`. There is NO `requirements.txt`, `pyproject.toml`, or lockfile; dependencies are pinned only inside the Modal image definitions (`modal_app.py`, `ml_modal.py`, `ml_p3.py`).

Stage legend: A=env/infra, B=English TTS data, C=real-speech scan/fetch/label, D=non-English data, E=mix build, F=train, G=soup/export, H=eval beds + decode, I=scoring, J=bundle/ship, K=retrain-4 multilingual P3, L=research/diagnostic.

### Orchestration (Modal)
| File | What | Stage |
|---|---|---|
| `pipeline/modal_app.py` (61 KB, ~1,460 lines) | The Modal app `parakeet-spoken-form`: every GPU/CPU entry point (list in section 3). Defines 5 images (`nemo_image`, `tts_image`, `scan_image`, `mix_image`, plus per-class use), the `/vol` volume, `hf` secret, retrain mix definitions (`retrain_mix`, `retrain3_mix`, `retrain3_mix_mp`, `retrain3_mix_v2`, `retrain3_mix_v3`), `R3_ARMS`, `RETRAIN_K`. | A-J |
| `pipeline/ml_modal.py` | Separate Modal app `parakeet-ml` (retrain-4 pilot P2): `p2_lang` per-language CTC reading decision on gold + real D2 rows; local entrypoint `p2 --gold <dir>`. | K |
| `pipeline/ml_p3.py` | Separate Modal app `parakeet-ml-p3` (retrain-4 step P3, no training): `scan_lang` (CPU) + `label_lang` (L4); entrypoints `scan`, `label`. Writes `/vol/ml/p3/<lang>/{cands.jsonl,rows.jsonl,stats.json,wav/}`. | K |
| `run_p3_pt.sh`, `pipeline/run_p3_{scan,label,all,clean}.sh` | Thin shell wrappers for `ml_p3.py::scan/label` (env `P3LANGS P3HOURS P3FILES P3LIMIT P3NEWH P3LOG`) and `run_p3_clean.sh` which `modal volume rm`s `ml/p3/<l>/{wav,rows.jsonl}` for de/es/fr/it/nl/pt. `run_p3_pt.sh` = the Brazilian-Portuguese redo. | K |

### Data: English synthetic (TTS)
| File | What / inputs / outputs |
|---|---|
| `readings.py` (28 KB) | Spoken-form readings of numbers/symbols: `cardinal ordinal digits pair year` and ~50 `k_*` kind functions (`k_year`, `k_year_any`, `k_year_bal`, `k_two_k_qty`, `k_money`, `k_ip`, `k_disfluent`, `k_twenty_pause`, ...). Orthography contract: lowercase words, tens/ones spaced, no hyphens, "am/pm" words. `SUB_ONE_P` (0.1 default) changes full regenerations. |
| `carriers*.txt` (6 files) | Sentence templates with `{slot}` markers: `carriers.txt` (main), `carriers_year.txt`/`carriers_year_b.txt` (fast-speech year supplement), `carriers_d9.txt` (far-year/disfluent), `carriers_biz.txt` (business shorthand: quarter/identifier/big/money/thousands_k/multiplier/percent), `carriers_mp.txt` (minimal-pair year carriers). Baked into the TTS image via `add_local_file`. |
| `gen.py` | `rows(n, seed, multi_frac, slots, control_frac, carriers)` expands carriers x readings into `{text,tts_text,template,kinds,split}`; dev rows use hash-held-out carriers (`dev_carriers`); `minimal_pairs(n_pairs, seed, and_frac)` (R3.31: each filled carrier voiced as "twenty X" AND "two thousand X", years 2010-2039). CLI: `python gen.py --rows N --seed S --out rows.jsonl`. |
| `voice.py` | `voice_rows(rows, out_dir)`: Kokoro-82M (kokoro==0.9.4) -> 16 kHz wav + NeMo manifest lines; speed ranges per row, voice split train/dev. |
| `ft_eval.filter_tts` (in ft_eval.py) | Drops TTS clips whose carrier words stock does not hear (>=90% literal words); writes `train_ok.jsonl`/`dev_ok.jsonl`. |

### Data: English real speech (replay + number crops)
| File | What |
|---|---|
| `data_e22.py` | Earnings-22 test (`hf-audio/open-asr-leaderboard`, earnings22 parquet) -> `/vol/e22/{wav,all.jsonl,numbers.jsonl}`. EVAL only. |
| `data_replay.py` | LibriSpeech-PC (OpenSLR 145 manifests + OpenSLR 12 train-clean-100/dev-clean) -> `/vol/replay/{lspc_train,lspc_train_numwords,lspc_dev}.jsonl` (digit-free, US spelling, title relabel). |
| `data_icsi.py` | LOCAL script: window-shaped crops of ICSI meetings (needs the licensed corpus on disk + a `meetings.json`), `python data_icsi.py <icsi_dir> <meetings.json> <out_dir> [hours] [--overlap]`; upload with `modal volume put parakeet-spoken-form <out_dir> replay/icsi`, then move manifest to `replay/icsi_train.jsonl`. `by_turns` vs `dominant` cross-talk modes (never use `dominant`). |
| `s0_scan.py` | Stage 0: caption-only scan of `espnet/yodas-granary` parquet shards (column projection; audio never fetched), flags number rows + 2% number-free sample -> `/vol/s0/scan/<lang>/<shard>.jsonl` + `.stats`. `pick_shards`, `scan_shard`. |
| `fetch.py` | S3: `fetch_shard` (download whole shard, keep scanned rows, write 16 kHz wavs -> `/vol/yodas/<lang>/{wav,manifests}`), `fetch_plain` (D1p number-free rows of an already-labelled shard -> `/vol/yodas/en_plain/`). |
| `labeller.py` | English candidate generator: caption number spans -> every plausible spoken reading (`spans`, `render`, `render_ranges`, `int_readings`, `money_readings`, `clock_readings`, `spelled_cands`, `row_ok`). Has doctests. CPU. |
| `label_gpu.py` | S3 relabel on GPU: stock precheck, `facebook/wav2vec2-large-960h-lv60-self` CTC picks the reading by margin, absolute-fit check, forced-alignment window crops. `Scorer`, `label_shard`, `validate`, `decide_text`. Writes `/vol/d1/rows/<shard>.jsonl`, `/vol/d1/crops/<shard>.jsonl` (rows3 / crops dirs for retrain-3 relabel). |
| `recrop.py` | Retrain-3 re-crop at quiet word gaps + number-onset sub-crops (`W2VScorer`, `recrop_shard`) -> `/vol/d1/crops2` -> `crops3`. |
| `veto.py` | HuBERT-large (`facebook/hubert-large-ls960-ft`) second acoustic veto on decided readings -> `/vol/d1/veto*/<tag>.json`. |
| `filter3.py` | CPU filters over recropped shards: D-1b edge agreement, D-4, label assert, unit abbreviations, F1 filler restoration, F2 report. `python filter3.py <root>` where root has `crops3/ dec3/ rows3/ veto3/`; writes `crops3_ok/` + `filter3_report.json`. |
| `restore_fillers.py` | Re-inserts fillers/repeats captions dropped, from v10pt's own decode (insertions only). `python restore_fillers.py <S>` - hard-coded to a scratch layout `<S>/rt/mix/<part>.jsonl` (see Gaps). |
| `restore_d1p.py` | Same restoration for D1p: `python restore_d1p.py <plain3.jsonl> <plain_all_v10.json> <out plain3_ok.jsonl>`. |
| `yearflip.py` | Metric (not data): year-reading flips across overlapping app windows on `e21_dev` (precursor of "twenty thousand N" blends). CLI `python yearflip.py <hyps.jsonl> <names>`. |

### Data: non-English
| File | What |
|---|---|
| `data_fleurs.py` | FLEURS TRAIN split (24 langs, `refs/convert/parquet`) -> `/vol/replay/fleurs/<lang>/*.wav` + `replay/fleurs_train_multi.jsonl` (rows with no written-form chars; 0.4 h/lang default). Test split never used. NB: wavs named by FLEURS id collide across speakers (fixed after-the-fact by `audit_mix3`/`clean_manifest`). |
| `harvest.py` | D2/D2p: VoxPopuli train (CC0) + YODAS asr_only, pseudo-labelled by STOCK Ultra (digit rows D2 / plain rows D2p), `langid` LID audit -> `/vol/d2/<lang>/{digit,plain}.jsonl` + `audit.jsonl`. |

### Mix, train, soup, export
| File | What |
|---|---|
| `mix_build.py` (37 KB) | `build` (retrain-2 -> `/vol/mix/`), `build3` (retrain-3 -> `/vol/mix3/`), `patch3`, `trim_d1`, `restore_unk_nonen` (Greek final-sigma fix, 2026-10-14), `build3b` (-> `/vol/mix3b/`, 2000-2009 cap undone, per-video cap), `build3c` (min-pairs x repeat + verbatim-filtered d1p/d10p), `mix_share(vol, mix)` (number-bearing hour share; the 35-40% rule), `trim_edges`, `Degrader` (opus/phone-band/SLR28 noise/RIR aug on ~25% of rows). Eval dedupe by 8-gram text + wav basename against e22/fleurs/vox/adv/e21_dev/e21x refs. |
| `ft_train.py` (53 KB) | `train(vol, ultra, run, mix, max_steps, lr, ..., init_from, kl_*, r3, wf_weight, en_weight, l2sp_c, l2sp_anchor, accumulate, dur_weight, stage_dir)`. NeMo Lightning fine-tune of decoder+joint only (encoder frozen, ~18M of 627M params). Losses: RNNT/TDT + KL to STOCK teacher (`add_blank_duration_kl`, `add_r3_losses`: per-tag blank/token/punct KL, wf_pen (written-form pieces), en_pen, anti-blend, optional duration KL `dur_weight`), decoupled L2-SP (`L2SP`), `HoldRows`, `NoDropout`, `SaveEvery` (`step<N>.pt` + `.done` marker), resume from `exp/<run>/last.ckpt`. Mix entries are `(manifest, hours, tag)` with tag in `free|anchor|anchor_num|span|nonen`. Writes `exp/<run>/{step<N>.pt, decoder_joint.pt, train_manifest.jsonl, last.ckpt}`. |
| `common.py` | `NUMBER_WORDS`, `WRITTEN_FORM`, `SPAN_CORE/LINK`, `number_span()` shared by trainers/filters/scorers. |
| `preflight.py` | v10b pre-flight (stage A span-mask audit + drift vs stock; stage B micro-train). Modal `preflight`. L. |
| `swap_weights.py` | LOCAL. `python swap_weights.py <ultra_coreml_dir> <decoder_joint.safetensors> <out_dir> [--check]`: rewrites the fp16 regions of `Decoder.mlmodelc/weights/weight.bin`, `JointDecisionv3.mlmodelc/weights/weight.bin` and copies the joint to `JointDecision.mlmodelc/weights/weight.bin` (handles LSTM gate reorder [i,f,g,o] -> [i,f,o,g], b_ih+b_hh). `--check` on stock tensors must reproduce the bundle byte-for-byte. Needs numpy + safetensors only. |
| `rescore.py` | Diagnostics: `run` (number re-check dry run), `score_texts` (TDT NLL of candidate texts), `greedy_vs_beam`, `cut_decode` (seam re-decode placement). Modal `rescore_dryrun`, `score_texts`, `seam_decode`, `year_beam`. Research, not needed to retrain. |

### Eval beds, decode, scorers
| File | What |
|---|---|
| `beds.py` | Builds eval beds under `/vol/beds/`: `vox`/`vox3` (VoxPopuli TEST, dev + SEALED 2/3 by id hash, LID+dedupe), `fleurs`/`fleurs3` (FLEURS TEST; wer=100/lang, num=digit utterances, wer_full=sealed), `noise` (OpenSLR 28 isotropic noises), `slr28` (downloads `rirs_noises.zip`), `e21_ref`/`e21x` (Earnings-21 refs from `revdotcom/speech-datasets` via GitHub API; 33 sealed calls cut in 20 s chunks), `e21_dev_cut` (11 local E21 calls as app-geometry windows; needs `/vol/e21local/audio/<fid>.wav` uploaded by hand). |
| `ft_eval.py` | `load_model`, `decode`, `transcribe`, `transcribe_loaded` (-> `/vol/hyps/<run>/<name>.jsonl`), `eval_sets`, `eval_retrain`, sets lists `RETRAIN_SETS`, `RETRAIN3_SETS`, `SEALED3_SETS`, encoder cache `encode_rows/cache_beds/eval_cached` (-> `/vol/enc/`), `filter_tts`, `e22_other`. |
| `score_retrain.py` (27 KB) | THE scorer, local CPU: `python score_retrain.py <hyps_root> <run,run,...> [--json out] [--exclude-from stock,v10] [--split dev|sealed] [--stock <name>] [--base] [--fleurs-base X] [--e21x-refs <dir>]`. Per run: WER e22_other / lspc_dev / d1_dev / fleurs_wer, English number-word leak (vox_dev, fleurs_num), digit emission, noise-bed numbers, adversarial value accuracy, year readings/values/truncation/garbles, blends/chimeras, filler rate, FLEURS paired bootstrap (`fleurs_bootstrap`), `score_sealed`. `SELECT_ORDER` + `select()` = the frozen model-selection ladder (blend, en_wf, chimeras, year_value, year_trunc, year_garble, adv_en_ok, fast_d1_ok, nonen_leak, fleurs_worst, wer_dev). |
| `e44_bar.py` | `python e44_bar.py <base_arm_dir> <cand_arm_dir>`: paired 44-call Earnings-21 number bar (exact/corrupted/dropped/invented/high-severity/blends/splices) over two runs' score files and transcript rows (input format in the script's docstring). |
| `score_mp.py` | Reading accuracy on held-out minimal pairs: `python3 score_mp.py <hyps_root> run1,run2 <tts/mp2/dev_ok.jsonl>`. |
| `score_gold.py` | Gold-set harm check: `python3 score_gold.py <hyps_root> run1,run2 <gold62 manifest>`. |
| `score_side.py` | Per-side year-reading accuracy over existing hyps: `python score_side.py <root> <runs> <beds>`. |
| `gold/owner_year_readings.json` | 62 listener-verified year clips (ground truth reading + heard value), keys "E21 <call> @<s>s|<year>". `gold/save_owner_years.py <listen_dir>` merges listening-page answers into it. Audio is NOT in repo. |
| `eval/probe_items.tsv` | 66 spoken-form probe items (59 formats + 7 controls; id, cat, spoken, expect). **Referenced by README but not read by any script in the repo.** |

### Tests
`test_r3.py` (pytest, imports labeller/filter3/mix_build/score_retrain/restore_fillers), `test_ml_cands.py` (plain script; `python test_ml_cands.py`), `test_ml_p3.py` (plain script). All CPU, no GPU.

### Multilingual (retrain-4 P1-P3; DATA only, no training exists)
| File | What |
|---|---|
| `ml_cands.py` | Number-grammar spoken-reading candidates for es/fr/de/pt/it/nl (`spans`, `render`, `needed`, `build_table`). Reads `ml_grammar_table.json`. |
| `ml_grammar_table.json` (200 KB) | Pre-dumped canonical readings from per-language number grammars (pt is Brazilian); the dumper is not in this repo. Regenerated via `build_table(texts_by_lang, dumper, workdir)`. |
| `ml_decide.py` | Per-language xlsr-53 CTC (`jonatasgrosman/wav2vec2-large-xlsr-53-{german,spanish,french,italian,portuguese,dutch}`) reading decision: `LangCTC`, `decide`. |
| `ml_p3_core.py` | Row building/CER/edge-quiet helpers (`make_row`, `Acc`). |

---------------------------------------------------------------------------------------------------

## 2. End-to-end restart path

Assumptions: macOS dev machine (the CoreML step needs a Mac), Modal for all GPU work. All `modal run` calls are from `pipeline/` (the images use `add_local_python_source` relative to cwd) or from repo root as `modal run pipeline/modal_app.py::...` (the module docstring uses the latter; `ml_p3` scripts `cd` into `pipeline/`). Modal CLI flag form: `--arg-name value` (underscores become dashes); booleans as `--flag` / `--no-flag`. Add `--detach` for anything >10 min.

### 2.1 Environment
1. **Python (local)**: Python 3.12 venv (`.venv` is gitignored). Minimum for the local scripts: `pip install modal safetensors numpy soundfile jiwer pytest` (score_retrain/filter3/mix tests are stdlib + numpy; `swap_weights` needs numpy+safetensors; `data_icsi` needs numpy+soundfile+ElementTree). There is no pinned local requirements file.
2. **Modal**: `pip install modal && modal setup` (workspace in the logs: `r-marshall-adams`, environment `main`). Volume is created on first use: `modal.Volume.from_name("parakeet-spoken-form", create_if_missing=True, version=2)` (volume format v2). Verify: `modal volume ls parakeet-spoken-form`.
3. **Secrets** (names only): Modal secret **`hf`** (a HuggingFace token with read access; used by `s0_scan_shard`, `s3_fetch_shard`, `d1p_fetch_shard`, `Labeller`, `d2_harvest_lang`, `build_beds`, `build_beds3`, `ml_p3.scan_lang`). Create: `modal secret create hf HF_TOKEN=...` (do it interactively; never commit the value). `hf auth login` locally for the final upload. No other API keys are used by this repo.
4. **Base model**: downloaded lazily by `ultra_path()` from HF `ncannings/parakeet-ultra-nemo` file `ultra.nemo` pinned to revision `68d18c58f3d5012f885f4e01039684cd398aff19`, symlinked to `/vol/ultra.nemo`; stock decoder+joint extracted once to `/vol/export/ultra/state.pt` by `ultra_state()`. Smoke test: `modal run pipeline/modal_app.py::env_check` (L4; prints trainable params by group, decoding strategy).
6. Images: `nemo_toolkit[asr,cu12]==2.7.3`, `torch==2.8.0` cu128, CUDA 12.8.1 devel Ubuntu 24.04, Python 3.12 (modal_app.py). TTS image: `kokoro==0.9.4`, `misaki[en]>=0.9.4`, spaCy `en_core_web_sm`, espeak-ng. ml images: torch 2.4.1 + transformers 4.44.2 (Python 3.11).
7. Unit tests before anything expensive: `cd pipeline && python -m pytest test_r3.py -q && python test_ml_cands.py && python test_ml_p3.py`.

### 2.2 Eval/forgetting data and beds (cheap; do these first, they gate everything)
```
modal run pipeline/modal_app.py::prep_e22                    # /vol/e22  (CPU, ~minutes)
modal run pipeline/modal_app.py::prep_replay                 # /vol/replay LibriSpeech-PC (CPU, <3 h timeout)
modal run pipeline/modal_app.py::prep_fleurs                 # /vol/replay/fleurs + fleurs_train_multi.jsonl (--hours-per-lang 0.4)
modal run pipeline/modal_app.py::build_beds --which "noise,fleurs,e21x,vox"    # v1 beds; noise also pulls SLR28 -> /vol/slr28
modal run pipeline/modal_app.py::build_beds3 --which "vox,fleurs,e21"          # retrain-3 beds (e21 needs /vol/e21local/audio first)
```
Manual uploads the code expects but does not create: `/vol/e21local/audio/<fid>.wav` for the 11 `E21_LOCAL` calls (4320211 4341191 4346818 4359971 4365024 4366522 4366893 4367535 4383161 ...; from the local Earnings-21 corpus: `modal volume put parakeet-spoken-form <dir> e21local/audio`); `/vol/beds/adv/all.jsonl` and `/vol/beds/fast/all.jsonl` (adversarial + fast-speech beds - NO builder in repo); ICSI (`data_icsi.py` locally then `modal volume put`). Then cache encoder outputs once: `modal run pipeline/modal_app.py::cache_beds3 [--sealed]` (L4, 64 GB RAM).

### 2.3 Synthetic TTS data (English; Kokoro, fan-out on L4)
```
modal run pipeline/modal_app.py::make_tts --name v10all --n-rows 13500 --seed 10      # historical v10all (13.5k rows)
modal run pipeline/modal_app.py::filter_tts --name v10all                              # -> tts/v10all/{train,dev}_ok.jsonl (stock-heard filter)
# supplements used by retrain-3 mixes (names must match retrain3_mix): [RECONSTRUCTED - n_rows/seed not in repo]
modal run pipeline/modal_app.py::make_tts --name v10pt  ...  # v10all + targeted sub-one rows: --slots <...> --sub-one (readings.SUB_ONE_P=1.0); merged with merge_tts
modal run pipeline/modal_app.py::make_tts --name d9sup --carriers carriers_d9.txt ...
modal run pipeline/modal_app.py::make_tts --name biz   --carriers carriers_biz.txt --slots quarter,identifier,big,money,thousands_k,multiplier,percent ...
modal run pipeline/modal_app.py::merge_tts --name v10pt --parts a,b
modal run pipeline/modal_app.py::make_tts_mp --name mp2 --n-pairs <N> --seed <S> --and-frac 0.15   # 1,309 train + 542 dev pairs after filter
modal run pipeline/modal_app.py::filter_tts --name mp2 ; modal run pipeline/modal_app.py::mp_balance --name mp2
```
Known flags (from signatures): `make_tts(name, n_rows, seed=1, shard=200, multi_frac=0.0, slots="", sub_one=False, control_frac=0.1, carriers="carriers.txt", speed="")`; `speed "lo,hi"` e.g. `"1.15,1.5"`; `make_tts_mp(name="mp", n_pairs=1500, seed=31, shard=200, and_frac=0.0)`. The plans quote `make_tts v12year --carriers carriers_year.txt --speed 1.15,1.5 --multi-frac 0.3 --n-rows 1800 --seed 1201`.

### 2.4 Real-speech English (YODAS-Granary) -> D1/D1p/D10
```
modal run pipeline/modal_app.py::s0_scan --langs en,... --en-shards 150 --big-shards 25     # caption scan, CPU, up to 4 h; -> /vol/s0/scan, stats json
modal run pipeline/modal_app.py::s3_fetch --lang en [--limit N]                              # audio, cpu=2 x <=40 containers
modal run pipeline/modal_app.py::s3_label --lang en --rows-dir d1/rows3 --labelled-only      # L4 x8; Labeller: wav2vec2 CTC picks reading (retrain-3 relabel)
modal run pipeline/modal_app.py::s3_veto --rows-dir d1/rows3 --out-dir d1/veto3              # HuBERT veto (L4 x6)
modal run pipeline/modal_app.py::s3_recrop --rows-dir d1/rows3                               # quiet-gap crops -> d1/crops2 (retrain-3 data fix)
modal run pipeline/modal_app.py::s3_decode_crops --crops-dir d1/crops3 --out-dir d1/dec3     # stock + v10pt greedy decodes (REQUIRES exp/v10pt/step1500.pt)
modal volume get parakeet-spoken-form /d1 <root>   # pull crops3 dec3 rows3 veto3 locally, then:
python pipeline/filter3.py <root>                  # -> <root>/crops3_ok, filter3_report.json ; re-upload crops3_ok to /d1/crops3_ok
modal run pipeline/modal_app.py::d1p_fetch ; modal run pipeline/modal_app.py::d1p_filter     # D1p number-free rows -> /vol/d1/plain3.jsonl (CER<=0.15)
modal run pipeline/modal_app.py::decode_manifest --manifest d1/plain3.jsonl --out d1/plain3_v10.json   # v10pt decode for filler restore
python pipeline/restore_d1p.py plain3.jsonl plain_all_v10.json plain3_ok.jsonl                # then upload as d1/plain3_ok.jsonl
```
(`s3_label` has a hidden coupling: default `rows_dir=d1/rows`/`write_crops=True` is the v1 path; the retrain-3 relabel used `--rows-dir d1/rows3 --no-write-crops --labelled-only`; confirm against commit 775c962 / d33114f before relying on defaults.)

### 2.5 Non-English (D2/D2p)
```
modal run pipeline/modal_app.py::d2_harvest --langs bg,cs,da,de,el,es,et,fi,fr,hr,hu,it,lt,lv,nl,pl,pt,ro,ru,sk,sl,sv,uk    # stock-labelled digit/plain rows, L4 each, <=8 h
```
Retrain-4 multilingual spoken-form labelling (data only, nothing trained on it):
```
cd pipeline && modal run ml_p3.py::scan  --langs de,es,fr,it,nl,pt --hours 4.5 --max-files 300
cd pipeline && modal run ml_p3.py::label --langs de,es,fr,it,nl,pt --new-hours 1.5          # per-lang xlsr CTC decide; /vol/ml/p3/<lang>/rows.jsonl
```
(`run_p3_*.sh` are wrappers; `run_p3_pt.sh` is the Brazilian-Portuguese rerun.)

### 2.6 Build the mix (CPU, mix_image 8 cpu / 32 GB)
```
modal run pipeline/modal_app.py::build_mix3                    # -> /vol/mix3 (d1, d1p, d10n, d10p, d2, d2p ...); asserts eval texts present
modal run pipeline/modal_app.py::clean_manifest                # replay/fleurs_train_multi.jsonl -> mix3/fleurs_train_multi.jsonl (drops id-collision rows)
modal run pipeline/modal_app.py::audit_mix3                    # wav exists, 16 kHz, duration==manifest; rewrites bad parts (keeps .pre_audit.jsonl)
modal run pipeline/modal_app.py::patch_mix3                    # year-ratio cap, label-noise drops, Greek/thin-language unk rows (needs d1/dec3)
modal run pipeline/modal_app.py::build_mix3b --per-video 12    # -> /vol/mix3b (2000-2009 cap undone) + prints SHARE
modal run pipeline/modal_app.py::build_mix3c --mp mp2 --mp-repeat 2   # -> tts/mp2/train_ok_x2.jsonl, mix3b/d1p_v.jsonl, d10p_v.jsonl
modal run pipeline/modal_app.py::share_v3                      # prints SHARE for retrain3_mix_v3; must be in [0.35, 0.40] (39.2% last time)
```
The mix spec for model B's last stage is `retrain3_mix_v3()` = `retrain3_mix_v2(verbatim=True)` with `tts/v10pt` at 4 h, `replay/lspc_train` at 40 h, minimal pairs x2. `restore_unk_nonen` (2026-10-14) fixes Greek final sigma but **has not been trained** (OBSERVATIONS 2.2).

### 2.7 Train
All training uses A100-40GB, `retrain3` timeout 24 h, `retries=2` with resume from `exp/<run>/last.ckpt`.

* Step 1 - warm start dependency: `exp/v10pt/step1500.pt` (and `exp/v10pt/train_manifest.jsonl`) must exist on the volume. If rebuilding from stock, use the older `run` entry point:
  `modal run pipeline/modal_app.py::run --run v10a --tts v10all [--kl-terms blank,gnorm --kl-weight <v9pr30 value> --kl-replay-only --kl-punct 30 --kl-punct-rows replay --max-steps 1500 --seed 1 ...]` **[RECONSTRUCTED - the v9pr30 flag set is only described in prose in private plan documents]**; v10pt = v10a recipe + 289 targeted TTS rows ("sub-one" amounts) = `tts/v10pt`, seed 1, step 1500. ~$3-4 per seed.
* Step 2 - screening/main run (candidate A lineage), **[RECONSTRUCTED from plan R3.31 / SBd0]**:
  `modal run --detach pipeline/modal_app.py::retrain3 --run r3_SBd0 --arm S-B --max-steps 4000 --save-every 250 --dur-weight 0 --init-from v10pt@1500 --mix mix3`
  Arms: `R3_ARMS = {S-A: wf1/l2sp1/lr1e-4, S-B: wf3/l2sp1/lr1e-4, S-C: wf1/l2sp0, S-E: wf1/l2sp1/lr5e-5}`. Defaults baked into `retrain3`: `batch_duration=200`, `accumulate=2`, `spec_augment=False`, `kl_weight=10.0`, `kl_punct=30.0`, `kl_tok=1.0`, `warmup=50`, `seed=1`, number-share assert 0.35-0.40 (`--no-check-share` to skip), L2-SP anchor = `exp/v10pt/step1500.pt` unless `--anchor-init`. Rehearsal: tiny `--max-steps` with `--gradshare` (R3_GRADSHARE=1 logs per-term gradient share on the joint last layer).
* Step 3 - soup the tail checkpoints: `modal run pipeline/modal_app.py::soup3 --run r3_SBd0 --steps 3500,3750,4000 --alphas 1.0,0.85,0.7,0.5 --base v10pt@1500 --name sbd0soup` -> `exp/sbd0soup_a<alpha>/step0.pt`. Winner `sbd0soup_a1.0@0` (picked by `score_retrain.select`).
* Step 4 - model B warm-start, **[RECONSTRUCTED from plan R3.31f/`$S/ws3.sh`]**:
  `modal run --detach pipeline/modal_app.py::retrain3 --run r3_W3 --arm S-B --init-from sbd0soup_a1.0@0 --anchor-init --lr 3e-05 --warmup 20 --max-steps 1000 --save-every 250 --dur-weight 0 --mix v3` (saves step 250/500/750/1000; mix v3 needs the section 2.6 build first).
* Step 5 - final soup/alpha blend: `modal run pipeline/modal_app.py::soup3 --run r3_W3 --steps 500,750,1000 --alphas 1.0,0.5 --base sbd0soup_a1.0@0 --name w3soup` -> `exp/w3soup_a1.0/step0.pt`, `exp/w3soup_a0.5/step0.pt`. The blend is toward the PRIOR model (`theta = base + a*(soup-base)`), except the joint's last-layer output rows for written-form pieces (`ft_train.wf_piece_ids`) keep the soup's values (a=1) so blending cannot undo the digit suppression.
* Cost/time (from the project's plan; not in repo): relabel ~$3, recrop+decode ~$2, D1p ~$2, TTS ~$0.50, beds ~$1, rehearsal ~$0.30, each 4000-step screening arm ~$1.50 to step 1000 and ~$8 for the remaining 3000 steps of two seeds, evals+alpha ~$3; whole retrain-3 compute ~$28 estimated, about $10 spent through SBd0, W3 ~$4, warm-start experiments ~$1 each, hard cap $60. Wall clock to a scored candidate ~6-7 h plus ~1.5 h for the 44-call app run.

### 2.8 Evaluate on the NeMo side (L4)
```
modal run pipeline/modal_app.py::eval_multi --runs "ultra,v10pt@1500,sbd0soup_a1.0@0,w3soup_a0.5@0" [--sealed] [--tag _c] [--skip e21_dev]
   # decodes every cached bed per run; hyps -> /vol/hyps/<run>@<step><tag>/<bed>.jsonl ; a "<path>.pt" entry also works
modal run pipeline/modal_app.py::eval_mp_dev --runs "ultra,w3soup_a0.5@0" --name mp2      # minimal-pair dev -> hyps/<run>/mp_dev.jsonl
modal run pipeline/modal_app.py::eval_mp_dev --runs ... --name gold62                        # listener-verified gold clips (needs tts/gold62/dev_ok.jsonl; no builder in repo)
modal volume get parakeet-spoken-form /hyps <local_hyps>
python pipeline/score_retrain.py <local_hyps> ultra_rt,v10pt@1500,w3soup_a0.5@0 --stock ultra_rt --base --json scores.json
python pipeline/score_retrain.py <local_hyps> w3soup_a0.5@0 --split sealed --stock ultra_rt --e21x-refs <refs>   # SEALED: once only
python3 pipeline/score_mp.py <local_hyps> w3soup_a0.5@0 <mp2 dev_ok.jsonl> ; python3 pipeline/score_gold.py ...
```
Older single-model paths: `eval_ckpts(run, steps, tts)`, `eval_retrain(run, step, sealed, r3)`, `baseline`. Selection among candidates: `score_retrain.select(res, runs, baseline)` walks `SELECT_ORDER` (frozen); sealed beds scored once for the winner only. Ship bar highlights: non-English WER <= stock + 0.15 pp (paired bootstrap per FLEURS language), number share 35-40%, zero `twenty thousand` blends, maintainer year clips not flipped.

### 2.9 Export to CoreML bundle (local Mac)
```
modal run pipeline/modal_app.py::export_weights --run w3soup_a0.5 --step 0      # -> /vol/export/w3soup_a0.5@0/decoder_joint.safetensors (fp32, NeMo names, decoder.* + joint.*)
modal volume get parakeet-spoken-form "export/w3soup_a0.5@0/decoder_joint.safetensors" ./decoder_joint.safetensors
python pipeline/swap_weights.py $MODELS/parakeet-ultra-stock-95eaa59a ./decoder_joint.safetensors $MODELS/parakeet-ultra-bestN
python pipeline/swap_weights.py <stock bundle> <stock decoder_joint.safetensors> /tmp/chk --check   # sanity: must be byte-identical
```
(Stock-tensors file for `--check`: `export_weights --run ultra`.) Bundle layout is the unchanged FluidInference layout: only `Decoder.mlmodelc/weights/weight.bin`, `JointDecision.mlmodelc/weights/weight.bin` and `JointDecisionv3.mlmodelc/weights/weight.bin` differ (the last two are byte-identical copies); encoder/preprocessor/tokenizer reused. The NeMo-vs-CoreML token-for-token check (README eval item 5) is **not scripted in this repo**.
Hash the three weight files (`shasum -a 256`) and record them.

### 2.10 Upload to HF (maintainer action)
* Stage a folder with the 3 weight files at their bundle-relative paths + README (model card front matter `license: cc-by-4.0`, `base_model: FluidInference/parakeet-ultra-coreml`, `library_name: coreml`, tags `parakeet tdt coreml spoken-form`).
* `hf auth login` (maintainer) then `hf upload MimicScribe/parakeet-ultra-spoken-coreml <staging> . --commit-message "..."` (B = commit `5edfc817ce69dcdf4ca7839bff5ad738e9d390b8`; earlier v10pt = `ccdb31c9`).

---------------------------------------------------------------------------------------------------

## 3. Modal specifics

**Apps**: `parakeet-spoken-form` (modal_app.py, the main one), `parakeet-ml` (ml_modal.py, image debian_slim py3.11), `parakeet-ml-p3` (ml_p3.py).
**Volume**: `parakeet-spoken-form` (v2 format), mounted at `/vol`. `HF_HOME=/vol/hf` in nemo_image (several functions override to `/tmp/hf` because HF/xet logs on the volume block `commit/reload`). **Secret**: `hf`.
**GPU choices**: A100-40GB (cpu=8, mem 32 GB) for `killtest`, `run`, `retrain`, `retrain3`; L4 for everything else GPU (eval, decode, labeller, recrop, veto, harvest, TTS, beds); `eval_multi`/`cache_beds3` L4 with 64 GB RAM; CPU-only for scans, mix build, soups, export.

Functions (`modal run pipeline/modal_app.py::<fn> --arg value`):
| Function | Key args (defaults) | Hardware / timeout | Purpose |
|---|---|---|---|
| `env_check` | - | L4 / 30 min | Load Ultra, freeze encoder, print trainable params |
| `prep_e22`, `prep_replay`, `prep_fleurs(hours_per_lang=0.4)` | | cpu 4-8 / 1-3 h | Eval + replay data |
| `make_tts(name,n_rows,seed=1,shard=200,multi_frac=0,slots="",sub_one=False,control_frac=0.1,carriers="carriers.txt",speed="")` | | tts_image; fans out `voice_shard` (L4, 1 h each) | Kokoro TTS sets -> `/vol/tts/<name>` |
| `make_tts_mp(name="mp",n_pairs=1500,seed=31,shard=200,and_frac=0)`, `mp_balance(name)` | | same | Year minimal pairs |
| `filter_tts(name)`, `merge_tts(name,parts)` | | L4 / CPU | TTS cleanup -> `*_ok.jsonl` |
| `killtest`, `baseline`, `run`, `eval_ckpts`, `retrain` | see signatures | A100 / L4 | Old (v9/v10) training + eval path; `retrain` = real-speech retrain-2 (arms main/stockinit/abl_en/abl_nonen) |
| `retrain3(run, arm="S-A", seed=1, max_steps=4000, save_every=250, en_weight=1.0, check_share=True, warmup=50, gradshare=False, dur_weight=1.0, init_from="v10pt@1500", lr=0.0, mix="mix3", anchor_init=False)` | `mix` in {mix3,mp,v2,v2v,v3} | A100-40GB, 24 h, retries=2 | Retrain-3 trainer. **Note default `dur_weight=1.0`; every candidate that shipped used `--dur-weight 0` (duration KL destroys year values).** |
| `soup(src,alphas,name)`, `seedsoup(srcs,name,step,weights)`, `soup3(run,steps="3500,3750,4000",alphas="1.0,0.85,0.7,0.5",base="v10pt@1500",name="")`, `shift_blank(run,step,deltas)` | | cpu / 30 min | Weight averaging, alpha blending, blank-bias shift |
| `export_weights(run="ultra",step=-1)` | `--step 0` for soups | cpu | decoder+joint safetensors |
| `eval_multi(runs,sealed=False,tag="",skip="")`, `cache_beds3(sealed=False)`, `eval_retrain(run,step=0,sealed=False,r3=False)`, `eval_mp_dev(runs,name="mp")` | | L4, 4-6 h | Eval |
| `s0_scan(langs,en_shards=150,big_shards=25)` (+`s0_scan_shard` x64), `s3_fetch(lang,limit)`, `s3_label(...)` (Labeller L4 x8), `s3_veto` (Vetoer), `s3_recrop` (Recropper), `s3_decode_crops` / `decode_manifest` (CropDecoder x8-16 L4), `d1p_fetch`, `d1p_filter(max_cer=0.15)`, `d2_harvest(langs)`, `decode_mix` | | scan_image CPU or L4 | Data pipeline stages |
| `build_beds(which)`, `build_beds3(which)` | | L4 cpu 4-8 | Eval beds |
| `build_mix`, `build_mix3(seed,d1_h=20,d10_h=6)`, `patch_mix3`, `build_mix3b(per_video=12)`, `build_mix3c(mp,mp_repeat)`, `share_v3`, `audit_mix3(fix=True,tol=0.1)`, `clean_manifest(src,dst,tol)` | | mix_image/nemo_image CPU 8-16, 32 GB, up to 8 h | Mix building and audits |
| `preflight`, `rescore_dryrun`, `score_texts`, `seam_decode`, `year_beam`, `debug_ckpt` | | L4 | Diagnostics/research |
| `validate_labeller(items,out)` (local entrypoint) | | | Labeller validation |
`ml_p3.py`: `scan_lang(lang,hours=4.5,max_files=10)` cpu 2/8 GB max 6 containers; `label_lang(lang,limit,new_hours=1.5)` L4; local entrypoints `scan`, `label`. `ml_modal.py`: `p2_lang` L4 x6, local entrypoint `p2 --gold <dir> --out p2_out.json --n-real 300`.

**Volume layout** (`/vol/...`): `ultra.nemo` (symlink to HF cache), `hf/`, `export/ultra/state.pt`, `export/<run>[@step]/decoder_joint.safetensors`, `exp/<run>/{step<N>.pt, decoder_joint.pt, last.ckpt, train_manifest.jsonl}`, `e22/`, `e21local/audio/`, `replay/{lspc_*.jsonl, fleurs/, fleurs_train_multi.jsonl, icsi/, icsi_train.jsonl}`, `librispeech/`, `slr28/`, `tts/<name>/{wav,train.jsonl,dev.jsonl,train_ok.jsonl,dev_ok.jsonl}`, `s0/{scan,scan_stats.json}`, `yodas/{en,en_plain}/{wav,manifests}`, `d1/{rows,rows3,crops,crops2,crops3,crops3_ok,dec3,veto,veto3,plain3.jsonl,plain3_ok.jsonl}`, `d2/<lang>/{digit,plain}.jsonl`, `mix/`, `mix3/`, `mix3b/`, `beds/{vox,fleurs,fleurs3,noise,adv,fast,e21x,e21_dev}`, `enc/` (cached encoder outputs), `hyps/<run>[@step][tag]/<bed>.jsonl`, `ml/p3/<lang>/`, `rescore/ yeardrop/ yearbeam/ seamb1/ seamsrc/`.

---------------------------------------------------------------------------------------------------

## 4. Artifacts: where they live

| Artifact | Location | In git? |
|---|---|---|
| Stock Ultra | HF `ncannings/parakeet-ultra-nemo@68d18c58...` -> `/vol/ultra.nemo` | no (pinned by hash in modal_app.py) |
| v10pt@1500 (the warm-start root) | `/vol/exp/v10pt/step1500.pt`; HF bundle `MimicScribe/parakeet-ultra-spoken-coreml@ccdb31c9`; local `$MODELS/parakeet-ultra-spoken-v10pt-1500` | NO; irreplaceable unless re-trained |
| Candidate A `sbd0soup_a1.0@0` | `/vol/exp/sbd0soup_a1.0/step0.pt`, `/vol/exp/r3_SBd0/step*.pt`; bundle `$MODELS/parakeet-ultra-sbd0soup` | no |
| **Model B `w3soup_a0.5@0`** | `/vol/exp/w3soup_a0.5/step0.pt`; `/vol/export/w3soup_a0.5@0/`; bundle `$MODELS/parakeet-ultra-best3`; a local backup copy (decoder_joint.safetensors, SHA256.txt, report files); HF `MimicScribe/parakeet-ultra-spoken-coreml@5edfc817ce69dcdf4ca7839bff5ad738e9d390b8` | no |
| Other bundles | `$MODELS/parakeet-ultra-{w3k,w2mpv,stock-95eaa59a}`, `...-spoken-{v9pr30,v10pts2,rt2_main_s2-3000}-1500/3000` | no |
| Mixes | `/vol/mix3/*` (d1, d1p, d10n, d10p, d2, d2p, fleurs_train_multi, d1_dev + `.pre_patch`/`.pre_audit`/`.pre_unk` originals), `/vol/mix3b/*` (d1, d10n, d1p_v, d10p_v), `/vol/mix/*` (retrain-2) | no (rebuildable from stages 2.4-2.6 if the volume's intermediates survive) |
| TTS sets | `/vol/tts/{v10all,v10pt,d9sup,biz,mp,mp2,...}` | no (rebuildable from carriers/readings/gen/voice; exact build args partly undocumented) |
| Manifests of record | `/vol/exp/<run>/train_manifest.jsonl` (v10pt: 21,115 rows) | no |
| Eval beds + cached encoder outputs | `/vol/beds/*`, `/vol/enc/*` | no |
| Hyps | `/vol/hyps/<run>/...` | no (pull with `modal volume get`) |
| Gold sets | `pipeline/gold/owner_year_readings.json` (62 clips, text + readings) | YES; the audio is Earnings-21 (local, not redistributed)|
| Probe set | `eval/probe_items.tsv` | YES (unused by code) |
| Candidate table | `pipeline/ml_grammar_table.json` | YES |
| Run logs | `p3_*.log` at repo root | untracked, not ignored |
| Grammar dumper (regenerates the candidate table) | not in this repo | private |
| `.gitignore` | `__pycache__/ *.wav *.nemo *.ckpt .env .venv/ *.pyc ` | |
| Scratch `$S` (`$S`) | the launch scripts (now preserved in `runs/`), `scores_all.json`, `sealed/`, report generator | scratch gone; scripts in `runs/` |

Things a restart needs that are NOT in git: the Modal volume contents above (esp. `exp/v10pt/step1500.pt`, `exp/sbd0soup_a1.0/step0.pt`), the `hf` Modal secret, ICSI corpus + `meetings.json`, Earnings-21 audio for `e21local`, `beds/adv` + `beds/fast` + `tts/gold62`, the grammar dumper (not in this repo), HF write access to `MimicScribe/parakeet-ultra-spoken-coreml`.

---------------------------------------------------------------------------------------------------

## 5. Gaps and stale statements

Stale / wrong:
1. `README.md` says "Status: in progress. Nothing is trained yet. This repo will hold the data generator, training scripts, evaluation, and the released weights." False: model B is trained, shipped to HF (5edfc817) and pinned in the consuming app. README also says "decoder and joint weights are written into the existing CoreML bundle" (true) but gives no commands, no HF repo name, no weights; the "Compute: a few GPU-hours" and "Evaluation 1-5" list is aspirational, not what scripts do (the repo has no LibriSpeech/AMI/five-FLEURS-language eval script; its scorers are `score_retrain.py` over beds listed in `ft_eval.RETRAIN3_SETS`). Item 5 "CoreML matches NeMo token for token" has no script here.
2. README "Scope" says the model "should never emit" `0-9 $ % € £ /` - true only for English; OBSERVATIONS shows non-English deliberately keeps digits (about 75% exact digits). README "First release: English. The other 24 languages follow" conflicts with B already being multilingual-neutral by KL-anchoring (non-English held at digits).
3. README License: "Model weights, when released" - already released (CC BY 4.0).
4. `modal_app.py` docstring says run `modal run pipeline/modal_app.py::env_check` from repo root; `ml_p3.py`/`.sh` assume `cd pipeline`. Inconsistent cwd conventions; `add_local_python_source` needs the pipeline dir importable, so run from `pipeline/` to be safe.
5. `retrain3()` default `dur_weight=1.0` and default `mix="mix3"`/`init_from="v10pt@1500"` do not match the B recipe (needs `--dur-weight 0`; W3 needs `--mix v3 --init-from sbd0soup_a1.0@0 --anchor-init --lr 3e-05 --warmup 20 --max-steps 1000`). Also `soup3` default `base=v10pt@1500` is wrong for the W3 soup (needs `sbd0soup_a1.0@0`). The comments above `R3_ARMS` mention "c=1 maintainer decision" but the actual shipped run dropped the duration KL, which the comment does not say.
6. `build_beds()` docstring-less; the `retrain_mix`/`RETRAIN_K` (retrain-2) path is superseded and its tag tables disagree with retrain3's.
7. `OBSERVATIONS.md` refers to `data_icsi.py --overlap` and `restore_unk_nonen` as "committed, not yet trained" - accurate, but nothing in the repo trains with the Greek fix yet.

Missing / not reproducible from repo alone:
1. **v10pt@1500 recipe**: the exact `run` flags (kl_weight, kl_terms, kl_exempt, kl_punct, tts_h, lspc_h, icsi_h, batch_duration) for v9pr30/v10a/v10pt are recovered in `runs/v12/train.sh` (v12 = the v10pt recipe + fast-year TTS); the TTS supplement args are only partly recorded. Without the checkpoint a from-scratch restart cannot reproduce the warm-start root.
2. **make_tts invocations** for `v10all`, `v10pt` (289 sub-one rows), `d9sup`, `biz`, `mp`, `mp2`, `v12year*` (rows, seeds, slots, speeds): only partially quoted in plans.
3. **Builders for** `beds/adv/all.jsonl` (adversarial bed), `beds/fast/all.jsonl` (fast-speech bed), `tts/gold62` (listener-verified gold manifest), `e22/other.jsonl` is built but `e21local/audio` upload step is manual, `d1/rows` initial relabel invocation.
4. **ICSI**: corpus is licensed/manual; `meetings.json` list not in repo; the 25 held-out meetings used as gates are not recorded here.
5. **No dependency lock for local scripts**, no Makefile/driver script chaining the stages, no CI. `pytest` is not documented as a dependency.
6. `restore_fillers.py` is hard-wired to `<S>/rt/mix/<part>.jsonl` (scratch layout) and the `filter3.py` expects four pulled-down directories whose provenance is only in docstrings.
9. Regenerating `ml_grammar_table.json` needs the private grammar dumper; outsiders use the table as shipped.
10. Threshold values for the ship gates (ship bar: non-EN WER <= stock + 0.15 pp, number share 35-40%) are asserted in code (`retrain3` share check) or plan prose only.
11. `p3_*.log`, `.serena/` are untracked clutter; `eval/probe_items.tsv` is unreferenced.

---------------------------------------------------------------------------------------------------

## 6. Training-relevant findings (2026-10-11 number-fidelity investigation) and what followed

From `number-fidelity-plan-2026-10-11.md` (the plan that triggered retrain-3):
* **Training-label audit of v10pt** (manifest of 21,115 rows: TTS 9,752, LibriSpeech-PC 7,045, ICSI 3,327, FLEURS 991): real-speech rows hold ZERO 2001-2039 years; TTS years about 330 rows (1.5%); 2010-2039 labelled "twenty X" x215 vs "two thousand X" x24 (10%). So the problem is missing real-speech year exposure, not mislabelling. Relabelling is NOT the fix; ADD year data. E21/E22 are eval-only (never train); SPGISpeech is non-commercial.
* **Year failures are a near-tie fork**: at the first year piece " two" vs " twe" sits at about 0.50 confidence; greedy commits and the rest of the year goes blank (dropped years: stock 6, v10pt 10-11 of 310) or two windows' readings are spliced ("twenty thousand twenty"). Refuted fixes: duration cap, beam search, blank-penalty oracle (0.5/1/2 recovered nothing), same-model rescoring (fixed 0-1/67, broke 24-63), decode-time top-2 probe (1/3 sites; STOP). Year FORM ("two thousand X") is mostly not an error (blind judge heard it at 16/18 sites).
* Teen/ty acoustic misses (7 of 18 model-both bucket) are encoder-limited; teen/ty TTS data was judged overfit-risk and skipped.
* Rules the plan sets for any training run: cost estimate and maintainer OK first (about $9-15 GPU + full re-eval), mined E21/AMI sites are EVALUATION data only, report year-form counts from the training manifest, package the plan's beds as the model's eval.
What the retrain actually found/did (see OBSERVATIONS.md), for a restarter:
* Duration KL (`dur_weight` > 0) destroys year values: keep 0. KL teacher is STOCK (not the init model): `add_r3_losses` copies the model before `init_from` loads.
* Minimal-pair year TTS (same voice, both readings, `gen.minimal_pairs`, pairs x2, 15% "and") fixes the left-context prior; α=0.5 blend toward the prior winner keeps the readings gain without the full recipe's ~3 year-value loss in dates (W2mpv app bar failed by a hair: exact 9606->9602).
* Number-row share must be 35-40% of hours (B: 39.2%); do not downsample the mix to save compute (steps stream batches); no AMI in any mix (eval overlap).
* Open model-side weaknesses worth a future run (OBSERVATIONS section 2): short-phrase drops (23/38 of B's largest drops were words really said; not a calibration offset; fixes: anchor non-number tokens on span/anchor_num rows, gate P(non-blank)>0.5 on human-verified rows, ICSI `by_turns` hand-offs never `dominant`, L2-SP ~0.3), Greek final sigma (fixed in `restore_unk_nonen`, untrained), mixed digit+word outputs ("десет 000"), decimals/currency in small languages (stock prior), "GAAP"->"gap". Gates for any new arm: 44-call E21 bar, years (blind set + held-out pairs), content drops+insertions on E21-44+AMI-16, app FLEURS all 25 languages, synthetic number banks, live invariants, one sealed-holdout look. A new run is justified only by a measured model-side problem.
