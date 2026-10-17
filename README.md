# parakeet-spoken-form

A fine-tune of [Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra) that writes numbers
and symbols as the words the speaker said. "I'll be there at four fifteen" stays "four fifteen", not
`4:15` or `415`. "One two seven point zero point zero point one" stays as spoken, not `127.0.0.1`.

Status (2026-10-17): model B ("retrain-3") is trained and published as CoreML weights at
[MimicScribe/parakeet-ultra-spoken-coreml](https://huggingface.co/MimicScribe/parakeet-ultra-spoken-coreml)
(revision `5edfc817`), queued for the next MimicScribe release. The next round (all 25 languages, long
numbers across window seams) is in the data phase: labelling, census and held-out tooling are done, and no
training run starts before the census passes its bar. **Start at [RESTART.md](RESTART.md)** for the state,
then [docs/RECIPE.md](docs/RECIPE.md) (sources to a gated run) and [docs/CONCEPTS.md](docs/CONCEPTS.md)
(the ideas worth borrowing), with the numbers in [docs/RESULTS.md](docs/RESULTS.md). Census, held-out, judge and seam scorers: [eval/quality/](eval/quality/).

## Why

Parakeet (NVIDIA's parakeet-tdt-0.6b-v3 and its derivatives) decides per decode whether to write a
number as digits or as words. A streaming transcriber decodes overlapping windows and stitches them
together. When two windows render the same number differently, the stitch has nothing to align on,
which produces duplicated or dropped text.

Measured in [MimicScribe](https://mimicscribe.app)'s streaming pipeline with Parakeet Ultra, 16
files, about 14.5 hours of audio (Earnings-21, podcasts, SCOTUS, AMI):

- 193 window seams carried the same number written two different ways.
- 7 numbers came out garbled across the 11 Earnings-21 calls, for example `2028 2028` and
  `EU9 forty nine million`.

Converting digits back to words after decoding removes most seam mismatches. But it guesses the
reading: `1,300` becomes "one thousand three hundred" even when the speaker said "thirteen
hundred". A model that only writes spoken form gives every window the same output and keeps the
words that were said.

The mixing likely comes from the training labels. In a sample of the English YouTube-Commons
(YTC) manifest of [Granary](https://huggingface.co/datasets/nvidia/Granary) (the first 2 GB,
3.1M segments), 11.0% of labels contain digits only and 14.8% contain number words only.

## Scope

The model's tokenizer has 15 non-letter pieces besides punctuation: `0`–`9`, `$`, `%`, `€`, `£`,
and `/`. In English the fine-tuned model should never emit them; the other 24 languages keep stock
behaviour (digits). Number punctuation (`:` in times, `.` in
decimals, `,` in thousands) goes with them.

First release: English. The other 24 languages Parakeet supports follow, starting with Spanish,
French, Italian and Portuguese.

## Approach

- **Base:** Parakeet Ultra, the NeMo checkpoint.
- **What trains:** the prediction network and joint only, about 18M of 627M parameters. The
  encoder is frozen.
- **Data:**
  - Synthetic speech of number-dense sentences with several spoken readings per value, voiced
    with [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M).
  - Real speech with no digits in its labels, so the rest of the model's behaviour holds:
    LibriSpeech-PC, FLEURS.
- **Compute:** a single GPU on Modal. The full run is expected to take a few GPU-hours.
- **Export:** the decoder and joint weights are written into the existing CoreML bundle layout, so
  the int8 encoder is reused unchanged.

## Evaluation

Each step is compared against unmodified Parakeet Ultra run through the same scripts:

1. Held-out synthetic sentences: number spans exact, no digit tokens.
2. Real speech with numbers (Earnings-22): values preserved, nothing dropped or invented.
3. General accuracy: LibriSpeech, AMI, VoxPopuli, and five FLEURS languages.
4. Error shapes: homophones ("to"/"two"), deletions after numbers, tokens per frame.
5. Export check: the CoreML bundle matches NeMo token for token.

`eval/probe_items.tsv` is the spoken-form probe set: 59 number formats and 7 controls.

## Credits

This work builds on other people's open models, data and code:

- **Base model:** [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) by NVIDIA,
  and [Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra) by Moondream, the checkpoint this
  fine-tune starts from.
- **CoreML port and runtime:** [FluidInference](https://huggingface.co/FluidInference)'s CoreML
  conversions ([parakeet-ultra-coreml](https://huggingface.co/FluidInference/parakeet-ultra-coreml),
  [parakeet-tdt-0.6b-v3-coreml](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml)), whose
  bundle layout the export reuses, and the [FluidAudio](https://github.com/FluidInference/FluidAudio) SDK.
- **Training framework:** [NVIDIA NeMo](https://github.com/NVIDIA/NeMo), [PyTorch](https://pytorch.org),
  [Lightning](https://github.com/Lightning-AI/pytorch-lightning).
- **Synthetic speech:** [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) and
  [Piper](https://github.com/rhasspy/piper) with [piper-voices](https://huggingface.co/rhasspy/piper-voices).
- **Label checks (CTC alignment):** [wav2vec 2.0 XLSR-53 fine-tunes](https://huggingface.co/jonatasgrosman)
  by Jonatas Grosman (de, es, fr, it, nl, pt),
  [wav2vec2-large-960h-lv60-self](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self) and
  [hubert-large-ls960-ft](https://huggingface.co/facebook/hubert-large-ls960-ft) by Meta.
- **Augmentation:** [RIRS_NOISES](https://www.openslr.org/28/) (OpenSLR 28) room responses and noise.
- **Training data:** [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli),
  [YODAS-Granary](https://huggingface.co/datasets/espnet/yodas-granary) and
  [Granary](https://huggingface.co/datasets/nvidia/Granary), [FLEURS](https://huggingface.co/datasets/google/fleurs),
  [LibriSpeech-PC](https://www.openslr.org/145/), [ICSI meetings](https://groups.inf.ed.ac.uk/ami/icsi/).
- **Evaluation data:** [Earnings-21 and Earnings-22](https://github.com/revdotcom/speech-datasets) by Rev,
  [AMI](https://groups.inf.ed.ac.uk/ami/corpus/), [LibriSpeech](https://www.openslr.org/12/).
- **Audio and text tools:** [librosa](https://github.com/librosa/librosa),
  [soundfile](https://github.com/bastibe/python-soundfile),
  [Hugging Face Transformers](https://github.com/huggingface/transformers),
  [langid.py](https://github.com/saffsd/langid.py).

## License

Training data licenses: VoxPopuli (CC0), YODAS-Granary (CC BY 3.0; machine transcripts, used only where a
reading check confirms them), FLEURS (CC BY 4.0), LibriSpeech-PC (CC BY 4.0), ICSI meetings (CC BY 4.0).

Code: Apache-2.0. Model weights: CC BY 4.0, derived from
[parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (NVIDIA, CC BY 4.0) and
[Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra) (Moondream, CC BY 4.0).

Observations, known weaknesses and ideas for the next run: [OBSERVATIONS.md](OBSERVATIONS.md).
