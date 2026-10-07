# parakeet-spoken-form

A fine-tune of [Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra) that writes numbers
and symbols as the words the speaker said. "I'll be there at four fifteen" stays "four fifteen", not
`4:15` or `415`. "One two seven point zero point zero point one" stays as spoken, not `127.0.0.1`.

Status: in progress. Nothing is trained yet. This repo will hold the data generator, training
scripts, evaluation, and the released weights.

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
and `/`. The fine-tuned model should never emit them. Number punctuation (`:` in times, `.` in
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

## License

Code: Apache-2.0. Model weights, when released: CC BY 4.0, derived from
[parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (NVIDIA, CC BY 4.0) and
[Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra) (Moondream, CC BY 4.0).
