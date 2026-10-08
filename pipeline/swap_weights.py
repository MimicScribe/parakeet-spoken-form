"""Write fine-tuned decoder + joint weights into a copy of the Parakeet Ultra CoreML bundle.

The encoder is frozen, so every tensor keeps its shape and the CoreML programs are reused as
they are: only the fp16 data regions of the two `weight.bin` files change.

    python swap_weights.py <ultra_coreml_dir> <decoder_joint.safetensors> <out_dir>

The safetensors file holds NeMo-named tensors (`decoder.prediction.*`, `joint.*`), written by
`modal_app.py::export_weights`. Running it on Ultra's own tensors must reproduce the bundle
byte for byte (`--check`).

CoreML weight.bin layout (coremltools MILBlob StorageFormat): a 64-byte header, then per blob a
64-byte metadata record (sentinel 0xDEADBEEF, dtype, size, data offset) followed by the data.
PyTorch LSTM gates are ordered [i, f, g, o]; CoreML's are [i, f, o, g], with one bias equal to
b_ih + b_hh.
"""

import filecmp
import shutil
import struct
import subprocess
import sys

import numpy as np
from safetensors.numpy import load_file

H = 640
LSTM = "decoder.prediction.dec_rnn.lstm"
DECODER = [  # (tensor, metadata offset, shape, transform)
    ("decoder.prediction.embed.weight", 64, (8193, 640), "plain"),
    (f"{LSTM}.weight_ih_l0", 10487168, (2560, 640), "gates"),
    (f"{LSTM}.weight_hh_l0", 13764032, (2560, 640), "gates"),
    (f"{LSTM}.bias_l0", 17040896, (2560,), "bias"),
    (f"{LSTM}.weight_ih_l1", 17046080, (2560, 640), "gates"),
    (f"{LSTM}.weight_hh_l1", 20322944, (2560, 640), "gates"),
    (f"{LSTM}.bias_l1", 23599808, (2560,), "bias"),
]
JOINT = [
    ("joint.enc.weight", 64, (640, 1024), "plain"),
    ("joint.enc.bias", 1310848, (640,), "plain"),
    ("joint.pred.weight", 1312192, (640, 640), "plain"),
    ("joint.pred.bias", 2131456, (640,), "plain"),
    ("joint.joint_net.2.weight", 2132800, (8198, 640), "plain"),
    ("joint.joint_net.2.bias", 12626304, (8198,), "plain"),
]
WEIGHT_FILES = ["Decoder.mlmodelc/weights/weight.bin", "JointDecisionv3.mlmodelc/weights/weight.bin",
                "JointDecision.mlmodelc/weights/weight.bin"]


def reorder(t: np.ndarray) -> np.ndarray:
    return np.concatenate([t[0:H], t[H:2 * H], t[3 * H:4 * H], t[2 * H:3 * H]])


def tensor(sd: dict, name: str, kind: str) -> np.ndarray:
    if kind == "bias":
        layer = name[-2:]
        return reorder(sd[f"{LSTM}.bias_ih_{layer}"].astype(np.float32)
                       + sd[f"{LSTM}.bias_hh_{layer}"].astype(np.float32))
    t = sd[name].astype(np.float32)
    return reorder(t) if kind == "gates" else t


def patch(path: str, entries: list, sd: dict) -> None:
    with open(path, "r+b") as f:
        for name, meta, shape, kind in entries:
            f.seek(meta)
            sentinel, dtype, size, data = struct.unpack("<IIQQ", f.read(24))
            assert sentinel == 0xDEADBEEF and dtype == 1 and data == meta + 64, (name, hex(sentinel), dtype)
            t = tensor(sd, name, kind)
            assert t.shape == shape and size == t.size * 2, (name, t.shape, shape, size)
            assert np.isfinite(t).all() and np.abs(t).max() < 65504, f"{name}: not representable in fp16"
            f.seek(data)
            f.write(np.ascontiguousarray(t, dtype=np.float16).tobytes())


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    src, weights, out = args
    if sys.platform == "darwin":  # APFS clone: no extra disk until the weights are rewritten
        subprocess.run(["cp", "-cR", src, out], check=True)
    else:
        shutil.copytree(src, out)
    sd = load_file(weights)
    patch(f"{out}/{WEIGHT_FILES[0]}", DECODER, sd)
    patch(f"{out}/{WEIGHT_FILES[1]}", JOINT, sd)
    # The app's v2-era joint program reads the same six blobs at the same offsets.
    shutil.copyfile(f"{out}/{WEIGHT_FILES[1]}", f"{out}/{WEIGHT_FILES[2]}")
    if "--check" in sys.argv:
        same = [filecmp.cmp(f"{src}/{w}", f"{out}/{w}", shallow=False) for w in WEIGHT_FILES]
        print("byte-identical to source:", dict(zip(WEIGHT_FILES, same)))
        sys.exit(0 if all(same) else 1)
    print("wrote", out)


if __name__ == "__main__":
    main()
