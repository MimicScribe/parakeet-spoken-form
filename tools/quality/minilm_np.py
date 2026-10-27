"""all-MiniLM-L6-v2 sentence embeddings in pure numpy (no torch / sentence-transformers). CPU only; set VECLIB_MAXIMUM_THREADS=2 to keep it light.
Weights: a local snapshot of sentence-transformers/all-MiniLM-L6-v2 (Apache-2.0) holding vocab.txt and model.safetensors. The directory comes from
the environment variable MINILM_DIR, else from the Hugging Face cache (HF_HOME or ~/.cache/huggingface). Nothing is downloaded by this module.
  from minilm_np import Embedder; E = Embedder(); X = E.encode(list_of_str)   # L2-normalised (n, 384)"""
import glob, json, os, re, struct, unicodedata
import numpy as np


def snapshot_dir():
    d = os.environ.get("MINILM_DIR")
    if d:
        return os.path.expanduser(d)
    hub = os.path.join(os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface")), "hub")
    snaps = sorted(glob.glob(os.path.join(hub, "models--sentence-transformers--all-MiniLM-L6-v2", "snapshots", "*")))
    if not snaps:
        raise FileNotFoundError("MiniLM weights not found: set MINILM_DIR to a snapshot of sentence-transformers/all-MiniLM-L6-v2 (vocab.txt, model.safetensors)")
    return snaps[-1]


def load_safetensors(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); base = 8 + n
        out = {}
        for k, v in h.items():
            if k == "__metadata__": continue
            s, e = v["data_offsets"]; f.seek(base + s); buf = f.read(e - s)
            if v["dtype"] not in ("F32", "F16"): continue
            out[k] = np.frombuffer(buf, dtype={"F32": np.float32, "F16": np.float16}[v["dtype"]]).reshape(v["shape"]).astype(np.float32)
    return out

class Tok:
    def __init__(self, snap):
        with open(os.path.join(snap, "vocab.txt"), encoding="utf8") as f:
            self.vocab = {l.rstrip("\n"): i for i, l in enumerate(f)}
    def basic(self, t):
        t = unicodedata.normalize("NFD", t.lower()); t = "".join(c for c in t if unicodedata.category(c) != "Mn")
        return re.findall(r"[a-z0-9]+|[^\sa-z0-9]", t)
    def wp(self, w):
        if w in self.vocab: return [self.vocab[w]]
        out, i = [], 0
        while i < len(w):
            j = len(w); cur = None
            while i < j:
                sub = ("##" if i else "") + w[i:j]
                if sub in self.vocab: cur = self.vocab[sub]; break
                j -= 1
            if cur is None: return [self.vocab["[UNK]"]]
            out.append(cur); i = j
        return out
    def encode(self, t, maxlen=64):
        ids = [self.vocab["[CLS]"]]
        for w in self.basic(t):
            ids += self.wp(w)
            if len(ids) >= maxlen - 1: break
        return ids[:maxlen - 1] + [self.vocab["[SEP]"]]

def gelu(x): return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))
def _erf(x):  # Abramowitz-Stegun 7.1.26 (max err 1.5e-7)
    s = np.sign(x); x = np.abs(x); t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y
def ln(x, g, b, eps=1e-12):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True); return (x - m) / np.sqrt(v + eps) * g + b

class Embedder:
    def __init__(self, snap=None):
        snap = snap or snapshot_dir()
        W = load_safetensors(os.path.join(snap, "model.safetensors")); self.W = {k.replace("0_Transformer.", "").replace("auto_model.", ""): v for k, v in W.items()}
        self.tok = Tok(snap); self.L = 6; self.H = 12
        if "embeddings.word_embeddings.weight" not in self.W: raise KeyError(list(self.W)[:5])
    def encode(self, texts, bs=64):
        out = []
        for i in range(0, len(texts), bs):
            ids = [self.tok.encode(t) for t in texts[i:i + bs]]; n = max(map(len, ids))
            X = np.zeros((len(ids), n), dtype=np.int64); M = np.zeros((len(ids), n), dtype=np.float32)
            for r, s in enumerate(ids): X[r, :len(s)] = s; M[r, :len(s)] = 1
            out.append(self._fwd(X, M))
        return np.concatenate(out)
    def _fwd(self, X, M):
        W = self.W; B, T = X.shape
        h = W["embeddings.word_embeddings.weight"][X] + W["embeddings.position_embeddings.weight"][:T][None] + W["embeddings.token_type_embeddings.weight"][0][None, None]
        h = ln(h, W["embeddings.LayerNorm.weight"], W["embeddings.LayerNorm.bias"])
        bias = (1.0 - M)[:, None, None, :] * -1e9
        for l in range(self.L):
            p = f"encoder.layer.{l}."
            def lin(x, n): return x @ W[p + n + ".weight"].T + W[p + n + ".bias"]
            q, k, v = (lin(h, f"attention.self.{n}").reshape(B, T, self.H, -1).transpose(0, 2, 1, 3) for n in ("query", "key", "value"))
            a = q @ k.transpose(0, 1, 3, 2) / np.sqrt(q.shape[-1]) + bias; a = np.exp(a - a.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
            c = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, -1)
            h = ln(h + lin(c, "attention.output.dense"), W[p + "attention.output.LayerNorm.weight"], W[p + "attention.output.LayerNorm.bias"])
            h = ln(h + lin(gelu(lin(h, "intermediate.dense")), "output.dense"), W[p + "output.LayerNorm.weight"], W[p + "output.LayerNorm.bias"])
        e = (h * M[:, :, None]).sum(1) / M.sum(1, keepdims=True); return e / np.linalg.norm(e, axis=1, keepdims=True)
