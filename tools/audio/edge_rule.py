"""edge_rule: where to cut an audio clip so it ends in a quiet valley, the way a windowed streaming decoder places its window boundaries
(numpy only; audio / VAD loaders import soundfile / torch lazily).

The rule is a reference for what a windowed decoder does when it moves a nominal window boundary to a quieter spot nearby, written so a training clip can
end where a decoded window would. Every size is a parameter of Track(frame=..., vad_chunk=..., sr=...); the defaults below are one common configuration.
  frame      80 ms = 1280 samples at 16 kHz (the candidate step: one encoder frame); energy per frame = mean |x| (NOT squared)
  vad_chunk  4096 samples (256 ms): one speech probability per chunk (this implementation takes the MAX of the 512-sample Silero probabilities in the chunk)
  valley     a run of candidate frames whose chunk probability is < VALLEY_THR (0.15); relaxed when the minimum probability in the search lies in
             (VALLEY_THR, RELAX_MAX = 0.35): threshold = min(RELAX_CAP = 0.40, min + RELAX_ADD = 0.08)
  pick       the WIDEST valley (tie: lower mean probability), then its LOWEST-energy frame; no valley -> the lowest-energy frame over the whole search span
  search     +- SEARCH_S (2.0 s) around the nominal position; the caller passes the window of the search to snap(track, lo_s, hi_s)
  dip rule   (for clips with no word gap in range) cut at the quietest VAD + RMS point ONLY when it is a real dip:
             dip_db = local speech level - level at the cut >= DIP_DB, else exclude the clip
Unverified for any particular decoder: how it turns a vad_chunk of samples into one probability."""
import numpy as np

SR = 16000
FRAME = 1280            # 80 ms
VAD_CHUNK = 4096        # 256 ms
VALLEY_THR = 0.15
RELAX_MAX = 0.35
RELAX_CAP = 0.40
RELAX_ADD = 0.08
SEARCH_S = 2.0
DIP_DB = 20.0           # a proposed default: on a study of 150 real clips (900 edges) it kept 77 % of in-gap picks and blocked 62 % of inside-word picks (24 dB: 65 % / 79 %). Confirm by listening on your audio.
SPEECH_CTX_S = 3.0      # local speech level is measured within +-3 s of the cut


def load_audio(path, sr_out=SR):
    """mono float32 at sr_out (soundfile; scipy for resampling)"""
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != sr_out:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, sr_out)
        x = resample_poly(x, sr_out // g, sr // g).astype(np.float32)
    return x


_VAD = None


def vad_chunk_probs(x, vad_chunk=VAD_CHUNK, sr=SR):
    """one speech probability per vad_chunk samples (max over its 512-sample Silero probabilities). Needs torch + the silero-vad package
    (pip install silero-vad; or put its directory on SILERO_PYLIB)."""
    global _VAD
    import os, sys, torch
    if _VAD is None:
        extra = os.environ.get("SILERO_PYLIB")
        if extra and extra not in sys.path:
            sys.path.insert(0, extra)
        from silero_vad import load_silero_vad
        _VAD = load_silero_vad()
    m = _VAD
    m.reset_states()
    n = len(x) // 512
    probs = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(n):
            probs[i] = float(m(torch.from_numpy(x[i * 512:(i + 1) * 512]), sr).item())
    per = vad_chunk // 512
    nch = (len(x) + vad_chunk - 1) // vad_chunk
    out = np.ones(nch, dtype=np.float32)
    for c in range(nch):
        seg = probs[c * per:(c + 1) * per]
        out[c] = float(seg.max()) if len(seg) else 1.0
    return out


class Track:
    """audio-derived data for one clip: frame energies (mean |x| per `frame` samples), VAD chunk probabilities (one per `vad_chunk` samples)."""

    def __init__(self, x, vad=None, frame=FRAME, vad_chunk=VAD_CHUNK, sr=SR):
        self.n = len(x)
        self.frame, self.vad_chunk, self.sr = frame, vad_chunk, sr
        nf = len(x) // frame
        self.energy = np.abs(x[:nf * frame]).reshape(nf, frame).mean(1) if nf else np.zeros(0, dtype=np.float32)
        self.vad = np.asarray(vad if vad is not None else [], dtype=np.float32)

    def db(self):
        return 20 * np.log10(np.maximum(self.energy, 1e-6))

    def prob_at(self, frame_idx):
        k = (frame_idx * self.frame) // self.vad_chunk
        return float(self.vad[k]) if 0 <= k < len(self.vad) else 1.0


def snap(track, lo_s, hi_s):
    """the selection rule restricted to candidate frames whose START lies in [lo_s, hi_s] (seconds). Returns dict(frame, t, t_center, used_valley, valley_frames, energy_db) or None."""
    fr, sr = track.frame, track.sr
    lo = max(0, int(np.ceil(lo_s * sr / fr))); hi = min(len(track.energy) - 1, int(np.floor(hi_s * sr / fr)))
    if hi < lo:
        return None
    idx = list(range(lo, hi + 1))
    probs = [track.prob_at(i) for i in idx]
    en = [float(track.energy[i]) for i in idx]
    best = None; used = False; vw = 0
    if len(track.vad):
        mn = min(probs)
        thr = min(RELAX_CAP, mn + RELAX_ADD) if (mn > VALLEY_THR and mn < RELAX_MAX) else VALLEY_THR
        valleys = []; s = None
        for k, p in enumerate(probs):
            if p < thr:
                s = k if s is None else s
            elif s is not None:
                valleys.append((s, k - 1)); s = None
        if s is not None:
            valleys.append((s, len(probs) - 1))
        if valleys:
            def key(v):
                a, b = v
                return (b - a + 1, -float(np.mean(probs[a:b + 1])))
            a, b = max(valleys, key=key)
            kbest = min(range(a, b + 1), key=lambda k: en[k])
            best = idx[kbest]; used = True; vw = b - a + 1
    if best is None:
        best = idx[min(range(len(idx)), key=lambda k: en[k])]
    return dict(frame=best, t=best * fr / sr, t_center=(best + 0.5) * fr / sr, used_valley=used, valley_frames=vw, energy_db=float(20 * np.log10(max(track.energy[best], 1e-6))))


def speech_level_db(track, t, ctx=SPEECH_CTX_S):
    """local speech level at time t: 75th percentile of frame dB over the frames within +-ctx whose VAD probability is >= 0.5 (all frames when none are)"""
    db = track.db(); a = max(0, int((t - ctx) * track.sr / track.frame)); b = min(len(db), int((t + ctx) * track.sr / track.frame) + 1)
    if b <= a:
        return None
    fr = list(range(a, b))
    sp = [i for i in fr if track.prob_at(i) >= 0.5] if len(track.vad) else []
    sel = sp if len(sp) >= 3 else fr
    return float(np.percentile(db[sel], 75))


def dip_db(track, t_cut, lvl_db=None):
    """how far below the local speech level the frame at the cut is (positive = quieter)"""
    k = int(t_cut * track.sr / track.frame)
    if not (0 <= k < len(track.energy)):
        return None
    lv = lvl_db if lvl_db is not None else speech_level_db(track, t_cut)
    if lv is None:
        return None
    return lv - float(20 * np.log10(max(track.energy[k], 1e-6)))


def snap_with_dip(track, lo_s, hi_s, min_dip=DIP_DB):
    """the pick + the minimum-dip rule: returns (pick|None, dip, ok); ok False -> exclude the clip ('no_dip')"""
    p = snap(track, lo_s, hi_s)
    if p is None:
        return None, None, False
    d = dip_db(track, p["t_center"])
    return p, d, (d is not None and d >= min_dip)
