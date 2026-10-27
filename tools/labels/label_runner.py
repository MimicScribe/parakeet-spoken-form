"""Pipeline + GPU/CPU sampler for the never-reject labeller. Stdlib only at import time (torch / soundfile / pynvml are imported inside the functions that need them),
so the ordering / error / sampler logic is testable offline (test_label_runner.py); label_local.py is the command-line driver around run_part.

Why: a plain loop does load -> forward -> score -> write one row at a time on ONE thread, so the GPU sits idle while the main thread waits on the file read and on the CPU scoring. Here
  * LOAD   threads (default 8): file read + mono + resample + feature normalisation, `window` rows ahead of the GPU (I/O bound, GIL released in libsndfile / numpy)
  * FORWARD main thread, strictly in row order: the only GPU stage, so the card is fed back to back
  * SCORE  threads (default = cpu): CTC losses (on the device of the log-probs), greedy text, forced alignment, record
  * WRITE  main thread, in INPUT order (the output file is row-for-row the input), flushed every few rows, `commit` (e.g. a remote-volume flush) called on a time / row budget
Per-row records come from labeller_core.score_row; `t_s` is this row's score time."""
import collections, concurrent.futures as cf, json, os, resource, threading, time

SR = 16000


class StageError:
    """a stage raised for this row: the writer turns it into a status=error record"""

    def __init__(self, exc):
        self.exc = exc


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.t = collections.defaultdict(float)

    def add(self, k, dt):
        with self.lock:
            self.t[k] += dt

    def snapshot(self):
        with self.lock:
            return {k: round(v, 2) for k, v in self.t.items()}


def run_pipeline(items, load, forward, score, load_workers=8, window=32, score_workers=2, max_pending=16, stats=None):
    """generator: yields (item, result) in INPUT order. result = score(item, L, F), or StageError.
    load(item) -> L runs in `load_workers` threads, up to `window` items ahead; forward(item, L) -> F runs in the calling thread (serial, in order);
    score(item, L, F) -> R runs in `score_workers` threads; at most `max_pending` scored-or-scoring items are held (back-pressure)."""
    stats = stats or Stats()
    ex_load = cf.ThreadPoolExecutor(max(1, load_workers), thread_name_prefix="load")
    ex_score = cf.ThreadPoolExecutor(max(1, score_workers), thread_name_prefix="score")

    def timed(k, fn, *a):
        t = time.time()
        try:
            return fn(*a)
        finally:
            stats.add(k, time.time() - t)

    def safe_score(item, L, F):
        try:
            return timed("score", score, item, L, F)
        except Exception as e:
            return StageError(e)
    lf, pend = collections.deque(), collections.deque()
    it = iter(items)

    def fill():
        while len(lf) < window:
            try:
                x = next(it)
            except StopIteration:
                return
            lf.append((x, ex_load.submit(timed, "load", load, x)))

    def done(v):
        f = cf.Future()
        f.set_result(v)
        return f
    try:
        fill()
        while lf:
            x, f = lf.popleft()
            fill()
            t = time.time()
            try:
                L = f.result()
                stats.add("wait_load", time.time() - t)
                F = timed("forward", forward, x, L)
                pend.append((x, ex_score.submit(safe_score, x, L, F)))
            except Exception as e:
                pend.append((x, done(StageError(e))))
            while pend and (pend[0][1].done() or len(pend) > max_pending):
                x0, f0 = pend.popleft()
                yield x0, f0.result()
        while pend:
            x0, f0 = pend.popleft()
            yield x0, f0.result()
    finally:
        ex_load.shutdown(wait=False, cancel_futures=True)
        ex_score.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------------------------------------------- GPU / CPU sampler

def nvml_reader():
    """-> read() -> (util_pct, mem_used_mib) from NVML (needs the `nvidia-ml-py` wheel + the driver); None when unavailable (CPU test, missing wheel)"""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None

    def read():
        u = pynvml.nvmlDeviceGetUtilizationRates(h)
        m = pynvml.nvmlDeviceGetMemoryInfo(h)
        return float(u.gpu), m.used / 1048576
    return read


def fmt_gpu_line(part, n, busy, peak, mem_mib, cores, rss_gib, rows, rps, seq=0):
    """One utilisation line a launcher can parse: key=value pairs after the `gpu_util` tag.
    busy = NVML 'percent of time a kernel was running', mean over the n one-second samples of the last window (idle waits count as 0)"""
    f = lambda v, d=1: "na" if v is None else f"{v:.{d}f}"
    return (f"gpu_util part={part} seq={seq} n={n} busy={f(busy)} max={f(peak, 0)} mem_mib={f(mem_mib, 0)} cores={f(cores, 2)} "
            f"rss_gib={f(rss_gib, 2)} rows={rows} rps={f(rps, 1)}")


class Sampler(threading.Thread):
    """daemon thread: every `sample_s` read the GPU (NVML) and this process's CPU time; every `every_s` (5 s: short probes still get several lines) call emit(line). Never raises into the job."""

    def __init__(self, emit, part=0, every_s=5.0, sample_s=1.0, read_gpu=None, rows_done=lambda: 0, clock=time.time, cpu_time=time.process_time):
        super().__init__(daemon=True, name="gpu-sampler")
        self.emit, self.part, self.every_s, self.sample_s = emit, part, every_s, sample_s
        self.read_gpu, self.rows_done, self.clock, self.cpu_time = read_gpu, rows_done, clock, cpu_time
        self.stop_ev = threading.Event()
        self.utils, self.mems = [], []
        self.t0 = self.t_last = clock()
        self.cpu_last, self.rows_last = cpu_time(), 0
        self.lines = 0
        self.win = []          # per emitted window: (busy|None, cores, mem_mib|None, rss_gib|None)

    def summary(self):
        """whole-run numbers for done.json: mean NVML busy over all windows and over the active ones (from the first busy > 0 on, like the guard), CPU cores used, peaks"""
        b = [w[0] for w in self.win if w[0] is not None]
        first = next((i for i, v in enumerate(b) if v > 0), None)
        act = b[first:] if first is not None else []
        avg = lambda x: round(sum(x) / len(x), 1) if x else None
        mem = [w[2] for w in self.win if w[2] is not None]
        rss = [w[3] for w in self.win if w[3] is not None]
        return dict(windows=len(self.win), gpu_busy_all=avg(b), gpu_busy_active=avg(act), cores_used=avg([w[1] for w in self.win]),
                    gpu_mem_peak_mib=max(mem) if mem else None, rss_peak_gib=round(max(rss), 2) if rss else None)

    def sample(self):
        if self.read_gpu:
            try:
                u, m = self.read_gpu()
                self.utils.append(u)
                self.mems.append(m)
            except Exception:
                pass

    def line(self):
        """build + reset the window; returns the line (also counted)"""
        now, cpu, rows = self.clock(), self.cpu_time(), self.rows_done()
        dt = max(1e-6, now - self.t_last)
        busy = sum(self.utils) / len(self.utils) if self.utils else None
        peak = max(self.utils) if self.utils else None
        mem = max(self.mems) if self.mems else None
        try:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1048576 if os.uname().sysname == "Linux" else 1073741824)   # KiB on Linux, bytes on macOS
        except Exception:
            rss = None
        self.win.append((busy, (cpu - self.cpu_last) / dt, mem, rss))
        s = fmt_gpu_line(self.part, len(self.utils), busy, peak, mem, (cpu - self.cpu_last) / dt, rss, rows, (rows - self.rows_last) / dt, self.lines)
        self.utils, self.mems = [], []
        self.t_last, self.cpu_last, self.rows_last = now, cpu, rows
        self.lines += 1
        return s

    def run(self):
        last_emit = self.clock()
        while not self.stop_ev.wait(self.sample_s):
            try:
                self.sample()
                if self.clock() - last_emit >= self.every_s:
                    last_emit = self.clock()
                    self.emit(self.line())
            except Exception:
                pass

    def stop(self):
        self.stop_ev.set()
        try:
            if self.utils:
                self.emit(self.line())
        except Exception:
            pass


# ---------------------------------------------------------------------------------------------------------------- the part runner

def read_audio(path, sr_out=SR):
    """mono float32 at 16 kHz (same as a plain loop: soundfile read, channel mean, torchaudio resample)"""
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != sr_out:
        import torch, torchaudio
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, sr_out).numpy()
    return x


def run_part(rows, out_path, proc, model, vocab, C, W, done_ids=(), load_workers=8, window=32, score_workers=2, max_pending=16, flush_every=25, commit=None,
             commit_every_rows=2000, commit_every_s=120.0, log=print, progress_every_s=20.0, sampler_every_s=5.0, part=0, gpu_reader="nvml"):
    """label `rows` into out_path (appended, resumable: rows whose id is in done_ids are skipped). Returns (summary dict, hist). C = labeller_core, W = w2v_torch."""
    import numpy as np, torch, torch.nn.functional as F
    param = next(model.parameters())
    dev, half = param.device, param.dtype == torch.float16
    BL = vocab["<pad>"]
    inv = {v: k for k, v in vocab.items()}
    todo = [r for r in rows if r["id"] not in done_ids]
    stats = Stats()
    counter = [0]

    def load(r):
        x = read_audio(r["audio_filepath"])
        inp = proc(x, sampling_rate=SR, return_tensors="pt").input_values
        return x, inp

    def forward(r, L):
        inp = L[1].to(dev)
        if half:
            inp = inp.half()
        with torch.no_grad():
            return F.log_softmax(model(inp).logits.float(), -1)[0]

    def score(r, L, lp):
        t = time.time()
        x = L[0]
        with torch.no_grad():
            lossf = W.losses_fn(lp, BL)
            lpc = lp.cpu()
            hyp = W.greedy_text(lpc, BL, inv)
            rec = C.score_row(r, lossf, free_hyp=hyp, align_fn=W.align_fn(lpc, BL))
        rec["duration"] = round(len(x) / SR, 3)
        rec["rms_onset_s"] = W.rms_onset(x)
        rec["t_s"] = round(time.time() - t, 3)
        return rec

    reader = nvml_reader() if gpu_reader == "nvml" and dev.type == "cuda" else (gpu_reader if callable(gpu_reader) else None)
    sampler = Sampler(log, part=part, every_s=sampler_every_s, read_gpu=reader, rows_done=lambda: counter[0])
    sampler.start()
    t0 = time.time(); hist = {}; n_new = 0; last_commit = time.time(); last_prog = 0.0; since_commit = 0
    try:
        with open(out_path, "a", encoding="utf-8") as fo:
            for r, res in run_pipeline(todo, load, forward, score, load_workers, window, score_workers, max_pending, stats):
                rec = res if not isinstance(res, StageError) else dict(id=r["id"], label=r.get("label"), status="error", error=repr(res.exc)[:200], spans=[])
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                hist[rec["status"]] = hist.get(rec["status"], 0) + 1; n_new += 1; counter[0] = n_new; since_commit += 1
                if n_new % flush_every == 0:
                    fo.flush()
                now = time.time()
                if commit and (since_commit >= commit_every_rows or now - last_commit >= commit_every_s):
                    fo.flush(); commit(); last_commit = now; since_commit = 0
                if now - last_prog >= progress_every_s:
                    last_prog = now
                    log(f"label {n_new + len(done_ids)}/{len(rows)} t={round(now - t0)}s")
            fo.flush()
    finally:
        sampler.stop()
    log(f"label {len(rows)}/{len(rows)} done stages={stats.snapshot()} sampler={sampler.summary()}")
    return dict(rows_in=len(rows), rows_resumed=len(done_ids), rows_new=n_new, status=hist, sec=round(time.time() - t0), stages=stats.snapshot(), sampler=sampler.summary()), hist
