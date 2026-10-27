import sys, os, time, random, threading
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import label_runner as R


def test_order_is_input_order_with_random_latency():
    random.seed(3)
    lat = {i: random.random() * 0.01 for i in range(60)}
    def load(i): time.sleep(lat[i]); return i * 10
    def fwd(i, L): return L + 1
    def score(i, L, F): time.sleep(lat[(i * 7) % 60]); return (i, L, F)
    out = list(R.run_pipeline(range(60), load, fwd, score, load_workers=8, window=16, score_workers=3))
    assert [x for x, _ in out] == list(range(60))
    assert all(r == (i, i * 10, i * 10 + 1) for i, r in out)


def test_stage_errors_stay_on_their_row_and_do_not_stop_the_run():
    def load(i):
        if i == 3: raise FileNotFoundError("gone")
        return i
    def fwd(i, L):
        if i == 5: raise RuntimeError("cuda oom")
        return L
    def score(i, L, F):
        if i == 7: raise ValueError("bad")
        return i
    out = list(R.run_pipeline(range(10), load, fwd, score, window=4))
    kinds = {i: type(r.exc).__name__ if isinstance(r, R.StageError) else "ok" for i, r in out}
    assert [i for i, _ in out] == list(range(10))
    assert kinds[3] == "FileNotFoundError" and kinds[5] == "RuntimeError" and kinds[7] == "ValueError"
    assert sum(1 for v in kinds.values() if v == "ok") == 7


def test_window_bounds_read_ahead_and_pending_bounds_scoring_backlog():
    started, consumed = [0], [0]; lock = threading.Lock(); peak = [0]
    def load(i):
        with lock: started[0] += 1; peak[0] = max(peak[0], started[0] - consumed[0])
        return i
    def score(i, L, F): time.sleep(0.002); return i
    for _, _ in R.run_pipeline(range(80), load, lambda i, L: L, score, load_workers=4, window=6, score_workers=1, max_pending=3):
        with lock: consumed[0] += 1
    assert peak[0] <= 6 + 3 + 2, peak           # window + pending + the one in the main thread (+ slack)


def test_forward_runs_on_the_calling_thread_in_order():
    seen = []; main = threading.get_ident()
    def fwd(i, L): seen.append((i, threading.get_ident())); return L
    list(R.run_pipeline(range(20), lambda i: i, fwd, lambda i, L, F: i, window=5))
    assert [i for i, _ in seen] == list(range(20)) and all(t == main for _, t in seen)


def test_overlap_beats_sequential():
    n, a, b, c = 40, 0.02, 0.01, 0.02
    def load(i): time.sleep(a); return i
    def fwd(i, L): time.sleep(b); return L
    def score(i, L, F): time.sleep(c); return i
    t = time.time(); list(R.run_pipeline(range(n), load, fwd, score, load_workers=8, window=16, score_workers=3)); par = time.time() - t
    seq = n * (a + b + c)
    assert par < 0.55 * seq, (par, seq)         # ideal ~ n * max(stage) = 40% of sequential


def test_generator_close_does_not_hang():
    g = R.run_pipeline(range(1000), lambda i: i, lambda i, L: L, lambda i, L, F: i, window=4)
    assert next(g)[0] == 0
    g.close()


def test_gpu_line_roundtrip_and_na():
    s = R.fmt_gpu_line(2, 10, 63.24, 100.0, 5120.4, 1.623, 2.912, 1200, 18.44)
    assert s.startswith("gpu_util ") and "part=2" in s and "busy=63.2" in s and "cores=1.62" in s and "rows=1200" in s
    kv = dict(p.split("=") for p in s.split()[1:])
    assert kv["max"] == "100" and kv["mem_mib"] == "5120" and kv["rps"] == "18.4"
    assert "busy=na" in R.fmt_gpu_line(0, 0, None, None, None, 0.5, None, 0, 0.0)


def test_sampler_windows_and_resets():
    t = [0.0]; cpu = [0.0]; rows = [0]; vals = iter([(50.0, 1000.0), (100.0, 2000.0), (0.0, 1500.0)])
    lines = []
    s = R.Sampler(lines.append, part=1, read_gpu=lambda: next(vals), rows_done=lambda: rows[0], clock=lambda: t[0], cpu_time=lambda: cpu[0])
    for k in range(3):
        s.sample()
    t[0] = 10.0; cpu[0] = 15.0; rows[0] = 200
    l = s.line()
    assert "busy=50.0" in l and "max=100" in l and "mem_mib=2000" in l and "cores=1.50" in l and "rows=200" in l and "rps=20.0" in l and "n=3" in l and "part=1" in l
    assert s.utils == [] and s.line().count("busy=na") == 1       # window reset; no samples -> na (never a fake 0)


def test_sampler_summary_skips_idle_startup_windows():
    t = [0.0]; cpu = [0.0]; busy = iter([(0.0, 100.0), (60.0, 200.0), (80.0, 300.0)])
    s = R.Sampler(lambda l: None, read_gpu=lambda: next(busy), clock=lambda: t[0], cpu_time=lambda: cpu[0])
    for k in range(3):
        s.sample(); t[0] += 10; cpu[0] += 10; s.line()
    m = s.summary()
    assert (m["windows"], m["gpu_busy_all"], m["gpu_busy_active"], m["cores_used"], m["gpu_mem_peak_mib"]) == (3, 46.7, 70.0, 1.0, 300.0), m


def test_sampler_thread_emits_and_stops():
    lines = []
    s = R.Sampler(lines.append, every_s=0.05, sample_s=0.01, read_gpu=lambda: (80.0, 100.0))
    s.start(); time.sleep(0.3); s.stop(); s.join(1)
    assert len(lines) >= 2 and all("busy=80.0" in l for l in lines)


def test_sampler_survives_a_failing_reader():
    def bad(): raise RuntimeError("nvml gone")
    lines = []
    s = R.Sampler(lines.append, every_s=0.02, sample_s=0.005, read_gpu=bad)
    s.start(); time.sleep(0.1); s.stop(); s.join(1)
    assert lines and all("busy=na" in l for l in lines)


def test_nvml_reader_is_none_without_the_wheel_or_driver():
    r = R.nvml_reader()
    assert r is None or callable(r)


if __name__ == "__main__":
    import tinytest
    tinytest.run(globals())
