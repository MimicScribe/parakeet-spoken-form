import sys, traceback


def run(ns):
    fails = 0; n = 0
    for k, v in sorted(ns.items()):
        if k.startswith("test_") and callable(v):
            n += 1
            try:
                v()
            except Exception:
                fails += 1; print("FAIL", k); traceback.print_exc()
    print("%d tests, %d failed" % (n, fails))
    sys.exit(1 if fails else 0)
