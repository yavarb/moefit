"""Per-layer capacity plumbing for the heterogeneous-allocation experiment.

Checks, on a small synthetic trace:
  1. a per-layer cap list changes the result (so the plumbing is live);
  2. the greedy allocator spends exactly the budget and respects the
     per-layer minimum;
  3. the per-token capacity audit fires when a layer's cap is smaller
     than the pin budget plus the dynamic floor (the leak the audit is
     for), and stays silent for every legal configuration.

Needs numpy: /tmp/moefit-venv/bin/python tests/test_hetero_alloc.py
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import sim_paging as sp  # noqa: E402
import hetero_alloc as ha  # noqa: E402
from test_sim_equivalence import zipf_trace  # noqa: E402

L = sp.L


def main():
    fails = 0
    gold, picks, prior = zipf_trace(T=120, seed=3)
    u = sp.simulate(gold, None, prior, 32, "lru", 0, 10 ** 9)
    skew = [16] * (L // 2) + [48] * (L // 2)
    s = sp.simulate(gold, None, prior, skew, "lru", 0, 10 ** 9)
    print(f"uniform 32 served={u[0]:.4f}  skewed 16/48 served={s[0]:.4f}")
    if abs(u[0] - s[0]) < 1e-6:
        print("FAIL: per-layer caps had no effect"); fails += 1

    lay = {li: gold[:, li] for li in range(L)}
    caps, _ = ha.allocate_caps(lay, 32 * L)
    print(f"allocator: sum={sum(caps)} min={min(caps)} max={max(caps)}")
    if sum(caps) != 32 * L or min(caps) < ha.MIN_LAYER_CAP:
        print("FAIL: allocator budget/minimum violated"); fails += 1
    h = sp.simulate(gold, None, prior, caps, "lru", 0, 10 ** 9)
    print(f"hetero served={h[0]:.4f} (uniform {u[0]:.4f})")

    # audit must fire: cap 8 with sidecar pins 10 + dyn floor 4 -> 14 > 8
    leaky = [8] * L
    try:
        sp.simulate(gold, None, prior, leaky, "sidecar", 0, 10 ** 9)
        print("FAIL: capacity audit did not fire on a leaky config"); fails += 1
    except AssertionError as e:
        print("audit fired as expected:", str(e)[:70])
    # audit must stay silent on legal configs
    for mode, pick in (("lru", 0), ("prior", 0), ("probe", 12), ("sidecar", 0)):
        try:
            sp.simulate(gold, picks, prior, caps, mode, pick, 10 ** 9)
        except AssertionError as e:
            print(f"FAIL: audit fired on legal config {mode}: {e}"); fails += 1
    if fails:
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
