"""Repro: the prefetch throttle in solve_policy() cannot throttle when the
SSD is the bottleneck.

solve_policy() computes the per-token prefetch allowance from
    spare = SSD_rate * s_ms - (sync + PLE),   s_ms = max(compute_ms, ssd_ms)
When ssd_ms > compute_ms this reduces algebraically to spare == async, i.e.
"you may prefetch exactly as much as you just did". The fixed point then
settles on the unthrottled prefetch volume, so a predictor with 75% wasted
prefetches saturates the SSD and every token takes longer. The docstring
says async reads are "hidden by design"; in this regime they are not.

Uniform-random routing at cap 32 on the 24 GB tier is SSD-bound for every
policy, so it exercises the regime directly. Needs numpy (as the simulator
does): run with a python that has numpy, e.g.
    /tmp/specexp-venv/bin/python tests/test_sim_throttle.py
"""
import sys, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import sim_paging as sp  # noqa: E402

L, K, E = sp.L, sp.K, sp.E


def make_trace(T, seed=0):
    rng = np.random.default_rng(seed)
    gold = np.zeros((T, L, K), np.int16)
    picks = np.zeros((T, L, 6), np.int16)
    for t in range(T):
        for li in range(L):
            gold[t, li] = rng.choice(E, K, replace=False)
            picks[t, li] = rng.choice(E, 6, replace=False)   # ~all wasted
    prior = {li: np.arange(E) for li in range(L)}
    return gold, picks, prior


def spare_legacy(spec, sync_mb, async_mb, c_ms, st_ms):
    s_ms = max(c_ms, st_ms)
    return max(spec["ssd"] * 1024.0 / 1000.0 * s_ms
               - (sync_mb + sp.PLE_STREAM_MIB), 0.0)


def main():
    gold, picks, prior = make_trace(T=120)
    spec = sp.TIERS["24GB-M4"]
    cap = 32
    kw = {}
    if "throttle" in sp.solve_policy.__code__.co_varnames:
        kw["throttle"] = "legacy"
    lru = sp.solve_policy(gold, picks, prior, cap, "lru", 0, spec, **kw)
    pr = sp.solve_policy(gold, picks, prior, cap, "probe", 6, spec, **kw)
    srv, sync_mb, async_mb, tps, c_ms, st_ms = pr
    spare = spare_legacy(spec, sync_mb, async_mb, c_ms, st_ms)
    print(f"LRU   : tps={lru[3]:.1f} compute_ms={lru[4]:.1f} ssd_ms={lru[5]:.1f}"
          f" sync={lru[1]:.0f}MB async={lru[2]:.0f}MB")
    print(f"probe : tps={tps:.1f} compute_ms={c_ms:.1f} ssd_ms={st_ms:.1f}"
          f" sync={sync_mb:.0f}MB async={async_mb:.0f}MB")
    print(f"legacy spare formula at the fixed point = {spare:.1f} MB/token;"
          f" async actually issued = {async_mb:.1f} MB/token")
    ok = True
    if st_ms <= c_ms:
        print("test setup error: regime is not SSD-bound"); sys.exit(2)
    if abs(spare - async_mb) > 1.0:
        print("unexpected: spare != async in SSD-bound regime"); ok = False
    if async_mb < 0.5 * 6 * L * sp.EXPERT_MIB:
        print("unexpected: prefetch was throttled"); ok = False
    print("legacy throttle is a no-op when SSD-bound:",
          "CONFIRMED" if ok else "not reproduced")
    if "throttle" in kw:
        fixed = sp.solve_policy(gold, picks, prior, cap, "probe", 6, spec,
                                throttle="compute")
        print(f"fixed (--throttle compute): tps={fixed[3]:.1f} "
              f"compute_ms={fixed[4]:.1f} ssd_ms={fixed[5]:.1f} "
              f"sync={fixed[1]:.0f}MB async={fixed[2]:.0f}MB")
        if fixed[2] > 0.1 * async_mb:
            print("FAIL: compute-window throttle did not cut prefetch")
            sys.exit(1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
