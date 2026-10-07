"""The refactored simulate()/solve_policy() must reproduce the ORIGINAL
(tests/sim_reference.py, the code that produced results/sim_paging.json)
exactly: same served count, same sync bytes, same async bytes, for every
policy, with and without a finite prefetch allowance, on traces with
heavy timestamp ties (many experts touched in the same token).

Needs numpy: /tmp/specexp-venv/bin/python tests/test_sim_equivalence.py
"""
import os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import sim_paging as new  # noqa: E402
import sim_reference as ref  # noqa: E402

L, K, E = new.L, new.K, new.E


def zipf_trace(T, s=0.8, persist=0.3, seed=0):
    rng = np.random.default_rng(seed)
    w = 1.0 / np.arange(1, E + 1) ** s
    gold = np.zeros((T, L, K), np.int16)
    for li in range(L):
        perm = rng.permutation(E)
        p = w[perm] / w.sum()
        prev = rng.choice(E, K, replace=False, p=p)
        for t in range(T):
            keep = prev[rng.random(K) < persist]
            q = p.copy(); q[keep] = 0; q /= q.sum()
            new_e = rng.choice(E, K - len(keep), replace=False, p=q)
            cur = np.concatenate([keep, new_e])
            gold[t, li] = cur
            prev = cur
    picks = np.zeros((T, L, 12), np.int16)
    for t in range(T):
        for li in range(L):
            # half the picks are right, half random -> realistic waste
            right = gold[t, li][rng.permutation(K)[:6]]
            wrong = rng.choice(E, 6, replace=False)
            picks[t, li] = np.concatenate([right, wrong])
    prior = {li: np.argsort(-np.bincount(gold[:, li].ravel(), minlength=E))
             for li in range(L)}
    return gold, picks, prior


def main():
    gold, picks, prior = zipf_trace(T=150)
    fails = 0
    cases = []
    for cap in (32, 64, 128):
        for mode, pick in (("lru", 0), ("prior", 0), ("probe", 6),
                           ("probe", 12), ("sidecar", 0)):
            for allow in (10 ** 9, 40, 7):
                cases.append((cap, mode, pick, allow))
    t_ref = t_new = 0.0
    for cap, mode, pick, allow in cases:
        t0 = time.perf_counter()
        r = ref.simulate(gold, picks, prior, cap, mode, pick, allow)
        t1 = time.perf_counter()
        n = new.simulate(gold, picks, prior, cap, mode, pick, allow)
        t2 = time.perf_counter()
        t_ref += t1 - t0; t_new += t2 - t1
        same = all(abs(x - y) < 1e-12 for x, y in zip(r, n))
        if not same:
            fails += 1
            print(f"MISMATCH cap={cap} mode={mode} pick={pick} allow={allow}"
                  f"\n   ref={r}\n   new={n}")
    print(f"{len(cases)} cases, {fails} mismatches; "
          f"reference {t_ref:.1f}s vs refactor {t_new:.1f}s")
    spec = new.TIERS["32GB-M4P"]
    for cap, mode, pick in ((32, "probe", 6), (64, "sidecar", 0),
                            (64, "lru", 0), (128, "prior", 0)):
        r = ref.solve_policy(gold, picks, prior, cap, mode, pick, spec)
        n = new.solve_policy(gold, picks, prior, cap, mode, pick, spec,
                             throttle="legacy")
        if any(abs(x - y) > 1e-9 for x, y in zip(r, n)):
            fails += 1
            print(f"solve_policy MISMATCH cap={cap} mode={mode}: {r} vs {n}")
        else:
            print(f"solve_policy cap={cap} mode={mode}: tps={n[3]:.2f} identical")
    if fails:
        print("FAIL"); sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
