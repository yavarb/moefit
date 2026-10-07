"""How much does the prefetch-throttle fix change the probe and sidecar
rows? Run on a trace with CALIBRATED probe precision.

The ridge probe on synthetic features is far weaker than the real one
(coverage 0.10 at 14 candidates vs 0.44 measured), so instead of fitting
a probe, this script builds candidate lists directly: each gold expert of
token t+1 enters the P picks with probability q, the rest are wasted
picks drawn from the layer's build popularity. q is set so precision
matches the real probe (0.44 x 10 / 14 = 0.31 of picks are right, i.e.
about 70% of prefetches wasted, matching the ~75% in the shipped notes).

Then solve_policy() runs probe (P = 6, 12) and sidecar under
--throttle legacy (shipped) and --throttle compute (fix) at the shipped
caps and tiers. Output: results/throttle_delta_synth.json.

    .venv/bin/python experiments/throttle_delta.py --traces-dir results/traces_synth
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import sim_paging as sp  # noqa: E402

L, K, E = sp.L, sp.K, sp.E


def calibrated_picks(gold, prior_rank, P, precision, rng):
    """[T,L,P] picks: each gold expert kept with prob q = precision*P/K,
    the remaining slots filled with popular-but-wrong experts."""
    T = len(gold)
    q = min(1.0, precision * P / K)
    out = np.zeros((T, L, P), np.int16)
    for li in range(L):
        pop = prior_rank[li][:64]            # plausible wrong guesses
        for t in range(T):
            g = gold[t, li]
            keep = g[rng.random(K) < q][:P]
            gs = set(g.tolist())
            fill = [e for e in rng.permutation(pop).tolist()
                    if e not in gs][:P - len(keep)]
            out[t, li] = np.concatenate([keep, np.array(fill, np.int16)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-dir", default=str(ROOT / "results/traces_synth"))
    ap.add_argument("--caps", default="32,64,128")
    ap.add_argument("--precision", type=float, default=0.31)
    ap.add_argument("--out", default=str(ROOT / "results/throttle_delta_synth.json"))
    a = ap.parse_args()
    tdir = Path(a.traces_dir)
    bf, bl, bT = sp.load_split(tdir / "build.npz")
    hf, hl, hT = sp.load_split(tdir / "holdout.npz")
    gold = np.stack([hl[li][:hT] for li in range(L)], axis=1).astype(np.int16)
    prior = {li: np.argsort(-np.bincount(bl[li].ravel(), minlength=E))
             for li in range(L)}
    rng = np.random.default_rng(0)
    picks = {P: calibrated_picks(gold, prior, P, a.precision, rng)
             for P in (6, 12)}
    for P, pk in picks.items():
        hit = sum(len(set(pk[t, li].tolist()) & set(gold[t, li].tolist()))
                  for t in range(hT) for li in range(L))
        print(f"P={P}: coverage {hit / (hT * L * K):.3f} precision "
              f"{hit / (hT * L * P):.3f}", flush=True)
    out = []
    print(f"{'cap':>4} {'mode':8} {'P':>2} {'tier':9} {'throttle':8} "
          f"{'served':>6} {'sync':>6} {'async':>6} {'ct':>5} {'st':>5} {'tps':>5}")
    for cap in [int(c) for c in a.caps.split(",")]:
        for mode, P in (("lru", 0), ("probe", 6), ("probe", 12), ("sidecar", 0)):
            for tier, spec in sp.TIERS.items():
                if tier == "64GB-M4X":
                    continue
                if sp.FLOOR_MIB / 1024 + cap * L * sp.EXPERT_MIB / 1024 > spec["usable"]:
                    continue
                for throttle in ("legacy", "compute"):
                    if mode == "lru" and throttle == "compute":
                        continue
                    srv, sync_mb, async_mb, tps, c_ms, st_ms = sp.solve_policy(
                        gold, picks.get(P), prior, cap, mode, P, spec,
                        throttle=throttle)
                    row = dict(cap=cap, mode=mode, P=P, tier=tier,
                               throttle=throttle, served=round(srv, 3),
                               sync_mb=round(sync_mb, 1),
                               async_mb=round(async_mb, 1),
                               ct_ms=round(c_ms, 1), st_ms=round(st_ms, 1),
                               tps=round(tps, 1))
                    out.append(row)
                    print(f"{cap:>4} {mode:8} {P:>2} {tier:9} {throttle:8} "
                          f"{srv:6.3f} {sync_mb:6.0f} {async_mb:6.0f} "
                          f"{c_ms:5.1f} {st_ms:5.1f} {tps:5.1f}", flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
