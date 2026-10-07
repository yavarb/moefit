"""T3 fidelity, part 2: LRU hit-refresh bug impact + per-miss cost model.

Two questions this cycle, both against the measured Santa Cruz point
(13.0 tok/s = 76.92 ms/token, cap=143 experts/layer, synth traces):

1. HIT-REFRESH BUG. The shipped simulate() counts a hit
   (`if e in res[li]: served += 1`) WITHOUT refreshing the resident's
   LRU timestamp, so eviction order among survivors is insertion order
   (FIFO), not recency. The swarm retro says fix this "first". Before
   anyone re-benches eviction policies on the buggy baseline, quantify
   what the fix changes at the MATCHED operating point: served fraction,
   sync MB/token, and predicted tok/s. Implemented here as a standalone
   cache simulator (lru mode has no pins/prefetch, so the dynamics are
   just per-layer caches over the gold sequence); validated by
   reproducing the shipped numbers bit-for-bit with hit_refresh=False.

2. PER-MISS COST MODEL. gap_report found ~49.6 ms/token unexplained by
   both roofline terms. Decode is layer-serial, so a token's SSD cost is
   sum over layers of (misses_in_layer x per-miss cost), where per-miss
   cost = chunk_transfer + fixed overhead. Given measured 76.92 ms and
   each policy's (misses/token, sync MB/token), solve for the fixed
   per-miss overhead the SSD bandwidth hypothesis requires. This is the
   analytic complement to T7's Monte-Carlo over the same knobs.

usage: python3 experiments/fidelity_miss_model.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import sim_paging as sp  # noqa: E402

MEASURED_MS = 1000.0 / 13.0          # 76.92 ms/token
CAP = 143
SSD_ASSUMED = 7.4                    # GB/s, sim's M4 Max tier assumption


def lru_stats(gold, cap, hit_refresh):
    """Per-layer capacity-`cap` cache over gold (T, L, K).

    hit_refresh=False reproduces shipped sim_paging lru semantics
    (timestamps updated on insert only); True is true LRU.
    Returns (served_fraction, misses_per_token, per_layer_miss_means).
    """
    T, L, K = gold.shape
    hits = misses = 0
    layer_misses = np.zeros(L)
    caches = [dict() for _ in range(L)]
    for t in range(T):
        for li in range(L):
            c = caches[li]
            for e in gold[t, li]:
                e = int(e)
                if e in c:
                    hits += 1
                    if hit_refresh:
                        c[e] = t
                else:
                    misses += 1
                    layer_misses[li] += 1
                    if len(c) >= cap:
                        victim = min(c.items(), key=lambda kv: kv[1])[0]
                        del c[victim]
                    c[e] = t
    per_layer = (layer_misses / T).tolist()
    return hits / (T * L * K), misses / T, per_layer


def main():
    bf, bl, bT = sp.load_split(ROOT / "results/traces_synth/build.npz")
    hf, hl, hT = sp.load_split(ROOT / "results/traces_synth/holdout.npz")
    rng = np.random.default_rng(0)
    hs = np.sort(rng.choice(hT, size=min(8000, hT), replace=False))
    gold = np.stack([hl[li][hs] for li in range(sp.L)], axis=1).astype(np.int16)

    out = {"measured_ms_per_token": round(MEASURED_MS, 2),
           "cap": CAP, "traces": "synth (beta 2.0, shift 0.3, seed 0)"}

    # ---- 1. hit-refresh bug impact ----------------------------------
    bug = {}
    for refresh in (False, True):
        srv, mpt, per_layer = lru_stats(gold, CAP, refresh)
        sync_mb = mpt * sp.EXPERT_MIB
        st_ms = (sync_mb + sp.PLE_STREAM_MIB) / 1024.0 / SSD_ASSUMED * 1000.0
        c_ms = (sp.READ_FLOOR_MIB + sp.GOLD_MIB) / 1024.0 \
            / (546.0 * sp.DRAM_EFF) * 1000.0
        label = "true_lru" if refresh else "shipped_lru_no_refresh"
        bug[label] = {
            "served": round(srv, 3),
            "misses_per_token": round(mpt, 2),
            "sync_mb_per_token": round(sync_mb, 1),
            "predicted_tps_at_ssd7.4": round(1000.0 / max(c_ms, st_ms), 1),
            "mean_layer_misses": round(float(np.mean(per_layer)), 3),
            "max_layer_misses": round(float(np.max(per_layer)), 3),
        }
    # validation: shipped-semantics replica must match sp.simulate lru
    sp.pin_counts = None
    zero_picks = np.zeros((len(gold), sp.L, 1), np.int16)
    prior_rank = {li: np.argsort(-np.bincount(bl[li].ravel(), minlength=sp.E))
                  for li in range(sp.L)}
    srv_sim, sync_sim, _ = sp.simulate(gold, zero_picks, prior_rank,
                                       CAP, "lru", 0, 0)
    bug["validation_vs_shipped_simulate"] = {
        "replica_served": bug["shipped_lru_no_refresh"]["served"],
        "simulate_served": round(srv_sim, 3),
        "replica_sync_mb": bug["shipped_lru_no_refresh"]["sync_mb_per_token"],
        "simulate_sync_mb": round(sync_sim, 1),
    }
    delta = bug["true_lru"]["served"] - bug["shipped_lru_no_refresh"]["served"]
    bug["fix_delta"] = {
        "served_delta_pp": round(100 * delta, 2),
        "predicted_tps_delta": round(bug["true_lru"]["predicted_tps_at_ssd7.4"]
                                     - bug["shipped_lru_no_refresh"]["predicted_tps_at_ssd7.4"], 1),
    }
    out["hit_refresh_bug"] = bug

    # ---- 2. per-miss cost feasibility -------------------------------
    # policies -> (misses/tok, sync MB/tok) at cap 143
    pol = {
        "shipped_lru": (bug["shipped_lru_no_refresh"]["misses_per_token"],
                        bug["shipped_lru_no_refresh"]["sync_mb_per_token"]),
        "true_lru": (bug["true_lru"]["misses_per_token"],
                     bug["true_lru"]["sync_mb_per_token"]),
    }
    sp_srv, sp_sync, _ = sp.simulate(gold, zero_picks, prior_rank,
                                     CAP, "prior", 0, 0)
    pol["prior"] = (round((1 - sp_srv) * sp.L * sp.K, 2), round(sp_sync, 1))
    mask = np.zeros(sp.E, bool)
    mask[:CAP] = True
    miss_first = (~mask[gold]).sum() / len(gold)
    pol["static_first"] = (round(miss_first, 2),
                           round(miss_first * sp.EXPERT_MIB, 1))

    feas = {}
    for name, (mpt, sync_mb) in pol.items():
        rows = {}
        for bw in (2.5, 3.0, 4.0, 5.46, 7.4):
            bytes_ms = (sync_mb + sp.PLE_STREAM_MIB) / 1024.0 / bw * 1000.0
            resid = MEASURED_MS - bytes_ms
            rows[f"bw{bw}"] = {
                "bytes_time_ms": round(bytes_ms, 1),
                "fixed_overhead_ms_per_miss": round(resid / mpt, 3) if resid >= 0 else None,
                "feasible": resid >= -1e-9,
            }
        per_miss_transfer_ms = sp.EXPERT_MIB / 1024.0 / 7.4 * 1000.0
        feas[name] = {
            "misses_per_token": mpt, "sync_mb_per_token": sync_mb,
            "latency_only_ms_per_miss": round(MEASURED_MS / mpt, 3),
            "pure_transfer_ms_per_miss_at7.4": round(per_miss_transfer_ms, 3),
            "by_bw": rows,
        }
    out["per_miss_cost_feasibility"] = feas
    out["interpretation"] = (
        "latency_only = fixed per-miss cost with zero bandwidth term; "
        "pure_transfer = 2.69 MiB at 7.4 GB/s. If silicon shows per-miss "
        "overhead between these, split overhead vs BW accordingly; a "
        "bw whose rows are infeasible is ruled out for that policy."
    )

    outp = ROOT / "results/fidelity_miss_model.json"
    outp.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    print("wrote", outp)


if __name__ == "__main__":
    main()
