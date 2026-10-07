"""T6 local analysis A: routing-traffic characterization + mechanism headroom.

Complements T1 (gap_report: sim 36.6 vs silicon 13.0 tok/s at matched
cap=143, 49.6 ms/tok unexplained) and T3 (fidelity_m4max_36gb.py: constant
calibration). This answers traffic-side questions that decide WHICH
mechanism can close what:

  1. Reuse-distance law: per layer, the distribution of D = number of
     distinct other experts touched between an expert's previous touch and
     its current touch. D < C is exactly an LRU hit at capacity C, so one
     pass gives the whole analytic served(C) curve and the per-token miss
     sequence (burstiness), which the sim collapses to a mean.
  2. LRU miss dynamics at the silicon operating point (cap=143):
     per-token sync-miss distribution. The p99/mean SSD MB burst ratio is
     a candidate contributor to the sim<->silicon gap that a flat-rate
     SSD term ignores.
  3. Mechanism headroom at cap=143: LRU vs static prior (pin half) vs
     oracle next-token sidecar via the shipped simulator. Decomposes the
     traffic into floor (LRU) + pinning gain + perfect-prediction gain —
     tells design_inventor whether prediction or pinning is the lever,
     and how much of the gap is traffic-side vs model-side.

Algorithm (exact, per layer, O(T*K*log T)):
  Sweep tokens t. For touch of expert y with previous touch p:
  D = |{z != y : z touched in (p, t)}|.
  Maintain a Fenwick over "previous touch" values: when z is touched at t
  with previous touch q, every future query with p in [q, t) must see z's
  first post-boundary touch -> range-add +1 over [q, t-1]; at query time
  point-read at p. Experts touched at token t itself have last=z's own t
  and are not in the BIT yet (queries run before updates), so they are
  added via the (K-1) - dup term, where dup counts same-token others whose
  last[z] > p (already inside (p, t) via an earlier token — impossible
  since then their marker would cover p; the dup term guards exactly that
  case and keeps the count distinct). Verified against the shipped
  sp.simulate LRU served fraction at cap=512-equivalent (see main()).

SIMULATED on the synthetic traces (results/traces_synth, fitted to the real
router traces per calibration.json: overall prior14 fit 0.188 vs target
0.187). No silicon numbers are invented here; the only measured reference
is results/measured_m4max_36gb.json (13.0 tok/s @ cap=143).

usage: python3 experiments/analysis_t6_traffic_headroom.py [--eval-sub 8000]
writes results/analysis_t6_traffic_headroom.json (+ misses npy)
"""
import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import sim_paging as sp  # noqa: E402

CAP = 143            # silicon operating point (round(0.28 x 512))
E, L, K = sp.E, sp.L, sp.K


def sweep_layer(gold, li):
    """Exact per-touch distinct stack distance D[t, j] for one layer.

    Linearize touches at position s = t*K + j (slot order = the order
    brute-force LRU consumes them). For a touch of z at s with previous
    touch q: every future query whose boundary p satisfies q < p < s sees
    z's first post-boundary touch at s -> Fenwick range-add +1 over
    positions [q+1, s-1] (for cold q=-1 that is [0, s-1], which is right:
    any later query boundary already seen z's first touch). Query at
    p = last position of the querying expert counts exactly the distinct
    z != y touched in (p, s). D < cap <=> LRU(cap) hit, position-exact.
    """
    T = gold.shape[0]
    n = T * K
    bit = [0] * (n + 3)
    last = [-1] * E
    D = np.zeros((T, K), np.int32)

    def point(p):                      # value at linear position p
        s = 0
        i = p + 2                      # 1-based index for position p (shift 1)
        while i > 0:
            s += bit[i]
            i -= i & -i
        return s

    def rng_add(a, b):                 # +1 over linear positions [a, b]
        i = a + 2
        while i <= n + 2:
            bit[i] += 1
            i += i & -i
        j = b + 3
        while j <= n + 2:
            bit[j] -= 1
            j += j & -j

    col = gold[:, li, :]
    for t in range(T):
        row = col[t]
        base = t * K
        for j in range(K):
            y = int(row[j])
            s = base + j
            p = last[y]
            if p < 0:
                D[t, j] = n + 1        # cold touch: compulsory miss
            else:
                D[t, j] = point(p)
            rng_add(p + 1, s - 1)
            last[y] = s
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-sub", type=int, default=8000)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--out", default=str(
        ROOT / "results/analysis_t6_traffic_headroom.json"))
    a = ap.parse_args()

    bf, bl, bT = sp.load_split(ROOT / "results/traces_synth/build.npz")
    hf, hl, hT = sp.load_split(ROOT / "results/traces_synth/holdout.npz")
    rng = np.random.default_rng(0)
    hs = np.sort(rng.choice(hT, size=min(a.eval_sub, hT), replace=False))
    gold = np.stack([hl[li][hs] for li in range(L)], axis=1).astype(np.int16)
    T = gold.shape[0]
    res = {"kind": "SIMULATED", "traces": "results/traces_synth (holdout)",
           "eval_tokens": int(T), "cap": a.cap,
           "silicon_ref": "measured_m4max_36gb.json: 13.0 tok/s @ cap 143"}
    print(f"eval tokens={T}", flush=True)

    # ---- one sweep per layer: D + per-token misses at cap -------------
    t0 = time.time()
    all_D = []
    per_layer = {}
    misses = np.zeros(T, np.int64)
    for li in range(L):
        D = sweep_layer(gold, li)
        hit = D < a.cap
        misses += (~hit).sum(axis=1)
        d = D.ravel()
        cold_frac = float((d >= T * K).mean())
        pl = {"touches": int(d.size),
              "cold_frac": round(cold_frac, 3),
              "median_D": int(np.median(d)),
              "p90_D": int(np.percentile(d, 90)),
              "p99_D": int(np.percentile(d, 99)),
              "served_at_64": round(float((d < 64).mean()), 3),
              "served_at_cap": round(float((d < a.cap).mean()), 3),
              "served_at_256": round(float((d < 256).mean()), 3),
              "served_at_384": round(float((d < 384).mean()), 3)}
        per_layer[li] = pl
        all_D.append(d)
        if li % 12 == 11:
            print(f"  layer {li+1}/48 done {time.time()-t0:.0f}s", flush=True)
    D_all = np.concatenate(all_D)
    sweep_s = time.time() - t0

    # analytic served curve (cold touches count as misses: d >= T*K fails
    # every d < C test, which is correct — first touch is a compulsory miss)
    res["reuse_distance"] = {
        "sweep_runtime_s": round(sweep_s, 1),
        "percentiles_D": {f"p{q}": int(np.percentile(D_all, q))
                          for q in (50, 75, 90, 95, 99)},
        "cold_frac_overall": round(float((D_all >= T * K).mean()), 3),
        "served_curve_analytic": {
            str(c): round(float((D_all < c).mean()), 3)
            for c in (16, 32, 64, 96, a.cap, 192, 256, 384, 448)},
        "per_layer": per_layer,
    }
    s_cap = np.array([per_layer[li]["served_at_cap"] for li in range(L)])
    order = np.argsort(s_cap)
    res["reuse_distance"]["coldest12_layers"] = [int(i) for i in order[:12]]
    res["reuse_distance"]["hottest12_layers"] = [int(i) for i in order[-12:]]
    res["reuse_distance"]["served_cap_mean_std_min_max"] = [
        round(float(s_cap.mean()), 3), round(float(s_cap.std()), 3),
        round(float(s_cap.min()), 3), round(float(s_cap.max()), 3)]
    print("served_curve_analytic:",
          res["reuse_distance"]["served_curve_analytic"], flush=True)

    # ---- 2. LRU miss dynamics per token at cap (from the same sweep) --
    mb = misses * sp.EXPERT_MIB
    res["lru_miss_dynamics"] = {
        "cap": a.cap,
        "residency_served_analytic": round(float(s_cap.mean()), 3),
        "miss_mean_per_token": round(float(misses.mean()), 1),
        "miss_p50_p90_p99_max": [int(np.percentile(misses, q))
                                 for q in (50, 90, 99)] + [int(misses.max())],
        "ssdMB_mean": round(float(mb.mean()), 1),
        "ssdMB_p99": round(float(np.percentile(mb, 99)), 1),
        "ssdMB_max": round(float(mb.max()), 1),
        "burst_p99_over_mean": round(float(np.percentile(mb, 99) / mb.mean()), 2),
        "tokens_over_100MB": int((mb > 100).sum()),
        "tokens_over_120MB": int((mb > 120).sum()),
        "min_misses_token": int(misses.min()),
    }
    print("miss dyn:", json.dumps(res["lru_miss_dynamics"]), flush=True)
    np.save(ROOT / f"results/analysis_t6_misses_cap{a.cap}.npy", misses)

    # ---- 3. mechanism headroom at cap --------------------------------
    # 3a. shipped simulator (note: its LRU hit path does NOT refresh the
    # LRU timestamp -> it is insertion-order/FIFO eviction, not true LRU;
    # see WINS 2026-10-06. Numbers kept for comparability with shipped docs.)
    zero_picks = np.zeros((T, L, 1), np.int16)
    prior_rank = {li: np.argsort(-np.bincount(bl[li].ravel(), minlength=E))
                  for li in range(L)}
    spec = dict(dram=546.0, ssd=7.4, usable=99.0)   # M4 Max tier numbers
    headroom = {}
    for mode in ("lru", "prior", "sidecar"):
        srv, sync_mb, async_mb, tp, c_ms, st_ms = sp.solve_policy(
            gold, zero_picks, prior_rank, a.cap, mode, 0, spec,
            throttle="legacy")
        headroom["shipped_" + mode] = dict(
            served=round(srv, 3), sync_MB=round(sync_mb, 0),
            async_MB=round(async_mb, 0), tps=round(tp, 1),
            compute_ms=round(c_ms, 1), stream_ms=round(st_ms, 1))
        print("shipped", mode, headroom["shipped_" + mode], flush=True)

    # 3b. true-semantics replay (positional OrderedDict LRU with hit
    # refresh; same capacity, same pin rules). This is what oMLX's LRU
    # eviction actually does; the sweep above already matches it exactly.
    def simulate_true(mode):
        n_pin = a.cap // 2 if mode == "prior" else 0
        caches = []
        pins = []
        for li in range(L):
            c = OrderedDict()
            if mode == "prior":
                for e in prior_rank[li][:n_pin]:
                    c[int(e)] = 1
            pins.append(set(c))
            caches.append(c)
        srv = async_ = 0
        for t in range(T):
            for li in range(L):
                c = caches[li]
                pin = pins[li]
                for e in gold[t, li]:
                    e = int(e)
                    if e in c:
                        srv += 1
                        if e not in pin:
                            c.move_to_end(e)
                    else:
                        c[e] = 1
                        async_ += 0     # miss, charged sync below
                        while len(c) - len(pin & c.keys()) > a.cap - len(pin):
                            for kx in c.keys():
                                if kx not in pin:
                                    del c[kx]
                                    break
            if mode == "sidecar" and t + 1 < T:
                newpin = [set() for _ in range(L)]
                for li in range(L):
                    c = caches[li]
                    for e in gold[t + 1, li]:
                        e = int(e)
                        newpin[li].add(e)
                        if e not in c:
                            c[e] = 1
                            async_ += 1
                    while len(c) - len(newpin[li] & c.keys()) > a.cap - K:
                        for kx in c.keys():
                            if kx not in newpin[li]:
                                del c[kx]
                                break
                    pins[li] = newpin[li]
        n = T * L * K
        sync_mb = (n - srv) * sp.EXPERT_MIB / T
        async_mb = async_ * sp.EXPERT_MIB / T
        c_ms = ((sp.READ_FLOOR_MIB + sp.GOLD_MIB + async_mb) / 1024.0
                / (spec["dram"] * sp.DRAM_EFF) * 1000.0)
        st_ms = ((sync_mb + async_mb + sp.PLE_STREAM_MIB) / 1024.0
                 / spec["ssd"] * 1000.0)
        return dict(served=round(srv / n, 3), sync_MB=round(sync_mb, 0),
                    async_MB=round(async_mb, 0),
                    tps=round(1000.0 / max(c_ms, st_ms), 1),
                    compute_ms=round(c_ms, 1), stream_ms=round(st_ms, 1))

    for mode in ("lru", "prior", "sidecar"):
        headroom["true_" + mode] = simulate_true(mode)
        print("true", mode, headroom["true_" + mode], flush=True)
    res["headroom_at_cap"] = headroom
    res["sim_note"] = ("shipped_* uses sp.solve_policy (FIFO eviction bug: "
                       "hit path never refreshes the LRU timestamp); "
                       "true_* replays positional OrderedDict LRU with hit "
                       "refresh — the semantics oMLX eviction uses. "
                       "true_sidecar = gold oracle, unthrottled async.")

    # 3c. static-pinning split scan (pin fractions, NOT a hyperparam
    # grid — it tests whether ANY pin fraction beats pure LRU at cap)
    def true_prior(pin_n):
        caches = []
        pins = []
        for li in range(L):
            c = OrderedDict()
            for e in prior_rank[li][:pin_n]:
                c[int(e)] = 1
            pins.append(set(c))
            caches.append(c)
        srv = 0
        for t in range(T):
            for li in range(L):
                c = caches[li]
                pin = pins[li]
                npin = len(pin)
                dyn = a.cap - npin
                for e in gold[t, li]:
                    e = int(e)
                    if e in c:
                        srv += 1
                        if e not in pin:
                            c.move_to_end(e)
                    else:
                        c[e] = 1
                        while len(c) - npin > dyn:
                            for kx in c.keys():
                                if kx not in pin:
                                    del c[kx]
                                    break
        return round(srv / (T * L * K), 4)

    res["pin_scan_at_cap"] = {f"pin_{p}": true_prior(int(a.cap * p))
                              for p in (0.0, 0.125, 0.25, 0.5)}
    print("pin scan:", res["pin_scan_at_cap"], flush=True)

    # cross-check: analytic sweep vs both LRU implementations at cap.
    res["cross_check"] = {
        "sweep_analytic_served_cap": res["lru_miss_dynamics"]["residency_served_analytic"],
        "true_lru_served_cap": headroom["true_lru"]["served"],
        "shipped_fifo_lru_served_cap": headroom["shipped_lru"]["served"],
    }

    lru_h, pri_h, side_h = (headroom["true_lru"], headroom["true_prior"],
                            headroom["true_sidecar"])
    res["headroom_decomposition"] = {
        "basis": "true_* (hit-refresh LRU)",
        "lru_served": lru_h["served"],
        "pinning_gain_pp": round(100 * (pri_h["served"] - lru_h["served"]), 2),
        "perfect_pred_gain_pp": round(100 * (side_h["served"] - lru_h["served"]), 2),
        "residual_miss_frac_LRU": round(1 - lru_h["served"], 3),
        "shipped_sim_undercount_pp": round(
            100 * (headroom["true_lru"]["served"]
                   - headroom["shipped_lru"]["served"]), 2),
        "note": ("perfect_pred_gain_pp = upper bound on ANY next-token "
                 "prefetcher's served-fraction gain at this cap; "
                 "pinning_gain_pp = gain of a static hot-set costing zero "
                 "prefetch bandwidth. SIMULATED, synth traces, cap=143."),
    }

    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
