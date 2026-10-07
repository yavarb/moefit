"""T6 cycle 2: prediction quality (probe vs oracle) + IO coalescing headroom.

Two levers the lab has ranked but not yet measured on TRAFFIC:
  (a) prediction — T8 showed oracle prefetch wins but a coact predictor is
      net-negative at install=0.30 ms. Missing number: what does the actual
      PLE-features probe (ridge regression, now numerically fixed to f64)
      achieve? precision@P, caught-misses per token, priced under lead's
      serial model (caught miss = -0.82 ms, speculative install = +0.30 ms).
  (b) read coalescing / layout — glm_writeups' design implication: raising
      the 1.79 GB/s duty cycle beats hit-rate tuning. Physical IO is
      171 KB x ~16 reads per expert (scattered pages). If experts are
      PACKED CO-LOCATED (cluster layout), reads for co-missed experts in a
      layer-step share IOs; the serial per-layer step cost A=0.20 ms is
      paid per IO BATCH not per expert. Question answered here: what
      FRACTION of layer-steps has >=2 misses, and does the co-visit graph
      give clusters that cover co-misses? That bounds the coalescing win.

Serial model per layer-step: io = A + B*k (k misses in that layer this
token). If k misses collapse to g IO groups, io = A + B*k stays (bytes
same) BUT A is paid g times, not once: saving = A*(g-1) per step. Plus
duty-cycle: 171 KB page reads that hit adjacent packed experts merge into
sequential reads at ~3x the 1.79 GB/s duty rate.

SIMULATED on locked synth traces (dir results/traces_synth, seed 0).
Latency constants are MEASURED (lead_silicon 463c856). No silicon run here.

usage: python3 experiments/analysis_t6_predict_coalesce.py
writes results/analysis_t6_predict_coalesce.json
"""
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import sim_paging as sp  # noqa: E402

CAP = 143
L, K, E = sp.L, sp.K, sp.E
WARM = 200
# measured constants (M4 Max 36 GB, lead_silicon 463c856)
IO_A = 0.20
IO_B = 0.52
INSTALL = 0.30
SYNC = 0.122
COMPUTE = 18.1


def true_lru_miss_matrix(gold, cap, warm=WARM):
    """Per-token per-layer sync misses under positional LRU (hit refresh).
    Returns (T-warm, L) int8 matrix + served fraction."""
    T = gold.shape[0]
    M = np.zeros((T, L), np.int16)
    caches = [OrderedDict() for _ in range(L)]
    srv = 0
    for t in range(T):
        for li in range(L):
            c = caches[li]
            for e in gold[t, li]:
                e = int(e)
                if e in c:
                    srv += 1
                    c.move_to_end(e)
                else:
                    c[e] = 1
                    if len(c) > cap:
                        c.popitem(last=False)
                    M[t, li] += 1
    n = T * L * K
    return M[warm:], srv / n


def main():
    bf, bl, bT = sp.load_split(ROOT / "results/traces_synth/build.npz")
    hf, hl, hT = sp.load_split(ROOT / "results/traces_synth/holdout.npz")
    rng = np.random.default_rng(0)
    hs = np.sort(rng.choice(hT, size=min(8000, hT), replace=False))
    gold = np.stack([hl[li][hs] for li in range(L)], axis=1).astype(np.int16)
    T = gold.shape[0]
    res = {"kind": "SIMULATED", "traces": "results/traces_synth seed0",
           "cap": CAP, "eval_tokens": int(T),
           "constants": "SERIAL measured (A .20/B .52/install .30/sync .122), compute 18.1 ASSUMED"}
    print(f"eval tokens={T}", flush=True)

    # ---- baseline miss matrix (true LRU) ------------------------------
    t0 = time.time()
    M, served = true_lru_miss_matrix(gold, CAP)
    Tw = M.shape[0]
    res["baseline"] = {
        "served": round(served, 4),
        "miss_per_tok": round(float(M.sum()) / Tw, 1),
        "layer_steps_dist": {str(k): int((M == k).sum()) for k in range(0, 6)},
        "frac_steps_ge2miss": round(float((M >= 2).mean()), 3),
        "frac_steps_ge1miss": round(float((M >= 1).mean()), 3),
        "runtime_s": round(time.time() - t0, 1),
    }
    tot = COMPUTE + (IO_A * (M > 0).sum() + IO_B * M.sum()) / Tw \
        + INSTALL * M.sum() / Tw + SYNC * L
    res["baseline"]["serial_ms_tok"] = round(float(tot), 1)
    res["baseline"]["serial_tps"] = round(1000.0 / float(tot), 2)
    print("baseline:", res["baseline"], flush=True)

    # ---- (a) probe prediction quality ---------------------------------
    t0 = time.time()
    probe = {}
    for P in (6, 12):
        picks = sp.probe_picks(bf.astype(np.float64), bl, hf, hs, P)
        finite = bool(np.isfinite(picks).all())
        # exact accounting pass: prefetch t's picks (legal lookahead: t-1
        # emitted), then consume token t.
        caches = [OrderedDict() for _ in range(L)]
        caught = wasted = n_pick = gold_miss = 0
        M2 = np.zeros((T, L), np.int16)      # sync-miss matrix WITH prefetch
        for t in range(T):
            # prefetch t's probe picks (known after t-1 emitted); async
            # arrivals assumed hidden (T8 pipelined convention)
            newins = [set() for _ in range(L)]
            if t > 0:
                for li in range(L):
                    c = caches[li]
                    ni = newins[li]
                    for e in picks[t, li]:
                        e = int(e)
                        n_pick += 1
                        if e in c:
                            wasted += 1        # already resident, no gain
                        else:
                            c[e] = 1
                            ni.add(e)          # install paid async
                            if len(c) > CAP:
                                c.popitem(last=False)   # collateral eviction
            # consume token
            for li in range(L):
                c = caches[li]
                ni = newins[li]
                for e in gold[t, li]:
                    e = int(e)
                    if e in c:
                        c.move_to_end(e)
                        if e in ni:
                            caught += 1        # prefetched AND used
                    else:
                        c[e] = 1
                        if len(c) > CAP:
                            c.popitem(last=False)
                        gold_miss += 1
                        M2[t, li] += 1
        prec = caught / max(n_pick, 1)
        # serial-model price of the prefetch-augmented run: misses that
        # survive pay full serial cost; async installs hidden. Picks that
        # were evicted before use = pure loss (their collateral eviction
        # shows up in M2).
        ms2 = sp.serial_ms(M2[WARM:], COMPUTE)
        new_ms = float(ms2.mean())
        # naive price for cross-check: -0.82/caught, +0.30/unused-new-pick
        d_ms = -0.82 * caught / Tw + 0.30 * (n_pick - wasted - caught) / Tw
        probe[P] = dict(finite=finite, precision=round(prec, 4),
                        picks_per_tok=int(n_pick / Tw),
                        caught_per_tok=round(caught / Tw, 1),
                        wasted_resident_per_tok=round(wasted / Tw, 1),
                        gold_miss_per_tok_now=round(gold_miss / Tw, 1),
                        serial_ms_tok=round(new_ms, 1),
                        serial_tps=round(1000.0 / new_ms, 2),
                        delta_ms_tok_naive=round(float(d_ms), 2),
                        runtime_s=round(time.time() - t0, 1))
        print(f"probe P={P}:", probe[P], flush=True)
    res["probe_prediction"] = probe

    # ---- (b) co-visit coalescing bound --------------------------------
    # k-distribution already gives the ceiling: coalescing can only merge
    # misses WITHIN a layer-step (same token, same layer).
    ge2 = float((M >= 2).mean())
    # upper bound: every >=2 step merges to ONE io batch:
    save_ms = IO_A * (M >= 2).sum() / Tw     # one A saved per multi-miss step
    res["coalescing_bound"] = {
        "steps_ge2_frac": ge2,
        "A_saved_ms_per_tok_max": round(float(save_ms), 2),
        "tps_if_merged": round(1000.0 / (float(tot) - save_ms), 2),
        "note": ("A is per-IO-BATCH in a pipelined world; bound assumes ALL "
                 "co-misses in a layer-step land in one IO (perfect layout). "
                 "Byte term B*k and duty cycle unchanged; separate lever: "
                 "sequential packing raises the 1.79 GB/s duty toward the "
                 "3.8-5.6 GB/s microbench, bounded ~3x on miss bytes."),
    }
    # duty-cycle bound: bytes are (io+install portion). io_ms measured term:
    io_ms = (IO_A * (M > 0).sum() + IO_B * M.sum()) / Tw
    res["coalescing_bound"]["io_ms_tok"] = round(float(io_ms), 1)
    # if byte time halves (packing -> sequential at ~2x duty), new total:
    res["coalescing_bound"]["tps_if_duty_2x"] = round(
        1000.0 / (float(tot) - 0.5 * io_ms), 2)
    print("coalescing:", res["coalescing_bound"], flush=True)

    # co-visit structure: do co-missed experts CLUSTER? build co-visit
    # graph on BUILD traces (top-K per layer), find connected components
    # via greedy seed clustering, then measure: among steps with k>=2 on
    # holdout, what fraction of pairs fall in the same cluster?
    t0 = time.time()
    bT2 = min(len(bl[0]), 3000)
    ncl = {}
    for li in (0, 12, 24, 36, 47):
        C = np.zeros((E, E), np.int32)
        for t in range(bT2):
            row = bl[li][t]
            for i in range(K):
                a = int(row[i])
                for j in range(i + 1, K):
                    b = int(row[j])
                    C[a, b] += 1
                    C[b, a] += 1
        # greedy clusters: seed = highest-degree unclaimed, radius = top-freq partners
        deg = C.sum(1)
        claimed = np.zeros(E, bool)
        pairs_same = pairs_all = 0
        # clusters = neighbors of each seed with edge weight >= 2
        clusters = []
        for seed in np.argsort(-deg):
            if claimed[seed]:
                continue
            mem = [int(seed)] + [int(x) for x in np.nonzero(C[seed] >= 2)[0]
                                 if not claimed[x]]
            for x in mem:
                claimed[x] = True
            clusters.append(set(mem))
        # fraction of holdout co-miss pairs (consecutive LRU misses at cap)
        # in same cluster — cheap proxy: pairs within same gold row
        cidx = np.full(E, -1)
        for ci, cl in enumerate(clusters):
            for x in cl:
                cidx[x] = ci
        hrows = hl[li][hs]
        for t in range(min(T, 1500)):
            row = hrows[t]
            for i in range(K):
                for j in range(i + 1, K):
                    pairs_all += 1
                    a, b = cidx[int(row[i])], cidx[int(row[j])]
                    if a >= 0 and a == b:
                        pairs_same += 1
        ncl[li] = dict(n_clusters=len(clusters),
                       claimed_frac=round(float(claimed.mean()), 3),
                       coact_pairs_in_same_cluster=round(pairs_same / max(pairs_all, 1), 3))
        print("layer", li, ncl[li], flush=True)
    res["covisit_clusters"] = ncl
    res["covisit_clusters"]["runtime_s"] = round(time.time() - t0, 1)

    Path(ROOT / "results/analysis_t6_predict_coalesce.json").write_text(
        json.dumps(res, indent=1))
    print("wrote results/analysis_t6_predict_coalesce.json")


if __name__ == "__main__":
    main()
