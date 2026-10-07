"""T7: how much of the 2.8x sim->silicon gap is trace uncertainty vs
hardware-knob uncertainty vs genuinely missing physics?

Two questions, both local, no silicon runs:

 (1) TRACE BAND. The sim was run on a synthetic trace (results/traces_synth)
     that only matches shipped aggregate statistics (prior@14, prev, and the
     LRU/prior served table at 4 caps). Different (beta, shift) generator
     configs that pass the SAME calibration gates can still disagree on
     served fraction at cap=143 -- the exact operating point of the M4 Max 36 GB 13.0 tok/s baseline. This quantifies that band so the 2.8x ratio
     gets an honest error bar instead of being a point estimate.

 (2) KNOB ATTRIBUTION. Treat silicon's 76.9 ms/token as ground truth and
     ask which combinations of (effective DRAM BW, effective SSD random-read
     BW, PLE MB/tok, per-miss latency) can reproduce it, given the sim's
     byte accounting at cap=143. Output: posterior-ish marginals over knobs
     (accepted by rejection sampling on a prior of *plausible* hardware
     values) and the discriminating observable each regime predicts.

Usage:
  python3 experiments/t7_gap_uncertainty.py \
      --trace-dirs results/traces_synth,/tmp/.../b1.5_s0.3_seed0,... \
      --out results/t7_gap_uncertainty.json

All numbers here are SIMULATED except the 13.0 tok/s / 76.92 ms/tok and
22.61 GB anchors from results/measured_m4max_36gb.json.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import sim_paging as sp

# --- silicon anchors (measured, results/measured_m4max_36gb.json) ---
MEAS_TPS = 13.0
MEAS_MS = 1000.0 / MEAS_TPS           # 76.92 ms/token
MEAS_CAP = 143                        # 0.28 residency x 512 experts/layer
TIER = "48GB-M4M"                     # usable 33 GiB covers the 22.6 GiB footprint


def served_sync(traces_dir, cap=MEAS_CAP, mode="lru", eval_sub=8000):
    """served fraction + sync/async MB per token, shipped solve_policy path.
    Also computes mean |delta| vs the shipped served table (the same gate
    experiments/synth_validate.py uses to accept a synthetic trace)."""
    tdir = Path(traces_dir)
    bf, bl, bT = sp.load_split(tdir / "build.npz")
    hf, hl, hT = sp.load_split(tdir / "holdout.npz")
    rng = np.random.default_rng(0)
    hs = np.sort(rng.choice(hT, size=min(eval_sub, hT), replace=False))
    gold = np.stack([hl[li][hs] for li in range(sp.L)], axis=1).astype(np.int16)
    prior_rank = {li: np.argsort(-np.bincount(bl[li].ravel(), minlength=sp.E))
                  for li in range(sp.L)}
    spec = sp.TIERS[TIER]
    srv, sync_mb, async_mb, t, c_ms, st_ms = sp.solve_policy(
        gold, None, prior_rank, cap, mode, 0, spec, throttle="legacy")
    # shipped served table (probe_picks=6 rows are the ones synth_validate used)
    ship = {(r["cap"], r["mode"]): r
            for r in json.loads((ROOT / "results/sim_paging.json").read_text())
            if r["probe_picks"] == 6}
    errs = []
    for gcap in (32, 64, 128, 192):
        for gmode in ("lru", "prior"):
            s2, _, _ = sp.simulate(gold, None, prior_rank, gcap, gmode, 0,
                                   10 ** 9)
            key = (gcap, gmode)
            if key in ship:
                sv = ship[key].get(f"served_{TIER}")
                if sv is not None:
                    errs.append(abs(s2 - sv))
    gate = float(np.mean(errs)) if errs else float("nan")
    cal = {}
    cp = tdir / "calibration.json"
    if cp.exists():
        c = json.loads(cp.read_text())
        cal = dict(beta=c["config"]["beta"], shift=c["config"]["shift"],
                   seed=c["config"]["seed"],
                   prior14_fit=c["overall"]["prior14_fit"],
                   prev_fit=c["overall"]["prev_fit"])
    return dict(served=round(srv, 4), sync_mb=round(sync_mb, 1),
                async_mb=round(async_mb, 1), sim_tps=round(t, 1),
                sim_ct_ms=round(c_ms, 1), sim_st_ms=round(st_ms, 1),
                gate_mean_abs_delta=round(gate, 4), **cal)


def knob_attribution(sync_mb, async_mb, n=200000, seed=7):
    """Rejection-sample hardware knobs that reproduce MEAS_MS.

    Timing model (superset of sim roofline):
      compute_ms = (FLOOR + GOLD + async)/DRAM_eff
      stream_ms  = (sync + async + PLE)/SSD_eff
      miss_ms    = n_miss * miss_latency        (per-miss fixed overhead)
      tok_ms     = max(compute, stream) + miss_ms
    Accepted if |tok_ms - MEAS_MS| <= 3 ms (bench noise band).
    Priors:
      DRAM_eff GB/s: uniform 50..293   (293 = shipped calibration, 54% peak)
      SSD_eff  GB/s: uniform 1.0..7.4  (7.4 = sequential spec)
      PLE MB/tok:    log-uniform 0.3..400 (0.3 = sim charge)
      miss_latency ms/expert: log-uniform 0.005..1.0 (~66 misses/tok at cap143)
    """
    rng = np.random.default_rng(seed)
    dram = rng.uniform(50.0, 293.0, n)
    ssd = rng.uniform(1.0, 7.4, n)
    ple = np.exp(rng.uniform(np.log(0.3), np.log(400.0), n))
    lat = np.exp(rng.uniform(np.log(0.005), np.log(1.0), n))
    nmiss = MISS_PER_TOK  # sync misses per token (computed from sync MB)
    dram_mb = sp.READ_FLOOR_MIB + sp.GOLD_MIB + async_mb
    c_ms = dram_mb / 1024.0 / dram * 1000.0
    st_ms = (sync_mb + async_mb + ple) / 1024.0 / ssd * 1000.0
    tok = np.maximum(c_ms, st_ms) + nmiss * lat
    acc = np.abs(tok - MEAS_MS) <= 3.0
    def summ(x):
        if acc.sum() == 0:
            return None
        a = x[acc]
        return dict(p10=round(float(np.percentile(a, 10)), 3),
                    med=round(float(np.median(a)), 3),
                    p90=round(float(np.percentile(a, 90)), 3))
    def frac_regime(mask):
        if acc.sum() == 0:
            return None
        return round(float((mask & acc).sum() / acc.sum()), 3)
    return dict(
        n_accepted=int(acc.sum()), n_draws=n,
        dram_gbps=summ(dram), ssd_gbps=summ(ssd),
        ple_mb=summ(ple), miss_lat_ms=summ(lat),
        regime_frac=dict(
            stream_bound=frac_regime(st_ms >= c_ms),
            compute_bound=frac_regime(c_ms > st_ms),
            miss_dominated=frac_regime(nmiss * lat > 0.5 * tok)),
        # what pure-SSD single-knob calibration would need (no miss term)
        ssd_only_gbps=round((sync_mb + async_mb + 0.3) / 1024.0
                            / MEAS_MS * 1000.0, 2),
    )


def residency_predictor(caps=(71, 143, 170), traces_dir=str(ROOT / "results/traces_synth")):
    """SIMULATED predictions at alternate residencies for the two rival
    gap hypotheses anchored at M4 Max 36 GB cap=143 -> 13.0 tok/s:
      h1_flat_overhead: keep sim's spec roofline, add a constant
          49.6 ms/tok runtime overhead (the T4 unexplained term as fixed
          cost: scheduling, page-in latency independent of bytes).
      h2_ssd_random26: single-knob calibration, effective SSD random-read
          BW = ssd_only_gbps (T3), no extra overhead term.
    The two hypotheses agree at cap=143 by construction and DIVERGE at
    other residencies -- one cheap bench point at 0.14 residency
    (cap~71) picks between them: h1 ~9.8 vs h2 ~6.7 tok/s.
    """
    tdir = Path(traces_dir)
    bf, bl, bT = sp.load_split(tdir / "build.npz")
    hf, hl, hT = sp.load_split(tdir / "holdout.npz")
    rng = np.random.default_rng(0)
    hs = np.sort(rng.choice(hT, size=min(8000, hT), replace=False))
    gold = np.stack([hl[li][hs] for li in range(sp.L)], axis=1).astype(np.int16)
    prior_rank = {li: np.argsort(-np.bincount(bl[li].ravel(), minlength=sp.E))
                  for li in range(sp.L)}
    spec = sp.TIERS[TIER]
    tab = {}
    for cap in caps:
        srv, sync, async_ = sp.simulate(gold, None, prior_rank, cap, "lru",
                                        0, 10 ** 9)
        c_ms = (sp.READ_FLOOR_MIB + sp.GOLD_MIB + async_) / 1024.0 \
            / (spec["dram"] * sp.DRAM_EFF) * 1000.0
        st_spec = (sync + async_ + sp.PLE_STREAM_MIB) / 1024.0 \
            / spec["ssd"] * 1000.0
        ssd26 = (sync + async_ + sp.PLE_STREAM_MIB) \
            / 1024.0 / SSD_RAND_GBPS * 1000.0
        tab[cap] = dict(
            served=round(srv, 4), sync_mb=round(sync, 1),
            footprint_gib=round(sp.FLOOR_MIB / 1024 + cap * sp.L
                                * sp.EXPERT_MIB / 1024, 1),
            h1_flat_overhead_tps=round(1000 / (max(c_ms, st_spec)
                                               + FLAT_OVERHEAD_MS), 1),
            h2_ssd_random_tps=round(1000 / max(c_ms, ssd26), 1))
    return tab


FLAT_OVERHEAD_MS = 49.6   # T4 unexplained term at cap=143 (measured - sim stream term)
SSD_RAND_GBPS = 2.6       # T3 one-knob reconciliation value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dirs", required=True,
                    help="comma-separated dirs with build.npz/holdout.npz")
    ap.add_argument("--cap", type=int, default=MEAS_CAP)
    ap.add_argument("--out", default=str(ROOT / "results/t7_gap_uncertainty.json"))
    a = ap.parse_args()

    global MISS_PER_TOK
    rows = []
    for td in a.trace_dirs.split(","):
        r = served_sync(td.strip(), cap=a.cap)
        r["traces"] = str(td)
        rows.append(r)
        print(json.dumps(r), flush=True)

    syncs = np.array([r["sync_mb"] for r in rows])
    srvs = np.array([r["served"] for r in rows])
    tpss = np.array([r["sim_tps"] for r in rows])
    gates = np.array([r["gate_mean_abs_delta"] for r in rows])
    band = dict(
        served_min=float(srvs.min()), served_max=float(srvs.max()),
        sim_tps_min=float(tpss.min()), sim_tps_max=float(tpss.max()),
        ratio_min=float(MEAS_TPS / tpss.max()), ratio_max=float(MEAS_TPS / tpss.min()),
    )
    # gate: shipped served-table mean |delta| <= 0.03 (synth_validate-style)
    GATE = 0.03
    keep = gates <= GATE
    if keep.any():
        band[f"gated_{GATE:g}"] = dict(
            n_traces=int(keep.sum()),
            served_min=float(srvs[keep].min()), served_max=float(srvs[keep].max()),
            sim_tps_min=float(tpss[keep].min()), sim_tps_max=float(tpss[keep].max()),
            ratio_min=float(MEAS_TPS / tpss[keep].max()),
            ratio_max=float(MEAS_TPS / tpss[keep].min()),
            sim_tps_median=float(np.median(tpss[keep])),
            ratio_at_median_tps=round(MEAS_TPS / float(np.median(tpss[keep])), 2))

    # knob attribution at the mid-trace operating point
    med_sync = float(np.median(syncs))
    med_async = float(np.median([r["async_mb"] for r in rows]))
    MISS_PER_TOK = med_sync / sp.EXPERT_MIB
    attr = knob_attribution(med_sync, med_async)
    attr["sync_mb"] = round(med_sync, 1)
    attr["async_mb"] = round(med_async, 1)
    attr["miss_per_tok"] = round(MISS_PER_TOK, 1)

    pred = residency_predictor(traces_dir=a.trace_dirs.split(",")[0].strip())
    out = dict(kind="simulated + measured anchors", cap=a.cap, tier=TIER,
               measured=dict(tps=MEAS_TPS, ms_per_tok=round(MEAS_MS, 2)),
               per_trace=rows, trace_band=band, knob_attribution=attr,
               residency_predictor=pred)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


MISS_PER_TOK = 77.0
if __name__ == "__main__":
    main()
