"""Serial-latency design evaluator (T4 instrumentation).

Runs sim_paging.simulate with want_misses=True on LOCKED traces for one
policy config, then applies the MEASURED serial-latency constants
(moefit.metrics.SERIAL_CONSTANTS_MEASURED, from lead_silicon's Santa
Cruz microbench, results/gap_santa_cruz.json) to predict realistic
tok/s for oMLX-like serial miss resolution. This is the tool that makes
any design's miss-count reduction measurable in silicon-realistic
units, instead of the optimistic bandwidth-overlap model.

usage:
  python3 experiments/serial_predict.py --cap 143 --mode lru \
      --traces-dir results/traces_synth --out results/serial_lru143.json
  python3 experiments/serial_predict.py --cap 143 --mode prior ...
Also prints a side-by-side with the shipped bandwidth model when a tier
is given (--tier 48GB-M4M).
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from experiments import sim_paging as sp
from moefit.metrics import serial_model_from_misses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--mode", default="lru",
                    choices=("lru", "prior", "probe", "sidecar"))
    ap.add_argument("--probe-picks", type=int, default=6)
    ap.add_argument("--eval-sub", type=int, default=8000)
    ap.add_argument("--traces-dir",
                    default=str(ROOT / "results/traces_synth"))
    ap.add_argument("--tier", default="48GB-M4M",
                    help="tier for the shipped bandwidth-model side-by-side")
    ap.add_argument("--compute-ms", type=float, default=18.1,
                    help="ASSUMED compute ms/token (18.1 = DRAM_EFF@546)")
    ap.add_argument("--hit-refresh", action="store_true",
                    help="true-LRU hit refresh (oMLX 0.7.0 ExpertCache "
                         "semantics; wired into simulate() as of commit "
                         "9cd94bd). Without it the sim is FIFO-ish and "
                         "overstates misses (76.9 vs 58.3/tok at cap143).")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tdir = Path(a.traces_dir)
    bf, bl, bT = sp.load_split(tdir / "build.npz")
    hf, hl, hT = sp.load_split(tdir / "holdout.npz")
    rng = np.random.default_rng(0)
    hs = np.sort(rng.choice(hT, size=min(a.eval_sub, hT), replace=False))
    gold = np.stack([hl[li][hs] for li in range(sp.L)], axis=1
                    ).astype(np.int16)
    prior_rank = {li: np.argsort(-np.bincount(bl[li].ravel(),
                                              minlength=sp.E))
                  for li in range(sp.L)}
    picks = None
    if a.mode == "probe":
        picks = sp.probe_picks(bf.astype(np.float32), bl, hf, hs,
                               a.probe_picks)
    pick = a.probe_picks if a.mode == "probe" else 0

    srv, sync_mb, async_mb, M = sp.simulate(
        gold, picks, prior_rank, a.cap, a.mode, pick, 10 ** 9,
        hit_refresh=a.hit_refresh, want_misses=True)
    ser = serial_model_from_misses(M, compute_ms=a.compute_ms)

    out = dict(
        kind="simulated", policy=dict(mode=a.mode, cap=a.cap,
                                      probe_picks=a.probe_picks,
                                      hit_refresh=a.hit_refresh,
                                      traces_dir=str(tdir), eval_sub=len(hs)),
        served=round(srv, 3), sync_mb_per_tok=round(sync_mb, 1),
        async_mb_per_tok=round(async_mb, 1),
        serial_model=ser,
        note=("serial constants MEASURED on Santa Cruz (oMLX resolve "
              "discipline, not SSD ceiling; drive does 3.8-5.6 GB/s). "
              "compute_ms is ASSUMED. Prediction is SIMULATED."),
    )
    if a.tier and a.tier in sp.TIERS:
        _, _, _, tps, c_ms, st_ms = sp.solve_policy(
            gold, picks, prior_rank, a.cap, a.mode, pick,
            sp.TIERS[a.tier])
        out["shipped_bandwidth_model"] = dict(
            tier=a.tier, tps=tps, compute_ms=c_ms, stream_ms=st_ms)
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
        print("wrote", a.out, file=sys.stderr)


if __name__ == "__main__":
    main()
