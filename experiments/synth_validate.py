"""Compare LRU / pinned-hot-set / sidecar served fractions on a synthetic
trace with the shipped table (results/sim_paging.json). Used to choose the
synthetic generator's global knobs (beta, shift) and to state how far the
synthetic trace is from the real one.

usage: .venv/bin/python experiments/synth_validate.py [--traces-dir ...]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import sim_paging as sp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-dir", default=str(ROOT / "results/traces_synth"))
    ap.add_argument("--caps", default="32,64,128,192")
    ap.add_argument("--modes", default="lru,prior")
    a = ap.parse_args()
    tdir = Path(a.traces_dir)
    bf, bl, bT = sp.load_split(tdir / "build.npz")
    hf, hl, hT = sp.load_split(tdir / "holdout.npz")
    gold = np.stack([hl[li][:hT] for li in range(sp.L)], axis=1).astype(np.int16)
    prior_rank = {li: np.argsort(-np.bincount(bl[li].ravel(), minlength=sp.E))
                  for li in range(sp.L)}
    shipped = json.load(open(ROOT / "results/sim_paging.json"))
    ship = {(r["cap"], r["mode"]): r for r in shipped if r["probe_picks"] == 6}
    tier = "32GB-M4P" if "32GB-M4P" in sp.TIERS else None
    err = []
    print(f"{'cap':>4} {'mode':8} {'synthetic':>9} {'shipped':>8} {'delta':>7}")
    for cap in [int(c) for c in a.caps.split(",")]:
        for mode in a.modes.split(","):
            srv, sync_mb, async_mb = sp.simulate(
                gold, None, prior_rank, cap, mode, 0, 10 ** 9)
            key = (cap, mode)
            s_served = ship[key].get(f"served_{tier}") if key in ship else None
            d = srv - s_served if s_served is not None else float("nan")
            err.append(abs(d))
            print(f"{cap:>4} {mode:8} {srv:9.3f} {s_served if s_served is not None else float('nan'):8.3f} {d:+7.3f}")
    print(f"mean |delta| served = {np.nanmean(err):.3f}")


if __name__ == "__main__":
    main()
