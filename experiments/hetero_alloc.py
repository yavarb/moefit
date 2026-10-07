"""Experiment: heterogeneous per-layer expert capacity vs the uniform cap.

Hypothesis: layers differ in how concentrated their routing is
(results/ple_probe.json: top-14 prior coverage 0.11 .. 0.26 per layer;
results/trigger_eval.json: previous-token persistence 0.06 .. 0.51), so
giving concentrated layers fewer resident experts and diffuse layers more
should raise the served fraction at the same total RAM.

Method: measure each layer's LRU hit curve on the BUILD split, split the
same total budget 48*C across layers greedily (step 4, each layer at
least MIN_LAYER_CAP so pinned slots never exceed capacity), then replay
the HOLDOUT split with experiments/sim_paging.simulate() under per-layer
caps. An oracle allocation fitted on the holdout itself bounds what any
allocator could gain.

Result on the calibrated synthetic trace: no gain (see REVIEW.md). The
allocator is kept here, outside the shipped simulator, so the negative
result stays reproducible:
    .venv/bin/python experiments/hetero_alloc.py --traces-dir results/traces_synth
"""
import argparse, json, sys
from collections import OrderedDict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import sim_paging as sp  # noqa: E402

L, K, E = sp.L, sp.K, sp.E
MIN_LAYER_CAP = 16          # >= largest pin budget (12) + 4 dynamic slots
CAP_GRID = (4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256,
            320, 384, 448, 512)


def lru_hit_curve(rows, grid=CAP_GRID):
    """hits at each cap for a plain per-access LRU over rows [T,K]."""
    out = []
    for c in grid:
        od = OrderedDict()
        hits = 0
        for row in rows:
            for e in row:
                e = int(e)
                if e in od:
                    hits += 1
                    od.move_to_end(e)
                else:
                    od[e] = None
                    if len(od) > c:
                        od.popitem(last=False)
        out.append(hits)
    return out


def allocate_caps(lay, total, grid=CAP_GRID, step=4,
                  min_cap=MIN_LAYER_CAP, max_cap=E):
    """Greedy: give `step` experts at a time to the layer whose LRU hit
    curve (linearly interpolated between grid points) gains most."""
    curves = {li: lru_hit_curve(lay[li], grid) for li in range(L)}

    def hits(li, c):
        return float(np.interp(c, grid, curves[li]))

    caps = [min_cap] * L
    budget = total - sum(caps)
    assert budget >= 0, "total budget below 48 * MIN_LAYER_CAP"
    while budget >= step:
        best, best_gain = None, -1.0
        for li in range(L):
            if caps[li] + step > max_cap:
                continue
            g = hits(li, caps[li] + step) - hits(li, caps[li])
            if g > best_gain:
                best, best_gain = li, g
        if best is None:
            break
        caps[best] += step
        budget -= step
    return caps, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-dir", default=str(ROOT / "results/traces_synth"))
    ap.add_argument("--caps", default="32,64,128,192")
    ap.add_argument("--modes", default="lru,prior,sidecar")
    ap.add_argument("--tier", default="32GB-M4P")
    ap.add_argument("--out", default=str(ROOT / "results/hetero_alloc_synth.json"))
    a = ap.parse_args()
    tdir = Path(a.traces_dir)
    bf, bl, bT = sp.load_split(tdir / "build.npz")
    hf, hl, hT = sp.load_split(tdir / "holdout.npz")
    gold = np.stack([hl[li][:hT] for li in range(L)], axis=1).astype(np.int16)
    prior = {li: np.argsort(-np.bincount(bl[li].ravel(), minlength=E))
             for li in range(L)}
    spec = sp.TIERS[a.tier]
    out = []
    print(f"{'cap':>4} {'mode':8} {'alloc':14} {'served':>7} {'syncMB':>7} "
          f"{'asyncMB':>8} {'tps ' + a.tier:>12} {'caps':>9}")
    for cap in [int(c) for c in a.caps.split(",")]:
        allocs = {
            "uniform": [cap] * L,
            "hetero-build": allocate_caps({li: bl[li][:bT] for li in range(L)},
                                          cap * L)[0],
            "oracle-holdout": allocate_caps({li: hl[li][:hT] for li in range(L)},
                                            cap * L)[0],
        }
        for mode in a.modes.split(","):
            for name, caps in allocs.items():
                assert sum(caps) == cap * L
                srv, sync_mb, async_mb, tps, c_ms, st_ms = sp.solve_policy(
                    gold, None, prior, caps, mode, 0, spec)
                row = dict(cap=cap, mode=mode, alloc=name, served=round(srv, 4),
                           sync_mb=round(sync_mb, 1), async_mb=round(async_mb, 1),
                           tps=round(tps, 2), caps_min=min(caps),
                           caps_max=max(caps), tier=a.tier)
                out.append(row)
                print(f"{cap:>4} {mode:8} {name:14} {srv:7.4f} {sync_mb:7.1f} "
                      f"{async_mb:8.1f} {tps:12.2f} {min(caps):>4}..{max(caps):<4}",
                      flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
