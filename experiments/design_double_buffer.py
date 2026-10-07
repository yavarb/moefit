"""T4 INVENTION: within-layer double-buffer / staged expert install.

Mechanism (SIM under MEASURED serial constants — no silicon here):
oMLX 0.7.0 resolves expert misses serially per layer-step:
  exposed/step = A + B*k (fetch) + install*k (GPU-side install, on the
  critical path) + sync.
The double-buffer design stages fetched experts in a shadow buffer
while a separate install stream drains it: install of expert i
overlaps the fetch of expert i+1 (classic reader/writer pipeline).
Exposed/step becomes:
  A + B*k (fetch-bound when B >= install) + install * [k>0]  (ONE
  trailing install latency) + sync.
i.e. the per-miss install cost k*install collapses to a single install
latency per missing-step.

This is WITHIN-layer only — deliberately disjoint from T1's cross-layer
fetch pipeline and T2's sidecar prefetch (which remove the A+B*k term);
the three compose.

GPU idle fraction is measured on the simulated per-token timeline:
idle = (total - compute)/total, the fraction of wall time the GPU is
blocked on fetch/install/sync (lead_silicon measured GPU util median
42% on silicon, i.e. idle ~58% — the sim's baseline idle fraction is
its prediction of that quantity).

usage:
  python3 experiments/design_double_buffer.py --out results/design_double_buffer.json
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from experiments import sim_paging as sp
from moefit.metrics import (SERIAL_CONSTANTS_MEASURED as SC,
                            serial_model_from_misses)


def step_stats(M):
    """Per layer-step miss aggregation from a (T, L) miss matrix."""
    M = np.asarray(M)
    T, L = M.shape
    per_layer_miss = M.sum(axis=0) / T
    per_layer_steps = (M > 0).sum(axis=0) / T
    return dict(misses_per_tok=float(per_layer_miss.sum()),
                steps_with_miss_frac=float(per_layer_steps.mean()),
                miss_per_missing_step=float(
                    per_layer_miss.sum() / max(per_layer_steps.sum(), 1e-9)),
                per_layer_miss=per_layer_miss,
                per_layer_steps=per_layer_steps)


def serial_breakdown(st, constants, compute_ms):
    """Baseline: oMLX serial resolve (fetch + install both exposed)."""
    c = constants
    io = sum(c["io_A_ms"] * st["per_layer_steps"][li]
             + c["io_B_ms"] * st["per_layer_miss"][li]
             for li in range(len(st["per_layer_miss"])))
    inst = c["install_ms"] * st["misses_per_tok"]
    sync = c["sync_ms"] * len(st["per_layer_miss"])
    total = compute_ms + io + inst + sync
    return dict(mode="serial", compute=compute_ms, io=round(io, 2),
                install=round(inst, 2), sync=round(sync, 2),
                total_ms=round(total, 2), tps=round(1000 / total, 2),
                gpu_idle_ms=round(total - compute_ms, 2),
                gpu_idle_frac=round((total - compute_ms) / total, 3))


def double_buffer_breakdown(st, constants, compute_ms):
    """Double-buffer: install pipelines under fetch.

    Per missing-step with k misses (reader = SSD fetch A + B*i for the
    i-th expert; writer = install, inst each, starts when its expert's
    fetch completes):
      inst <= B (fetch-bound, holds for ALL measured candidates since
      B=0.52):  exposed = A + B*k + inst   (ONE trailing install)
      inst > B  (install-bound): exposed = A + B + inst*k
    vs serial A + B*k + inst*k. The install term collapses from
    inst*misses to inst*missing_steps.
    """
    c = constants
    n_layers = len(st["per_layer_miss"])
    io = sum(c["io_A_ms"] * st["per_layer_steps"][li]
             + c["io_B_ms"] * st["per_layer_miss"][li]
             for li in range(n_layers))
    n_missing_steps = st["steps_with_miss_frac"] * n_layers
    if c["install_ms"] <= c["io_B_ms"]:
        inst = c["install_ms"] * n_missing_steps
        bound = "fetch"
    else:
        # install-bound: fetch of the first expert + full install chain
        inst = c["install_ms"] * st["misses_per_tok"]
        io += -c["io_B_ms"] * st["misses_per_tok"] \
              + c["io_B_ms"] * n_missing_steps   # B*(k-1) hidden
        bound = "install"
    sync = c["sync_ms"] * n_layers
    total = compute_ms + io + inst + sync
    return dict(mode="double_buffer", compute=compute_ms, io=round(io, 2),
                install=round(inst, 2), sync=round(sync, 2),
                total_ms=round(total, 2), tps=round(1000 / total, 2),
                gpu_idle_ms=round(total - compute_ms, 2),
                gpu_idle_frac=round((total - compute_ms) / total, 3),
                pipeline_bound=bound,
                saving_vs_serial_install_ms=None)


def compare(st, constants, compute_ms, label):
    base = serial_breakdown(st, constants, compute_ms)
    db = double_buffer_breakdown(st, constants, compute_ms)
    db["saving_vs_serial_install_ms"] = round(
        base["install"] - db["install"], 2)
    db["tps_gain"] = round(db["tps"] - base["tps"], 2)
    return dict(window=label, compute_ms_assumed=compute_ms,
                serial=base, double_buffer=db,
                gpu_idle_frac_drop=round(
                    base["gpu_idle_frac"] - db["gpu_idle_frac"], 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=143)
    ap.add_argument("--eval-sub", type=int, default=8000)
    ap.add_argument("--cold-window", type=int, default=200,
                    help="first N holdout tokens = cold/k-large window")
    ap.add_argument("--traces-dir",
                    default=str(ROOT / "results/traces_synth"))
    ap.add_argument("--out", default=str(ROOT / "results/design_double_buffer.json"))
    a = ap.parse_args()

    tdir = Path(a.traces_dir)
    bf, bl, bT = sp.load_split(tdir / "build.npz")
    hf, hl, hT = sp.load_split(tdir / "holdout.npz")
    rng = np.random.default_rng(0)
    hs = np.sort(rng.choice(hT, size=min(a.eval_sub, hT), replace=False))
    gold = np.stack([hl[li][hs] for li in range(sp.L)],
                    axis=1).astype(np.int16)
    prior_rank = {li: np.argsort(-np.bincount(bl[li].ravel(),
                                              minlength=sp.E))
                 for li in range(sp.L)}

    # oMLX-like policy (hit-refresh true-LRU; <1% from decayed-count on
    # synth) — miss matrix M is (T, L)
    _, _, _, M = sp.simulate(gold, None, prior_rank, a.cap, "lru", 0,
                             10 ** 9, hit_refresh=True, want_misses=True)

    steady = step_stats(M)
    cold = step_stats(M[:a.cold_window])

    # named measured constant candidates (NOT a grid): lead_silicon's
    # microbench install 0.30; T2's newer probe eval_each 0.18 and
    # batched 0.28. Compute: 18.1 (546-bin) / 24.1 (410-bin, measured
    # chip; best-supported per T3).
    install_candidates = {
        "microbench_0.30": dict(SC, install_ms=0.30),
        "t2probe_eval_each_0.18": dict(SC, install_ms=0.18),
        "t2probe_batched_0.28": dict(SC, install_ms=0.28),
    }
    out = dict(
        kind="simulated", design="within-layer double-buffer / staged install",
        policy="lru hit_refresh=true (omlx-like)",
        cap=a.cap, traces_dir=str(tdir), eval_sub=len(hs),
        steady_window=dict(misses_per_tok=round(steady["misses_per_tok"], 2),
                           steps_with_miss_frac=round(
                               steady["steps_with_miss_frac"], 3),
                           miss_per_missing_step=round(
                               steady["miss_per_missing_step"], 2)),
        cold_window=dict(n=a.cold_window,
                         misses_per_tok=round(cold["misses_per_tok"], 2),
                         steps_with_miss_frac=round(
                             cold["steps_with_miss_frac"], 3),
                         miss_per_missing_step=round(
                             cold["miss_per_missing_step"], 2)),
        comparisons=[], notes=[
            "SIM under MEASURED serial constants (A/B/sync microbench; "
            "install candidates from lead_silicon + T2 silicon probe). "
            "No silicon run of this design exists.",
            "Double-buffer collapses install*k -> ONE install latency "
            "per missing-step (fetch-bound since B=0.52 >= install).",
            "Within-layer only; composes with T1 cross-layer fetch "
            "(which removes the A+B*k term) and T2 sidecar prefetch.",
        ])
    for iname, iconst in install_candidates.items():
        for comp in (24.1, 18.1):
            out["comparisons"].append(dict(
                install_constant=iname, **compare(
                    steady, iconst, comp, "steady")))
            out["comparisons"][-1]["cold_window"] = compare(
                cold, iconst, comp, "cold")
    # headline rows for the ledger-friendly summary
    out["headline"] = {}
    for iname in install_candidates:
        row = next(c for c in out["comparisons"]
                   if c["install_constant"] == iname
                   and c["compute_ms_assumed"] == 24.1)
        out["headline"][iname] = dict(
            steady_serial_tps=row["serial"]["tps"],
            steady_db_tps=row["double_buffer"]["tps"],
            steady_tps_gain=row["double_buffer"]["tps_gain"],
            steady_gpu_idle_serial=row["serial"]["gpu_idle_frac"],
            steady_gpu_idle_db=row["double_buffer"]["gpu_idle_frac"],
            cold_serial_tps=row["cold_window"]["serial"]["tps"],
            cold_db_tps=row["cold_window"]["double_buffer"]["tps"],
            cold_tps_gain=row["cold_window"]["double_buffer"]["tps_gain"])
    print(json.dumps(out, indent=1))
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", a.out, file=sys.stderr)


if __name__ == "__main__":
    main()
