"""T7 cycle 2: what the serial-latency model S predicts PER TOKEN, and
how the synth-trace band propagates through it.

Context: cycle-1 T7 quantified the trace band under the shipped
bandwidth model (ratio band 2.76-2.92x, knobs under-determined). Since
then lead_silicon's measured serial-latency model S (commit 463c856) is
the accepted SSD-bound time model, T3 validated it at 3 points, and T8
arbitrated TRUE-LRU as the residency policy that reconciles it. This
script supersedes t7_gap_uncertainty.residency_predictor (bandwidth
framing — retracted) and adds:

  (1) TOKEN-GAP DISTRIBUTION under S at cap=143 (true-LRU, locked synth
      trace, per-token replay, steady window): p50/p90/p95/p99 ms.
      S ties per-token time to the per-token miss burst, so it predicts
      a SPiky gap distribution that tracks the miss-count distribution;
      a byte-backlog/queue model (T8's Q) smooths bursts and predicts a
      flatter one. Means match (both ~13 tps) — the DISTRIBUTION is the
      discriminator T4's collector (tok_gap_ms p50/p90/p95/p99) can
      check on one existing bench shape, no new residency point needed.

  (2) TRACE BAND through S: steady-state tps at cap {92,143,180} under
      true-LRU x compute {18.1, 23.2, 24.2} ms for every synth config
      that passes the served-table gate (and the failing ones, flagged).
      Gives T1 an honest band (not a point) for the next silicon run.

  (3) MISS-COUNT SENSITIVITY: dtps/dmiss under S (~0.82 ms/miss) at
      each cap, with the band from trace spread — T2's traffic-side
      design target: every eviction/prefetch mechanism converts to
      tok/s through misses/tok.

All SIMULATED (synth traces) except the measured constants
(A=0.20, B=0.52, install=0.30, sync=0.122 ms; measured anchors
12.71/13.0 @cap143, 7.8 @cap92 crowded).

usage: python3 experiments/t7_serial_band.py [--trace-dirs ...]
"""
import argparse, json, sys
from collections import OrderedDict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import sim_paging as sp

# measured microbench constants (results/microbench_expert_reads_m4max_36gb.json)
IO_A_MS, IO_B_MS = 0.20, 0.52
INSTALL_MS, SYNC_MS = 0.30, 0.122
L = sp.L

# compute-term variants (T3's sensitivity set)
COMPUTE_MS = {"18.1": (sp.READ_FLOOR_MIB + sp.GOLD_MIB) / 1024.0
              / (546.0 * sp.DRAM_EFF) * 1000.0,               # 18.1 (sim 546)
              "23.2": 23.2,                                    # lead assumed
              "24.2": (sp.READ_FLOOR_MIB + sp.GOLD_MIB) / 1024.0
              / (410.0 * sp.DRAM_EFF) * 1000.0}                # 24.2 (410 bin)


def lru_miss_matrix(gold, cap, refresh=True):
    """Per-token per-layer TRUE-LRU misses. refresh=True = hit moves to
    end (oMLX ExpertCache semantics per T8 arbitration / T3 replica)."""
    caches = [OrderedDict() for _ in range(L)]
    M = np.zeros(gold.shape[:2], np.int32)
    for t in range(len(gold)):
        for li in range(L):
            c = caches[li]
            m = 0
            for e in set(int(x) for x in gold[t, li]):
                if e in c:
                    if refresh:
                        c.move_to_end(e)
                else:
                    m += 1
                    c[e] = 1
                    if len(c) > cap:
                        c.popitem(last=False)
            M[t, li] = m
    return M


def tok_ms_series(M, compute_ms):
    k = M.astype(float)
    io = np.where(k > 0, IO_A_MS + IO_B_MS * k, 0.0).sum(1)
    inst = INSTALL_MS * k.sum(1)
    return compute_ms + io + inst + L * SYNC_MS        # ms per token


def dist_stats(ms):
    return dict(mean_ms=round(float(ms.mean()), 2),
                tps=round(1000.0 / float(ms.mean()), 2),
                p50=round(float(np.percentile(ms, 50)), 1),
                p90=round(float(np.percentile(ms, 90)), 1),
                p95=round(float(np.percentile(ms, 95)), 1),
                p99=round(float(np.percentile(ms, 99)), 1),
                max=round(float(ms.max()), 1),
                cv=round(float(ms.std() / ms.mean()), 3))


def load_gold(td, eval_sub=8000, full=False):
    bf, bl, bT = sp.load_split(Path(td) / "build.npz")
    hf, hl, hT = sp.load_split(Path(td) / "holdout.npz")
    if full:
        hs = np.arange(hT)
    else:
        rng = np.random.default_rng(0)
        hs = np.sort(rng.choice(hT, size=min(eval_sub, hT), replace=False))
    gold = np.stack([hl[li][hs] for li in range(L)], axis=1).astype(np.int16)
    return gold


def gate(td):
    """mean |delta| vs shipped served table (synth_validate gate)."""
    ship = {(r["cap"], r["mode"]): r
            for r in json.loads((ROOT / "results/sim_paging.json").read_text())
            if r["probe_picks"] == 6}
    gold = load_gold(td)
    bf, bl, _ = sp.load_split(Path(td) / "build.npz")
    prior_rank = {li: np.argsort(-np.bincount(bl[li].ravel(), minlength=sp.E))
                  for li in range(L)}
    errs = []
    for gcap in (32, 64, 128, 192):
        for gmode in ("lru", "prior"):
            s2, _, _ = sp.simulate(gold, None, prior_rank, gcap, gmode, 0, 10 ** 9)
            sv = ship.get((gcap, gmode), {}).get("served_48GB-M4M")
            if sv is not None:
                errs.append(abs(s2 - sv))
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dirs", default=None,
                    help="comma list; default = locked + T7 regen set")
    ap.add_argument("--warm-frac", type=float, default=0.1)
    ap.add_argument("--out",
                    default=str(ROOT / "results/t7_serial_band.json"))
    a = ap.parse_args()

    scratch = ROOT.parent / ("homes/sheryl_analysis_b/cache/scratch/"
                             "t7_traces")
    default = ["results/traces_synth"] + \
        [str(scratch / d) for d in sorted(p.name for p in scratch.glob("b*"))]
    tds = (a.trace_dirs.split(",") if a.trace_dirs else default)

    # ---- (1) token-gap distribution, locked trace, cap143 ----
    gold = load_gold("results/traces_synth", full=True)
    warm = int(len(gold) * a.warm_frac)
    M143 = lru_miss_matrix(gold, 143)
    dist = {}
    for cname, c_ms in COMPUTE_MS.items():
        ms = tok_ms_series(M143[warm:], c_ms)
        miss = M143[warm:].sum(1)
        dist[f"comp{cname}"] = dict(
            **dist_stats(ms),
            miss_mean=float(miss.mean()),
            miss_p50=float(np.percentile(miss, 50)),
            miss_p95=float(np.percentile(miss, 95)),
            miss_max=float(miss.max()),
            corr_tps_vs_miss=round(float(np.corrcoef(ms, miss)[0, 1]), 3))
    # flat-traffic Q-style control: same mean bytes, no burstiness
    flat = np.full(len(M143) - warm, dist["comp23.2"]["mean_ms"])
    control = dict(
        S_shape_p95_over_mean=round(
            dist["comp23.2"]["p95"] / dist["comp23.2"]["mean_ms"], 3),
        Q_flat_p95_over_mean=1.0,
        note=("S keeps miss-burst shape in per-token time (corr~1); "
              "a smoothed byte-backlog (Q) flattens it. p95/mean is the "
              "checkable statistic from T4's tok_gap_ms fields."))

    # ---- (2)+(3) band through S across traces x caps ----
    CAPS = (92, 143, 180)
    per = []
    for td in tds:
        td = td.strip()
        g = gate(td)
        gold = load_gold(td, full=True)
        warm = int(len(gold) * a.warm_frac)
        row = dict(traces=td, gate_mean_abs_delta=round(g, 4),
                   gated=bool(g <= 0.03))
        for cap in CAPS:
            M = lru_miss_matrix(gold, cap)
            miss = M[warm:].sum(1)
            row[f"miss_{cap}"] = round(float(miss.mean()), 1)
            for cname, c_ms in COMPUTE_MS.items():
                ms = tok_ms_series(M[warm:], c_ms)
                row[f"tps_{cap}_comp{cname}"] = round(
                    1000.0 / float(ms.mean()), 2)
        per.append(row)
        print(json.dumps(row), flush=True)

    gated_rows = [r for r in per if r["gated"]]
    band = {}
    for cap in CAPS:
        for cname in COMPUTE_MS:
            v = [r[f"tps_{cap}_comp{cname}"] for r in gated_rows]
            band[f"cap{cap}_comp{cname}"] = dict(
                min=min(v), max=max(v),
                mid=round((min(v) + max(v)) / 2, 2))
    # sensitivity: tps gain per ONE miss/tok removed (single-layer perturb)
    sens = {}
    for cap in CAPS:
        r0 = [r for r in gated_rows][0]
        c_ms = COMPUTE_MS["23.2"]
        base = np.zeros((len(gold) - warm, L), np.int32)
        base[:, 0] = 3
        ms_a = tok_ms_series(base, c_ms)
        base2 = base.copy(); base2[:, 0] = 4
        ms_b = tok_ms_series(base2, c_ms)
        ms_per_tok_miss = float(ms_b.mean() - ms_a.mean())
        sens[f"cap{cap}"] = dict(
            ms_per_tok_miss=round(ms_per_tok_miss, 3),
            tps_per_tok_miss_at_point=round(
                r0[f"tps_{cap}_comp23.2"]
                - 1000.0 / (1000.0 / r0[f"tps_{cap}_comp23.2"]
                            + ms_per_tok_miss), 3))

    out = dict(kind="simulated (synth) + measured serial-latency constants",
               model=("S: tok = comp + sum_layers(0.20+0.52k) + 0.30*miss "
                      "+ 0.122*48 ms"),
               anchors=dict(cap143_measured="12.71-13.0",
                            cap92_measured="7.8 (crowded)"),
               distribution_cap143=dist, distribution_control=control,
               per_trace=per, gated_band=band, miss_sensitivity=sens)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
