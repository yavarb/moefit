"""sim↔silicon gap report: compare a sim_paging row against a measured run.

Usage:
  python3 experiments/gap_report.py \
      --sim results/sim_paging_matched_143.json --tier 48GB-M4M \
      --mode lru --cap 143 \
      --measured results/measured_santa_cruz_36gb.json \
      [--out results/gap_report_santacruz36_cap143.json]

Produces a canonical gap report (see moefit/metrics.py schema):
  - tok/s sim vs silicon with ratio and gap %
  - residency / footprint / expert-bytes residency alignment
  - sim roofline terms (compute vs stream ms/token)
  - reconciliation: effective DRAM GB/s silicon would need if the sim's
    compute-bound byte traffic model held.

--serial: instead of a bandwidth-model row from a results JSON, run
sim_paging.solve_policy_serial (MEASURED serial-latency constants,
hit_refresh=True = oMLX ExpertCache semantics) on the locked traces for
--cap/--mode and use that as the sim side. Use this when you want the
silicon-realistic prediction, not the optimistic overlap ceiling.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moefit.metrics import (sim_record_from_row, silicon_record_from_measured,
                            gap_report, format_gap_report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=None,
                    help="sim_paging-style results JSON (list of rows);"
                         " ignored with --serial")
    ap.add_argument("--tier", required=True, help="e.g. 48GB-M4M")
    ap.add_argument("--mode", default=None,
                    help="filter row by mode; default first matching cap")
    ap.add_argument("--cap", type=int, default=None,
                    help="filter row by cap (experts/layer)")
    ap.add_argument("--throttle", default=None)
    ap.add_argument("--serial", action="store_true",
                    help="run solve_policy_serial (hit_refresh=True) for "
                         "--cap/--mode on locked traces instead of "
                         "reading a bandwidth-model row")
    ap.add_argument("--traces-dir",
                    default=str(ROOT / "results/traces_synth"))
    ap.add_argument("--hit-refresh", action="store_true", default=None,
                    help="use hit_refresh=True for --serial (default: "
                         "true; pass --no-hit-refresh for FIFO)")
    ap.add_argument("--no-hit-refresh", dest="hit_refresh",
                    action="store_false")
    ap.add_argument("--measured", required=True,
                    help="measured_santa_cruz_*.json blob")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.serial:
        a.hit_refresh = True if a.hit_refresh is None else a.hit_refresh

    if a.serial:
        import numpy as np
        from experiments import sim_paging as sp
        if a.cap is None or a.mode is None:
            sys.exit("--serial requires --cap and --mode")
        spec = sp.TIERS.get(a.tier) or sp.TIERS["36GB-M4M36"]
        tdir = Path(a.traces_dir)
        bf, bl, _b = sp.load_split(tdir / "build.npz")
        hf, hl, hT = sp.load_split(tdir / "holdout.npz")
        rng = np.random.default_rng(0)
        hs = np.sort(rng.choice(hT, size=min(8000, hT), replace=False))
        gold = np.stack([hl[li][hs] for li in range(sp.L)],
                        axis=1).astype(np.int16)
        prior_rank = {li: np.argsort(-np.bincount(bl[li].ravel(),
                                                  minlength=sp.E))
                      for li in range(sp.L)}
        res = sp.solve_policy_serial(gold, prior_rank, a.cap, a.mode,
                                     spec, hit_refresh=a.hit_refresh)
        sim = dict(kind="sim",
                   source=f"serial:{a.mode}@cap{a.cap}"
                          f"{'+hr' if a.hit_refresh else ''}",
                   tps=res["tps"],
                   compute_ms_per_tok=res["compute_ms"],
                   stream_ms_per_tok=None, limiter="serial",
                   residency=a.cap / sp.E,
                   experts_per_layer_resident=a.cap,
                   config=dict(mode=a.mode, cap=a.cap, model="serial",
                               hit_refresh=a.hit_refresh,
                               misses_per_tok=res["misses_per_tok"],
                               io_ms=res["io_ms"],
                               install_ms=res["install_ms"],
                               sync_ms=res["sync_ms"]))
        sil = silicon_record_from_measured(a.measured, source=a.measured)
        rep = gap_report(sim, sil)
        rep["serial_model"] = dict(res)
        print(format_gap_report(rep))
        if a.out:
            Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
            print(f"\nwrote {a.out}")
        return

    rows = json.loads(Path(a.sim).read_text())
    if isinstance(rows, dict):
        rows = rows.get("rows", [rows])
    sel = [r for r in rows if f"tps_{a.tier}" in r]
    if a.cap is not None:
        sel = [r for r in sel if r.get("cap") == a.cap]
    if a.mode is not None:
        sel = [r for r in sel if r.get("mode") == a.mode]
    if a.throttle is not None:
        sel = [r for r in sel if r.get("throttle") == a.throttle]
    if not sel:
        sys.exit(f"no sim row matches tier={a.tier} cap={a.cap} "
                 f"mode={a.mode} throttle={a.throttle} in {a.sim}")
    if len(sel) > 1:
        print(f"note: {len(sel)} rows match; using first", file=sys.stderr)
    row = sel[0]

    sim = sim_record_from_row(row, a.tier,
                              source=f"{a.sim}#{row.get('mode')}"
                                     f"/cap{row.get('cap')}")
    sil = silicon_record_from_measured(a.measured, source=a.measured)
    rep = gap_report(sim, sil)
    print(format_gap_report(rep))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
