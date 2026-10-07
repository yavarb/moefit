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
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moefit.metrics import (sim_record_from_row, silicon_record_from_measured,
                            gap_report, format_gap_report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True,
                    help="sim_paging-style results JSON (list of rows)")
    ap.add_argument("--tier", required=True, help="e.g. 48GB-M4M")
    ap.add_argument("--mode", default=None,
                    help="filter row by mode; default first matching cap")
    ap.add_argument("--cap", type=int, default=None,
                    help="filter row by cap (experts/layer)")
    ap.add_argument("--throttle", default=None)
    ap.add_argument("--measured", required=True,
                    help="measured_santa_cruz_*.json blob")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

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
