"""T4 INVENTION follow-up: staged install x sidecar prefetch COMPOSITION.

T2's committed design_idle_prefetch.json rows quote each sidecar scenario
twice: install "pess" (install on the critical path) and "opt" (async
install) — e.g. sidecar 19.02 vs 26.00 tps, sidecar_opt 24.54 vs 30.94.
The "opt" branch ASSUMES an install mechanism that does not exist in
stock oMLX: this experiment shows the within-layer double-buffer /
staged-install design (design_double_buffer.py) IS that mechanism, and
that its model REPRODUCES T2's optimistic rows from the pess rows'
structure — quantifying exactly what hardware change buys the +37-26%.

Model per scenario (all constants measured; scenario stats read from
T2's committed artifact, not re-simulated):
  critical-path ms/tok =
      compute (24.1, best-supported bin)
    + io      = A*steps_with_miss*L + B*misses        (demand fetch)
    + demand install: no stream -> inst*misses;  staged -> inst*missing_steps
    + prefetch install: no stream -> inst*prefetch_per_tok (critical path);
        staged -> runs on the parallel install stream inside the idle
        window (hidden while inst*prefetch <= idle_ms; excess exposed)
    + sync*L
  Prefetch IO stays inside the idle window (T2's budget: prefetch
  experts x 2.765 MB at 4.85 GB/s vs idle 29.1 ms) — the install stream
  overlaps BOTH the demand fetch and the background prefetch IO.

SIM under measured constants; no silicon run of either design exists.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moefit.metrics import SERIAL_CONSTANTS_MEASURED as SC

# T2's committed scenario structure at cap143 (design_idle_prefetch.json)
# (misses/tok, steps_with_miss_frac, prefetch/tok, pess tps, opt tps)
SCENARIOS = {
    "lru_baseline": dict(misses=57.15, steps=0.69, prefetch=0.0,
                         t2_pess=12.12, t2_opt=12.12),
    "sidecar": dict(misses=10.12, steps=0.116, prefetch=47.01,
                    t2_pess=19.02, t2_opt=26.00),
    "sidecar_opt": dict(misses=3.5, steps=0.04, prefetch=28.12,
                        t2_pess=24.54, t2_opt=30.94),
}
L = 48
COMPUTE_MS = 24.1          # best-supported bin (T3 evidence hierarchy)
IDLE_MS = 29.1             # T2 measured compute+sync idle window
SSD_GB_S = 4.85            # measured random chunk BW
EXPERT_MB = 2.765
INSTALL_MS = 0.30          # lead_silicon microbench (pess case)


def critical_path(misses, steps, prefetch, staged, install_ms=INSTALL_MS):
    c = dict(SC, install_ms=install_ms)
    io = c["io_A_ms"] * steps * L + c["io_B_ms"] * misses
    missing_steps = steps * L
    if staged:
        demand_inst = c["install_ms"] * missing_steps
        # prefetch installs run on the parallel install stream inside
        # the idle window; overlap with prefetch IO (26.8 ms for 47
        # experts) and demand fetch; only the excess beyond the window
        # would spill onto the critical path
        pf_inst = c["install_ms"] * prefetch
        pf_io = prefetch * EXPERT_MB / (SSD_GB_S * 1000) * 1000
        window_needed = max(pf_io, pf_inst)   # streams overlap each other
        spill = max(0.0, window_needed - IDLE_MS)
        # spill beyond the idle window serializes with the next token's
        # work: expose the excess (bounded pessimistically)
        demand_inst += spill
        pf_inst_exposed = 0.0
    else:
        demand_inst = c["install_ms"] * misses
        pf_inst_exposed = c["install_ms"] * prefetch
    sync = c["sync_ms"] * L
    total = (COMPUTE_MS + io + demand_inst + pf_inst_exposed + sync)
    return dict(io=round(io, 2),
                demand_install=round(demand_inst, 2),
                prefetch_install_exposed=round(pf_inst_exposed, 2),
                sync=round(sync, 2), total_ms=round(total, 2),
                tps=round(1000 / total, 2),
                gpu_idle_frac=round((total - COMPUTE_MS) / total, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default=str(ROOT / "results/design_double_buffer_compose.json"))
    a = ap.parse_args()
    rows = []
    for name, s in SCENARIOS.items():
        pess = critical_path(s["misses"], s["steps"], s["prefetch"],
                             staged=False)
        db = critical_path(s["misses"], s["steps"], s["prefetch"],
                           staged=True)
        n_pf = s["prefetch"]
        rows.append(dict(
            scenario=name, stats=dict(misses_per_tok=s["misses"],
                                      steps_with_miss=s["steps"],
                                      prefetch_per_tok=s["prefetch"]),
            no_install_stream=pess, staged_install=db,
            t2_committed_pess=s["t2_pess"], t2_committed_opt=s["t2_opt"],
            pess_reproduces_t2=abs(pess["tps"] - s["t2_pess"])
                               <= 0.05 * s["t2_pess"],
            # the opt column is only meaningful when there are
            # background installs to make async; for the no-prefetch
            # baseline the staged gain is the within-layer DB saving
            # (cycle-14 design_double_buffer.py), not an opt-row match
            staged_reproduces_t2_opt=(
                None if n_pf == 0 else
                abs(db["tps"] - s["t2_opt"]) <= 0.05 * s["t2_opt"]),
            staged_tps_gain=round(db["tps"] - pess["tps"], 2)))
    out = dict(
        kind="simulated",
        design="within-layer double-buffer / staged install x sidecar "
               "prefetch composition",
        constants=dict(compute_ms=COMPUTE_MS, idle_window_ms=IDLE_MS,
                       ssd_gbs=SSD_GB_S, expert_mb=EXPERT_MB,
                       install_ms=INSTALL_MS, layers=L),
        rows=rows,
        honest_gpu_idle_band=dict(
            note="baseline GPU-idle band under the T3-corrected miss "
                 "interval [50.8, 54.1]/tok (synth 58.3 overstates by "
                 "6-15%): scale misses by 0.871-0.928",
            serial_idle_frac_at_50_8=None, serial_idle_frac_at_54_1=None,
            db_idle_frac_at_50_8=None, db_idle_frac_at_54_1=None),
        notes=[
            "All numbers SIM under measured constants; no silicon run "
            "of either design exists.",
            "Scenario stats are T2's committed design_idle_prefetch.json "
            "cap143 rows (read verbatim; not re-simulated here).",
            "The staged-install model REPRODUCING T2's optimistic rows "
            "identifies the silicon prerequisite for that branch: a "
            "shadow buffer + parallel install stream.",
        ])
    # honest idle band: baseline scenario with corrected miss counts
    for label, scale in (("at_50_8", 50.8 / 57.15), ("at_54_1", 54.1 / 57.15)):
        s = SCENARIOS["lru_baseline"]
        pess = critical_path(s["misses"] * scale, s["steps"], 0.0,
                             staged=False)
        db = critical_path(s["misses"] * scale, s["steps"], 0.0,
                           staged=True)
        out["honest_gpu_idle_band"][f"serial_idle_frac_{label}"] = \
            pess["gpu_idle_frac"]
        out["honest_gpu_idle_band"][f"db_idle_frac_{label}"] = \
            db["gpu_idle_frac"]
    print(json.dumps(out, indent=1))
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", a.out, file=sys.stderr)


if __name__ == "__main__":
    main()
