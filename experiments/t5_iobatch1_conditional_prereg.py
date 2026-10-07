#!/usr/bin/env python3
"""T5 AMENDMENT to the sidecar prereg + DROP re-pricing, registered BEFORE
T9's IO_BATCH=1 DB-isolating arm runs (arm staged in 9991556, not executed).

CORRECTION OF T5's OWN CYCLE-46 CLAIM (honesty fix, before data):
My notebook entry said the IO_BATCH=1 arm "can only shrink the saving term
further — direction favorable to the DROP". The SIGN IS WRONG for the saving
term: the arm isolates the double-buffer/install-overlap term D. If D > 0,
install is NOT fully hidden under fetch on the live box, so an averted miss
saves 0.52 (io) + D/m (residual install) — a LARGER saving, i.e. LESS
negative for spec page-in. The correct statement:

  per-miss cost on the DB-ON box = 0.52 + D/m, with D/m in [0, 0.30]
    (0.30 = serial install constant; D=0 <=> install fully hidden).

WHY THE DROP VERDICT CANNOT FLIP ANYWAY (pre-registered):
The whole outcome space of IO_BATCH=1 was already computed in b283105/ab6d37a:
D/m = 0        -> the 0.52-basis rows: -0.80..-2.09 tps (net regression)
D/m = 0.30     -> the 0.82-basis rows: -0.76..-2.06 tps (net regression)
Both endpoints negative; every interior point is a convex combination, so
spec page-in stays a regression for ALL possible IO_BATCH=1 outcomes. The
arm can only relocate the verdict INSIDE an already-computed bracket — it
cannot flip it. (The same bracketing already covers the cold-window
breakeven: p* in 9-38% over both bases, hist1 real-trace 10.6-25.4% ->
"marginal at best" unchanged.)

CONDITIONAL RE-PRICING FORMULAS (fixed before data, applied when the arm lands):
Let tps_b1 = median tps of the IO_BATCH=1 arm (3x n=1024, same prompt),
  ms_b1 = 1000/tps_b1, ms_on = 1000/15.72 (measured ON median, 75cdc0f).
  D  = ms_b1 - ms_on          (the pure DB/install-overlap term, ms/tok)
  S' = 0.52 + D/m             (new DB-ON saving constant, ms per averted miss)
  m  = the arm's logical miss/tok from T2's stats counters (t9_b1_stats*.json
       diffs over the arm's 3072 tokens; fallback: 102.6 MB/tok / 2.765 MB).
Sanity gates: if D < 0 or D/m > 0.30, the arm is inconsistent with the
committed model — report, do not re-price (flag for T9/T4 attribution).
Effects to apply on landing:
  1. DROP rows: net per issued spec = p*S' - C with C = 0.073..0.198 measured
     (contention unchanged — it was measured ON the default box, so D is
     already embedded in C). Breakeven precision p* = C/S'.
  2. SIDECAR PREREG BAND (t5_sidecar_silicon_prereg.json): replace the
     saving bracket (0.52, 0.82) with the singleton S' if the arm lands
     BEFORE the sidecar result; else score the sidecar against the original
     bracket and report the S' refinement as a post-hoc sensitivity.
  3. The JSON predicted band 18.5-38.1 tps is CANONICAL; the 81a6483 commit
     message's "17.6-31.0" was a stale draft figure — superseded, JSON wins.

No silicon numbers claimed. Registered 2026-10-08 before any IO_BATCH=1 run.
"""
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "t5_iobatch1_conditional_prereg.json"


def main():
    out = {
        "kind": "CONDITIONAL PRE-REGISTRATION + SELF-CORRECTION (pre-data)",
        "target": "T9 IO_BATCH=1 DB-isolating arm (experiments/t9_batch1_arm.sh, staged 9991556)",
        "correction": "T5 cycle-46 'direction favorable to the DROP' claim is SIGN-WRONG: "
                      "D>0 means a LARGER saving per averted miss (0.52 + D/m), not smaller. "
                      "Conclusion unchanged (see bracketing), reasoning corrected at the source.",
        "bounds": {
            "per_miss_cost_db_on": "[0.52, 0.82] ms (install in [0, 0.30] exposed)",
            "both_endpoints_computed": "0.52-basis -0.80..-2.09 tps; 0.82-basis -0.76..-2.06 tps",
            "flip_possible": False,
            "why": "both bracket endpoints net-negative; interior points are convex combinations",
        },
        "formulas": {
            "D_ms_per_tok": "1000/tps_b1 - 1000/15.72",
            "S_prime_ms_per_miss": "0.52 + D/m",
            "m_source": "t9_b1_stats diffs over 3072 tokens; fallback 102.6/2.765 = 37.1",
            "sanity_gates": "D < 0 or D/m > 0.30 -> do not re-price; flag attribution to T9/T4",
            "breakeven_p_star": "C/S' with C = 0.073..0.198 (unchanged, measured on default box)",
        },
        "sidecar_band_rule": "singleton S' replaces the (0.52, 0.82) saving bracket only if the "
                             "arm lands BEFORE the sidecar result; else post-hoc sensitivity. "
                             "Canonical band = t5_sidecar_silicon_prereg.json 18.5-38.1 tps "
                             "(commit-message 17.6-31.0 was a stale draft, superseded).",
        "registered_before": "any IO_BATCH=1 execution",
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
