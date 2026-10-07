#!/usr/bin/env python3
"""Breakeven-precision calculation for the T5 cold-window claim (6b359d4).

Cycle-27 claim under test: "with the contention breakeven, 10-25% precision
over a 52-expert budget is plausibly net-positive in the cold/early window."

This prices it with the committed constants — NO new simulation, deterministic
arithmetic, all provenance in-line:

Per issued speculative read (52/token budget, T2 contention probe a853292):
  contention cost  = 0.073..0.198 ms/expert  (MEASURED, Santa Cruz)
Per CONFIRMED spec (an averted demand miss):
  saving           = 0.52 ms (DB-ON basis, T4 Amendment 2 0af86e7, provisional)
                   or 0.82 ms (serial basis, model)
Breakeven precision p* = contention_per_issued / saving_per_confirmed.

Compare p* against the real-trace measured precisions (spec_pagein_realtrace_check.json,
6b359d4): hist1 25.4% (loose proxy: covered = t-1 top-10) / 10.6% (strict proxy:
covered = last-4-positions union).

Honesty: costs are MEASURED; the DB-ON saving basis is provisional pending T9;
precisions are signal-level on ONE cold real prompt with proxy residency.
"""
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "spec_pagein_coldwindow_breakeven.json"

BG_LO, BG_HI = 4.1 / 56.0, 10.7 / 54.0          # ms per issued spec (measured)
SAVE_SERIAL, SAVE_DBON = 0.82, 0.52             # ms per confirmed spec
RT = json.loads((REPO / "results" / "spec_pagein_realtrace_check.json").read_text())
P_LOOSE = RT["hist1_precision"]                 # covered = t-1's top-10
P_STRICT = RT["stricter_proxy_covered_last4"]["hist1_precision"]


def main():
    rows = []
    for name, save in (("serial 0.82 ms/miss", SAVE_SERIAL),
                       ("db_on 0.52 ms/miss (provisional, T4 Amd 2)", SAVE_DBON)):
        p_lo, p_hi = BG_LO / save, BG_HI / save
        rows.append({
            "saving_basis": name,
            "breakeven_precision": [round(p_lo, 3), round(p_hi, 3)],
            "hist1_loose_25.4pct_verdict": "pays" if P_LOOSE > p_hi else
                                           ("marginal" if P_LOOSE > p_lo else "loses"),
            "hist1_strict_10.6pct_verdict": "pays" if P_STRICT > p_hi else
                                            ("marginal" if P_STRICT > p_lo else "loses"),
        })
    out = {
        "kind": "simulated (deterministic breakeven arithmetic on measured constants)",
        "claim_under_test": "cycle-27 note: '10-25% precision over the 52-expert budget "
                             "is plausibly net-positive in the cold window' (6b359d4)",
        "contention_ms_per_issued": [round(BG_LO, 4), round(BG_HI, 4)],
        "realtrace_hist1_precision": {"loose_proxy": P_LOOSE, "strict_proxy": P_STRICT},
        "rows": rows,
        "verdict": "",
    }
    out["verdict"] = (
        "CYCLE-27 CLAIM CORRECTED: breakeven precision for spec page-in is "
        f'{rows[0]["breakeven_precision"][0]*100:.0f}-{rows[0]["breakeven_precision"][1]*100:.0f}% '
        f'(serial basis) and {rows[1]["breakeven_precision"][0]*100:.0f}-'
        f'{rows[1]["breakeven_precision"][1]*100:.0f}% (DB-ON basis). The real-trace hist1 '
        f'precision ({P_STRICT*100:.1f}% strict / {P_LOOSE*100:.1f}% loose proxy) sits AT or '
        "BELOW the DB-ON breakeven band — 'plausibly net-positive' was too generous. The honest "
        "cold-window verdict: MARGINAL at best under the loose proxy, LOSING under the strict "
        "proxy and the DB-ON basis; only hist3-class precision (22-45%) clears breakeven "
        "convincingly. Cold-window speculation needs either a better signal or a cheaper "
        "issue policy, not just the regime change."
    )
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
