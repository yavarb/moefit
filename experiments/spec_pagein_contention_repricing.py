#!/usr/bin/env python3
"""Re-price the T5 speculative-page-in result (245eb87) with T2's MEASURED
background-IO contention cost.

Why: the sim's "opt" bound charged ZERO wall-clock for background spec reads
(they were assumed to hide fully under compute+sync idle). T2's silicon
contention probe (a853292) MEASURED that background expert reads at the
~50-55 expert/token budget slow decode by +4.1..+10.7 ms/tok (5.5-13.2%),
i.e. ~0.08-0.2 ms per background expert — real time, not free.

This script applies that measured cost to the measured waste counts in
results/design_spec_pagein.json (no re-simulation; deterministic arithmetic
on committed artifacts). Output: results/design_spec_pagein_contention.json.

Honesty: contention constants are MEASURED (Santa Cruz, t2_contention_probe);
waste/hit counts are SIM (locked synth holdout); baseline 15.55 tps is
MEASURED n=1024. The re-priced tps bands are therefore SIM-re-priced-by-
measured-constants, not measured silicon numbers.
"""
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "results" / "design_spec_pagein.json"
OUT = REPO / "results" / "design_spec_pagein_contention.json"

# --- measured constants (provenance in-line) ---
# T2 contention probe, budget 50-56 bg experts/tok: slowdown 5.5% and 13.2%
# on decode runs whose no-contention medians were ~15.5 tps (~64.3 ms/tok):
# +4.1 and +10.7 ms/tok. => per-background-expert cost 4.1/52 .. 10.7/54 ms.
BG_COST_MS_LO = 4.1 / 56.0   # optimistic ~0.073 ms/expert
BG_COST_MS_HI = 10.7 / 54.0  # pessimistic ~0.198 ms/expert
# Serial per-miss saving (measured slope 0.797 ms/expert; model 0.52 io + 0.30 install)
MISS_COST_MS = 0.82
# DB-ON basis (T4 Amendment 2, 0af86e7): staged install ON by default on the box;
# if install is overlapped under fetch, an averted miss pays io-only on the critical path.
MISS_COST_MS_DBON = 0.52
# Measured n=1024 baseline on Santa Cruz @0.28 (t2_silicon report): 15.55 tps
BASE_TPS_MEASURED = 15.55
BASE_MS_MEASURED = 1000.0 / BASE_TPS_MEASURED


def reprice(row, label):
    issued = row["spec_issued_per_tok"]
    saved = row["demand_miss_base"] - row["demand_miss_per_tok"]
    contention_ms = issued * BG_COST_MS_LO, issued * BG_COST_MS_HI
    tps = lambda credit: 1000.0 / (BASE_MS_MEASURED - credit)

    def basis(miss_cost):
        saving_ms = saved * miss_cost
        net_lo = saving_ms - contention_ms[1]  # best case: low contention
        net_hi = saving_ms - contention_ms[0]
        return {
            "saving_ms_per_tok": round(saving_ms, 2),
            "net_ms_per_tok": [round(net_lo, 2), round(net_hi, 2)],
            "repriced_tps_on_measured_baseline": [round(tps(net_lo), 2), round(tps(net_hi), 2)],
            "delta_tps_vs_baseline": [round(tps(net_lo) - BASE_TPS_MEASURED, 2),
                                      round(tps(net_hi) - BASE_TPS_MEASURED, 2)],
        }

    return {
        "arm": label,
        "spec_issued_per_tok": issued,
        "demand_miss_saved_per_tok": round(saved, 3),
        "contention_ms_per_tok": [round(contention_ms[0], 2), round(contention_ms[1], 2)],
        "serial_basis_miss_cost_0.82": basis(MISS_COST_MS),
        "db_on_basis_miss_cost_0.52": basis(MISS_COST_MS_DBON),
    }


def main():
    data = json.loads(SRC.read_text())
    base_row = data["rows"][0]  # baseline LRU, no speculation
    assert base_row["spec_issued_per_tok"] == 0.0
    out = {
        "kind": "simulated (re-priced with measured contention constants)",
        "source_artifact": "results/design_spec_pagein.json (commit 245eb87)",
        "contention_provenance": "T2 t2_contention_probe (a853292), Santa Cruz MEASURED: "
        "+4.1..+10.7 ms/tok at 54-56 bg experts/tok on ~15.5 tps decode",
        "bg_cost_ms_per_expert": [round(BG_COST_MS_LO, 4), round(BG_COST_MS_HI, 4)],
        "miss_cost_ms": {"serial_basis": MISS_COST_MS, "db_on_basis": MISS_COST_MS_DBON,
                         "db_on_provenance": "T4 Amendment 2 (0af86e7): staged install ON by "
                         "default on the box; pending T9 A/B/A confirmation"},
        "measured_baseline": {"tps": BASE_TPS_MEASURED, "ms_per_tok": round(BASE_MS_MEASURED, 2)},
        "rows": [],
        "verdict": "",
    }
    for row in data["rows"]:
        if row["gate"] == "none (baseline LRU, no speculation)":
            continue
        row = dict(row)
        row["demand_miss_base"] = base_row["demand_miss_per_tok"]
        out["rows"].append(reprice(row, f'{row["gate"]} cancel={row["cancel"]}'))

    ser = out["rows"][0]["serial_basis_miss_cost_0.82"]["delta_tps_vs_baseline"]
    dbon = out["rows"][0]["db_on_basis_miss_cost_0.52"]["delta_tps_vs_baseline"]
    out["verdict"] = (
        "DROP, STRENGTHENED and ROBUST to the install-term calibration: with T2's measured "
        "contention cost charged, speculative page-in is NET NEGATIVE on the measured 15.55 "
        f"tps baseline under BOTH per-miss bases: serial 0.82 ms/miss -> {ser[0]:+.2f}..{ser[1]:+.2f} tps; "
        f"DB-ON 0.52 ms/miss (T4 Amendment 2) -> {dbon[0]:+.2f}..{dbon[1]:+.2f} tps — "
        "the DB-ON basis SHRINKS the saving, making the regression WORSE, so whichever way "
        "T9's A/B/A settles the install term, history-driven spec page-in loses."
    )
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["rows"], indent=1))
    print(out["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
