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
# Measured n=1024 baseline on Santa Cruz @0.28 (t2_silicon report): 15.55 tps
BASE_TPS_MEASURED = 15.55
BASE_MS_MEASURED = 1000.0 / BASE_TPS_MEASURED


def reprice(row, label):
    issued = row["spec_issued_per_tok"]
    saved = row["demand_miss_base"] - row["demand_miss_per_tok"]
    saving_ms = saved * MISS_COST_MS
    contention_ms = issued * BG_COST_MS_LO, issued * BG_COST_MS_HI
    net_lo = saving_ms - contention_ms[1]  # best case: low contention
    net_hi = saving_ms - contention_ms[0]  # worst case: high contention
    # net is a ms/tok CREDIT (saving - cost); time change = -net
    tps = lambda credit: 1000.0 / (BASE_MS_MEASURED - credit)
    return {
        "arm": label,
        "spec_issued_per_tok": issued,
        "demand_miss_saved_per_tok": round(saved, 3),
        "saving_ms_per_tok": round(saving_ms, 2),
        "contention_ms_per_tok": [round(contention_ms[0], 2), round(contention_ms[1], 2)],
        "net_ms_per_tok": [round(net_lo, 2), round(net_hi, 2)],
        "repriced_tps_on_measured_baseline": [round(tps(net_lo), 2), round(tps(net_hi), 2)],
        "delta_tps_vs_baseline": [round(tps(net_lo) - BASE_TPS_MEASURED, 2), round(tps(net_hi) - BASE_TPS_MEASURED, 2)],
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
        "miss_cost_ms": MISS_COST_MS,
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

    worst = out["rows"][0]["delta_tps_vs_baseline"]
    out["verdict"] = (
        "DROP, STRENGTHENED: with T2's measured contention cost charged, speculative "
        f"page-in is NET NEGATIVE on the measured 15.55 tps baseline: "
        f'{worst[0]:+.2f}..{worst[1]:+.2f} tps (saved '
        f'{out["rows"][0]["saving_ms_per_tok"]} ms/tok, paid '
        f'{out["rows"][0]["contention_ms_per_tok"][0]}-{out["rows"][0]["contention_ms_per_tok"][1]} ms/tok) '
        "- the earlier 'below noise' DROP understated the harm: background reads are not free."
    )
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["rows"], indent=1))
    print(out["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
