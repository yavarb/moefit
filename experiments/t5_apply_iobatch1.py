#!/usr/bin/env python3
"""T5: apply the REGISTERED IO_BATCH=1 conditional (088913d) to T9's
MEASURED BATCH1 arm (t9_batch1_C.json + t9_b1_stats{0,1}.json).

Registered formulas (results/t5_iobatch1_conditional_prereg.json):
  D  = 1000/tps_b1 - 1000/15.72
  m  = stats diffs over the arm's 3072 scored tokens (PRIMARY);
       fallback 37.1 only if stats unavailable
  S' = 0.52 + D/m, APPLIED ONLY IF the sanity gates pass:
       D >= 0 AND D/m <= 0.30. Gate failure -> NO re-price, attribution
       flagged to T9/T4. This script follows that rule mechanically.

Also composes the sidecar measurement (t5_sidecar_aba_scored.json,
S - C = 0.509 ms/expert measured) with the contention band to bound S —
independent of the gated S' re-price.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/t5_iobatch1_applied.json"

ON_TPS = 15.72            # measured ON median (75cdc0f / 447a027)
CONT = (0.073, 0.198)     # measured contention band (T2 probe, a853292)
INSTALL_MAX = 0.30        # serial install constant


def main():
    b1 = json.loads((REPO / "results/t9_silicon/t9_batch1_C.json").read_text())
    tps_b1_runs = [r["decode_tps"] for r in b1["runs"]]
    tps_b1 = sorted(tps_b1_runs)[len(tps_b1_runs) // 2]
    s0 = json.loads((REPO / "results/t9_silicon/t9_b1_stats0.json").read_text())
    s1 = json.loads((REPO / "results/t9_silicon/t9_b1_stats1.json").read_text())
    tokens = sum(r["tokens"] for r in b1["runs"])
    m = (s1["misses"] - s0["misses"]) / tokens   # PRIMARY source

    D = 1000.0 / tps_b1 - 1000.0 / ON_TPS
    d_over_m = D / m
    gates = {"D_nonnegative": D >= 0, "D_over_m_le_install": d_over_m <= INSTALL_MAX}
    reprice = gates["D_nonnegative"] and gates["D_over_m_le_install"]
    S_prime = 0.52 + d_over_m if reprice else None

    sidecar = json.loads((REPO / "results/t5_sidecar_aba_scored.json").read_text())
    smc = sidecar["descriptive_framework_readout"]["implied_S_minus_C_ms"]
    S_lo = smc + CONT[0]      # S bounds from measured S-C x measured C band
    S_hi = smc + CONT[1]

    out = {
        "kind": "T5 application of the registered IO_BATCH=1 conditional "
                "(088913d) to the MEASURED BATCH1 arm (T9, eligibility clean, "
                "Amendment-4 candidate A confirmed by T4's scoring)",
        "inputs": {
            "batch1_tps_runs": tps_b1_runs, "batch1_median": tps_b1,
            "on_median": ON_TPS,
            "m_from_stats_diffs": round(m, 3),
            "m_note": f"d_misses {s1['misses'] - s0['misses']} over "
                      f"{tokens} scored tokens; the arm's repeats ran WARM "
                      "(m ~9.6/tok vs 44 cold incl. prefill in the sidecar "
                      "run and ~37 first-run anchor) — m is strongly "
                      "regime-dependent, which is exactly why the prereg "
                      "carried an m band",
        },
        "gates": {**gates, "D_ms_per_tok": round(D, 2),
                  "D_over_m": round(d_over_m, 3)},
        "gate_result": "PASS -> S' re-priced" if reprice else
                       "FAIL -> NO re-price per the registered rule; "
                       "attribution flagged to T9/T4",
        "S_prime": None if S_prime is None else round(S_prime, 3),
        "attribution_flag": None if reprice else (
            f"D/m = {d_over_m:.3f} > 0.30 = the serial install constant: the "
            "measured DB term EXCEEDS install-exposure x misses at the arm's "
            "own window m. Either the window m understates the arm's true "
            "per-run misses (warmup contaminates the stats window: numerator "
            "includes warmup misses, denominator is scored-tokens only), or "
            "window=1 costs more than install exposure alone (fetch-order "
            "serialization: expert i+1's reads wait for expert i's install). "
            "Both readings are T4/T9 attribution territory; T5's rule is "
            "mechanical: no S' re-price."),
        "sidecar_composition_stands": {
            "note": "independent of the gated S' re-price: the sidecar "
                    "measurement bounds S via S - C = 0.509 with the "
                    "measured contention band",
            "S_minus_C_measured": smc,
            "S_bounds_from_measured_C": [round(S_lo, 3), round(S_hi, 3)],
            "install_exposure_bounds": [round(S_lo - 0.52, 3),
                                        round(S_hi - 0.52, 3)],
            "read": f"S in [{S_lo:.2f}, {S_hi:.2f}] ms/miss: the DB-ON "
                    "saving constant CANNOT be 0.52 (install fully hidden) "
                    "unless the contention floor itself is falsified — "
                    "install exposure is at least "
                    f"{max(S_lo - 0.52, 0):.2f} ms/miss on the live box.",
        },
        "provenance": {
            "registered": "results/t5_iobatch1_conditional_prereg.json (088913d)",
            "measured": "results/t9_silicon/t9_batch1_C.json + t9_b1_stats{0,1}.json",
            "sidecar": "results/t5_sidecar_aba_scored.json (d2f860f)",
        },
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
