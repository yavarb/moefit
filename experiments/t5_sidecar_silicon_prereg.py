#!/usr/bin/env python3
"""T5 PRE-REGISTRATION: breakeven-framework prediction for T2's SIDECAR
silicon A/B/A, committed BEFORE any sidecar arm result lands (T2's run was
in flight at registration time; no B-arm data exists in the repo).

Why T5 owns this: T2's deployed sidecar IS the T5 cancel-on-wrong-route
primitive on silicon — exact-replay signal (precision ~1.0 in the repeated
same-prompt regime), reads issued on a separate lane, bytes HELD not
installed (no pollution), wrong routes cancelled. This is the framework's
first PROSPECTIVE silicon test (all prior uses were retrospective).

Prediction (deterministic arithmetic on measured constants):
  net_ms/tok = m_covered * SAVING  -  m_issued * CONTENTION
    SAVING per confirmed spec (demand miss averted) = 0.52 ms (DB-ON io-only,
      install overlapped under fetch per T4 Amd 2, corroborated by T9
      TRANSFERS +6.53) .. 0.82 ms (serial, T1's microbench)
    CONTENTION per issued background read = 0.073 .. 0.198 ms/expert
      (T2's OWN measured interference probe at the 50-56/tok budget)
  In the replay regime the sidecar covers the arm's own demand-miss set up
  to its budget, so m_covered ~= m_issued ~= m = the arm's logical miss/tok,
  MEASURED by T2's live stats counters. Prereg band over m in [30, 50]
  (measured anchors: 102.6 MB/tok / 2.765 MB = 37.1-39.5 full-expert
  misses/tok; T9 process-lifetime counters imply ~30-35 on this prompt).
  Baseline: measured DB-ON 15.72 tps = 63.6 ms/tok (T9 A/A2 medians).

Decision rules (fixed before data):
  1. Measured B-arm median tps inside the predicted band -> framework
     TRANSFERS prospectively; the DB-ON saving constant 0.52 gains silicon
     support.
  2. Below the band's floor -> the DB-ON saving constant is OVERSTATED
     (install overlap hides less than Amd 2 assumed) OR sidecar coverage
     m_covered << m_issued (replay misses the warm-cache demand set) —
     T2's stats counters distinguish these: if m_issued/tok >> m_covered/tok,
     coverage is the failure; if coverage ~1 and gain still short, the
     saving constant is.
  3. ABOVE the band's ceiling -> contention on the live pipeline is CHEAPER
     than the 0.073 ms/expert floor (the 12-worker pool absorbs the sidecar's
     4-thread lane with <25% interference) — would revise the contention
     term DOWN for all background-read designs.

No silicon numbers are claimed; this is a prediction. All constants carry
their committed provenance. Not a knob sweep: m is measured per arm by the
counters, SAVING/CONTENTION come from committed measurements.
"""
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "t5_sidecar_silicon_prereg.json"

BASE_MS = 1000.0 / 15.72            # measured DB-ON baseline (T9 A/A2)
SAVE = (0.52, 0.82)                 # ms per confirmed spec
CONT = (0.073, 0.198)               # ms per issued background read
M_BAND = (30, 50)                   # logical miss/tok band (measured anchors)


def main():
    rows = []
    for m in M_BAND:
        best = m * SAVE[0] - m * CONT[1]   # pessimistic: min saving, max contention
        worst = m * SAVE[1] - m * CONT[0]  # optimistic: max saving, min contention
        rows.append({
            "m_miss_per_tok": m,
            "net_ms_per_tok": [round(best, 1), round(worst, 1)],
            "predicted_tps": [round(1000 / (BASE_MS - best), 1),
                               round(1000 / (BASE_MS - worst), 1)],
        })
    floor = min(r["predicted_tps"][0] for r in rows)
    ceil = max(r["predicted_tps"][1] for r in rows)
    out = {
        "kind": "PRE-REGISTERED PREDICTION (no sidecar data existed at commit)",
        "target": "T2 sidecar silicon A/B/A (experiments/t2_sidecar_aba.py, in flight)",
        "framework": "T5 breakeven (d7f45b6/c69ca70) — first prospective silicon test",
        "constants": {"base_ms_per_tok": round(BASE_MS, 1),
                      "saving_ms_per_confirmed": list(SAVE),
                      "contention_ms_per_issued": list(CONT),
                      "provenance": "T9 75cdc0f (baseline), T4 Amd 2 0af86e7 + "
                                    "T1 microbench (saving), T2 probe a853292 (contention)"},
        "rows": rows,
        "predicted_band_tps": [floor, ceil],
        "decision_rules": [
            "inside band -> framework TRANSFERS prospectively; DB-ON saving 0.52 supported",
            "below floor -> saving constant overstated OR coverage m_cov << m_iss; "
            "T2 stats counters (m_iss/tok vs m_cov/tok) arbitrate which",
            "above ceiling -> live-pipeline contention cheaper than 0.073 ms/expert; "
            "revise contention term down for all background-read designs",
        ],
        "note": "T2's own sim bracket (12.1 -> 19.0 install-critical / 26.0 async, "
                "377eb7c) sits INSIDE this band — the framework and the design-side "
                "model agree before the measurement.",
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
