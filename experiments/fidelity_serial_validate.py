"""T3 fidelity, part 3: validate lead_silicon's serial-latency model
OUT OF SAMPLE against two additional measured points.

Model under test (experiments/gap_m4max_36gb.py, commit 463c856, all
constants MEASURED on M4 Max 36 GB except compute):
    tok_ms = COMPUTE_MS + sum_layers(A + B*k if k>0) + INSTALL_MS*misses
             + 48*SYNC_MS
    A=0.20, B=0.52, INSTALL=0.30, SYNC=0.122 ms (microbench, F_NOCACHE)
    COMPUTE_MS = 23.2 is ASSUMED (128 GB 57.4 tok/s scaled 546->410 GB/s).

Already validated by lead_silicon: steady state cap=143 -> 12.1 tok/s
vs measured 12.71/13.0; cap=92 -> 8.9 vs crowded 7.8.

This script adds three independent checks (no new silicon, uses only
already-measured numbers):
  1. COLD-START TRANSIENT: the first bench (results/
     measured_m4max_36gb.json) ran a 16-token warmup from a cold
     ExpertCache at 6.43 tok/s (155.5 ms/token). Replay true-LRU from
     EMPTY caches and predict that transient.
  2. STEADY-STATE reproduction at cap=143 with warmup (sanity vs
     gap_m4max_36gb's 12.1) using an independently validated replica.
  3. COMPUTE-TERM SENSITIVITY: COMPUTE_MS is the model's largest
     unmeasured knob. Sweep it over {18.1 (sim DRAM_EFF@546 GB/s),
     23.2 (assumed), 24.2 (sim DRAM_EFF@410 GB/s)} and see which
     calibration the measured 12.7-13.0 tok/s favors.

usage: python3 experiments/fidelity_serial_validate.py
"""
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import sim_paging as sp  # noqa: E402

# measured constants (results/microbench_expert_reads_m4max_36gb.json)
IO_A_MS, IO_B_MS = 0.20, 0.52
INSTALL_MS, SYNC_MS = 0.30, 0.122
COMPUTE_ASSUMED = 23.2
COMPUTE_SIM_546 = (sp.READ_FLOOR_MIB + sp.GOLD_MIB) / 1024.0 \
    / (546.0 * sp.DRAM_EFF) * 1000.0        # 18.1 ms
COMPUTE_SIM_410 = (sp.READ_FLOOR_MIB + sp.GOLD_MIB) / 1024.0 \
    / (410.0 * sp.DRAM_EFF) * 1000.0        # 24.2 ms


def lru_misses(gold, cap):
    """True-LRU per-layer misses, from EMPTY caches. (T, L) int array.

    Mirrors oMLX ExpertCache semantics = hit refreshes (move_to_end).
    Same dynamics as gap_m4max_36gb.lru_misses; kept independent here.
    """
    caches = [OrderedDict() for _ in range(sp.L)]
    M = np.zeros(gold.shape[:2], np.int32)
    for t in range(len(gold)):
        for li in range(sp.L):
            c = caches[li]
            m = 0
            for e in set(int(x) for x in gold[t, li]):
                if e in c:
                    c.move_to_end(e)
                else:
                    m += 1
                    c[e] = 1
                    if len(c) > cap:
                        c.popitem(last=False)
            M[t, li] = m
    return M


def tok_ms(M_t, compute_ms):
    k = M_t.astype(float)
    miss = k.sum()
    io = np.where(k > 0, IO_A_MS + IO_B_MS * k, 0.0).sum()
    return compute_ms + io + INSTALL_MS * miss + sp.L * SYNC_MS


def main():
    _, hl, _ = sp.load_split(ROOT / "results/traces_synth/holdout.npz")
    gold = np.stack([hl[li] for li in range(sp.L)], axis=1).astype(np.int16)

    out: dict = {"model_constants": {"io_A_ms": IO_A_MS, "io_B_ms": IO_B_MS,
                               "install_ms": INSTALL_MS, "sync_ms": SYNC_MS},
           "compute_ms_candidates": {
               "assumed_546to410_scaled": round(COMPUTE_ASSUMED, 1),
               "sim_dram_eff_at_546": round(COMPUTE_SIM_546, 1),
               "sim_dram_eff_at_410": round(COMPUTE_SIM_410, 1)}}

    # cold transient: caches empty at t=0, exactly like the warmup run
    M143 = lru_misses(gold, 143)
    ms = np.array([tok_ms(M143[t], COMPUTE_ASSUMED) for t in range(len(M143))])
    out["cold_transient_cap143"] = {
        "measured_warmup_16tok_tps": 6.43,
        "measured_warmup_16tok_ms": round(1000 / 6.43, 1),
        "predicted_first16_mean_ms": round(ms[:16].mean(), 1),
        "predicted_first16_tps": round(1000 / ms[:16].mean(), 1),
        "predicted_first16_range_ms": [round(float(ms[:16].min()), 1),
                                       round(float(ms[:16].max()), 1)],
        "predicted_first128_mean_tps": round(1000 / ms[:128].mean(), 1),
        "predicted_steady_tps_warmup200": round(1000 / ms[200:].mean(), 1),
        "all_miss_floor_tps":
            round(1000 / (COMPUTE_ASSUMED + 48 * (IO_A_MS + IO_B_MS * 10)
                          + 480 * INSTALL_MS + sp.L * SYNC_MS), 1),
        "note": "warmup ran from a cold server (4.7 s cold load); synth "
                "holdout rows are a different prompt, so this is a "
                "ballpark test of the transient, not a matched-config one.",
    }

    # steady-state reproduction at both measured caps
    steady = {}
    for cap, measured in ((143, "12.71 / 13.0"), (92, "7.8 (crowded)")):
        M = lru_misses(gold, cap)[200:]
        m_ms = np.array([tok_ms(M[t], COMPUTE_ASSUMED) for t in range(len(M))])
        steady[f"cap{cap}"] = {
            "predicted_tps": round(1000 / m_ms.mean(), 1),
            "measured_tps": measured,
            "misses_per_token": round(float(M.sum(1).mean()), 1),
            "frac_layer_steps_with_miss": round(float((M > 0).mean()), 3),
        }
    out["steady_state"] = steady

    # compute-term sensitivity at cap=143 steady state
    M = M143[200:]
    sens = {}
    for label, c_ms in (("assumed_23.2", COMPUTE_ASSUMED),
                        ("sim_dram_eff_546_18.1", COMPUTE_SIM_546),
                        ("sim_dram_eff_410_24.2", COMPUTE_SIM_410)):
        per = np.array([tok_ms(M[t], c_ms) for t in range(len(M))])
        sens[label] = {"compute_ms": round(c_ms, 1),
                       "predicted_tps": round(1000 / per.mean(), 1)}
    sens["measured"] = "12.71 (256-tok iostat run) / 13.0 (median of 3 x 128)"
    out["compute_sensitivity_cap143"] = sens

    outp = ROOT / "results/fidelity_serial_validate.json"
    outp.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    print("wrote", outp)


if __name__ == "__main__":
    main()
