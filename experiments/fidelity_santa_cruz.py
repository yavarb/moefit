"""T3 fidelity: simulator vs measured Santa Cruz M4 Max 36 GB (13.0 tok/s).

Measured reference (results/measured_santa_cruz_36gb.json, 2026-10-06 idle
re-bench, commit f20bdd3/c1ca86b):
  - oMLX 0.7.0 expert offload, resident fraction 0.28 -> 143 of 512
    experts per layer (oMLX log: 18.98 of 67.95 GB expert tables resident,
    22.61 GB total process footprint)
  - decode 13.0 tok/s median of 3 x 128-token greedy runs
    (= 76.92 ms/token), 12.23-13.07 range
  - PLE n-gram table on SSD (qwen4_ple_ssd_offload), MTP off
  - during decode only 18-19% of 36 GB free

This script answers three questions WITHOUT running silicon:
  1. What does the shipped simulator predict at the *same* operating
     point (cap=143, M4 Max tier) under each residency policy?
  2. Which (served fraction, SSD bandwidth, PLE traffic) combination
     reproduces the measured 76.92 ms/token?
  3. Which constants diverge from the oMLX-measured geometry?

Uses the synthetic traces (results/traces_synth) because the real
router traces are not in this checkout; synth was fitted to within a
few points of the real LRU / pinned served fractions (calibration.json
overall prior14 fit 0.188 vs target 0.187).

usage: python3 experiments/fidelity_santa_cruz.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import sim_paging as sp  # noqa: E402

MEASURED_TPS = 13.0
CAP = 143                     # measured resident experts per layer (0.28 x 512)
TOK_MS = 1000.0 / MEASURED_TPS  # 76.92 ms/token
SSD_ASSUMED = 7.4             # GB/s the sim's M4 Max tier assumes
DRAM_M4MAX = 546.0            # GB/s peak, sim DRAM_EFF=0.537 -> 293 effective

# oMLX-measured geometry (log numbers from measured_santa_cruz_36gb.json)
OMLEX_EXPERT_GB = 67.95       # decimal GB total expert tables
OMLEX_RESIDENT_GB = 18.98
OMLEX_FOOTPRINT_GB = 22.61
N_EXPERTS = sp.L * sp.E


def main():
    bf, bl, bT = sp.load_split(ROOT / "results/traces_synth/build.npz")
    hf, hl, hT = sp.load_split(ROOT / "results/traces_synth/holdout.npz")
    rng = np.random.default_rng(0)
    hs = np.sort(rng.choice(hT, size=min(8000, hT), replace=False))
    gold = np.stack([hl[li][hs] for li in range(sp.L)], axis=1).astype(np.int16)
    prior_rank = {li: np.argsort(-np.bincount(bl[li].ravel(), minlength=sp.E))
                  for li in range(sp.L)}

    out = {"measured": {"decode_tps_median": MEASURED_TPS,
                        "ms_per_token": round(TOK_MS, 2),
                        "cap_per_layer": CAP}}

    # --- geometry audit: sim constants vs oMLX-measured ---------------
    omlx_per_expert_mib = OMLEX_EXPERT_GB * 1000.0 / N_EXPERTS / (1024 ** 2) * (1024 ** 2) / (1000.0 ** 0)  # MB->MiB below
    omlx_per_expert_mib = OMLEX_EXPERT_GB * 1e9 / N_EXPERTS / (1024 ** 2)
    sim_footprint_gib = sp.FLOOR_MIB / 1024 + CAP * sp.L * sp.EXPERT_MIB / 1024
    omlx_resident_from_frac = OMLEX_EXPERT_GB * (CAP / sp.E)
    out["geometry"] = {
        "sim_expert_mib": sp.EXPERT_MIB,
        "omlex_per_expert_mib": round(omlx_per_expert_mib, 3),
        "expert_size_gap_pct": round(100 * (omlx_per_expert_mib / sp.EXPERT_MIB - 1), 2),
        "sim_footprint_gib_at_cap143": round(sim_footprint_gib, 2),
        "omlex_footprint_gb": OMLEX_FOOTPRINT_GB,
        "omlex_resident_gb_from_0.28": round(omlx_resident_from_frac, 2),
        "omlex_resident_gb_log": OMLEX_RESIDENT_GB,
    }

    # --- policies at cap=143 ------------------------------------------
    zero_picks = np.zeros((len(gold), sp.L, 1), np.int16)
    policies = {}
    # NOTE: shipped simulate() references an undefined global `pin_counts`
    # when mode="prior" (NameError at head). The shipped prior rows in
    # results/sim_paging.json predate this. We supply the module-global it
    # expects so prior mode runs; fix belongs to the sim owner.
    sp.pin_counts = None
    for mode in ("lru", "prior", "sidecar"):
        srv, sync_mb, async_mb = sp.simulate(
            gold, zero_picks, prior_rank, CAP, mode,
            0 if mode != "sidecar" else sp.K, 0)
        policies[mode] = dict(mode=mode, served=round(srv, 3),
                              sync_mb_per_tok=round(sync_mb, 1),
                              async_mb_per_tok=round(async_mb, 1))

    # static-first: oMLX-style "keep the first `frac` experts resident"
    # (no hot-set, no LRU) — is that what oMLX 0.7.0 offload does?
    for frac_label, keep in (("static_first_143", CAP),
                             ("static_last_143", None)):
        if keep is None:
            mask = np.zeros(sp.E, bool)
            mask[sp.E - CAP:] = True
        else:
            mask = np.zeros(sp.E, bool)
            mask[:CAP] = True
        hit = mask[gold]                       # (T, L, K) bool
        miss_experts = (~hit).sum()
        sync_mb = miss_experts / len(gold) * sp.EXPERT_MIB
        policies[frac_label] = dict(
            mode=frac_label, served=round(hit.mean(), 3),
            sync_mb_per_tok=round(sync_mb, 1), async_mb_per_tok=0.0)

    # --- what each policy predicts, and what it would take -----------
    dram_mb = sp.READ_FLOOR_MIB + sp.GOLD_MIB
    c_ms = dram_mb / 1024.0 / (DRAM_M4MAX * sp.DRAM_EFF) * 1000.0
    out["compute_floor"] = {"dram_mb_per_tok": round(dram_mb, 1),
                            "compute_ms": round(c_ms, 2),
                            "compute_bound_tps": round(1000 / c_ms, 1)}

    for name, p in policies.items():
        p["predict_tps_at_ssd7.4"] = round(
            1000.0 / max(c_ms, (p["sync_mb_per_tok"] + sp.PLE_STREAM_MIB) / 1024.0
                         / SSD_ASSUMED * 1000.0), 1)
        p["predict_tps_at_ssd3.0"] = round(
            1000.0 / max(c_ms, (p["sync_mb_per_tok"] + sp.PLE_STREAM_MIB) / 1024.0
                         / 3.0 * 1000.0), 1)
        # inverse: effective SSD GB/s this policy's miss stream would need
        # to hit the measured 76.92 ms/token (stream-bound case)
        p["implied_ssd_gbps_for_13tps"] = round(
            (p["sync_mb_per_tok"] + sp.PLE_STREAM_MIB) / 1024.0
            / (TOK_MS / 1000.0), 2)
        # inverse: served fraction an SSD-bound LRU-like system would need
        # at the sim's assumed 7.4 GB/s to hit 13.0 tok/s
        budget_mb = SSD_ASSUMED * 1024.0 * (TOK_MS / 1000.0) - sp.PLE_STREAM_MIB
        need_sync = budget_mb
        miss_frac = need_sync / (sp.L * sp.K * sp.EXPERT_MIB)
        p["served_needed_at_ssd7.4"] = round(1 - miss_frac, 3)
    out["policies_at_cap143"] = policies

    # --- gap decomposition vs the README's headline -------------------
    # README sim row: 29.1 tok/s at cap 128 (48 GB M4 Max tier, pinned).
    out["headline_gap"] = {
        "sim_pinned_cap128_tps_48gb_tier": 29.1,
        "measured_omlx_cap143_tps": MEASURED_TPS,
        "ratio": round(29.1 / MEASURED_TPS, 2),
    }

    # --- PLE sensitivity -----------------------------------------------
    # sim charges 0.3 MiB/token for PLE gathers; the real table is
    # ~29.8 GiB streamed from SSD on every lookup. What per-token PLE
    # traffic would make LRU@143 exactly 13.0 at 7.4 GB/s?
    lru_sync = policies["lru"]["sync_mb_per_tok"]
    budget_mb = SSD_ASSUMED * 1024.0 * (TOK_MS / 1000.0) - lru_sync
    out["ple_sensitivity"] = {
        "sim_assumes_mb_per_tok": sp.PLE_STREAM_MIB,
        "ple_mb_per_tok_that_reconciles_lru@7.4gbps": round(max(budget_mb, 0.0), 1),
        "note": "PLE n-gram table is ~29.8 GiB on SSD; per-token gather "
                "size is unmeasured — candidate instrumentation target.",
    }

    outp = ROOT / "results/fidelity_santa_cruz.json"
    outp.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("wrote", outp)


if __name__ == "__main__":
    main()
