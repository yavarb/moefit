"""T3 fidelity, part 4: what each SHIPPED table row should predict under
the silicon-validated serial-latency model vs the shipped bandwidth model.

Why: the 48 GB silicon milestone is still framed as "confirm 54.8 tok/s
sim @prior/192" (swarm retro carry-forward). That 54.8 is a BANDWIDTH-model
number; the serial model — validated at 3 measured points on Santa Cruz
(cold transient 5.0 vs 6.43, steady cap143 12.1-12.75 vs 12.7-13.0, cap92
8.9 vs 7.8) — prices the same operating point very differently. This
table tells T1/T5 what to actually expect from the 48 GB run, so the
milestone isn't scored against an artifact.

HONEST FLAGS:
- Serial constants (IO A=0.20/B=0.52 ms per layer-step, install 0.30
  ms/expert, sync 0.122 ms/layer) were MEASURED on Santa Cruz (M4 Max
  36 GB, oMLX 0.7.0). Applying them to OTHER tiers is EXTRAPOLATION;
  a 48 GB M4 Max likely has a faster SSD but the serial-resolve
  discipline (oMLX, not the drive) is the same software.
- compute_ms is the model's one unmeasured knob; we use the bandwidth
  model's DRAM term per tier (5438 MiB/tok at DRAM_EFF 293/546).
- All rows are SIMULATED on the locked synth traces (beta 2.0, shift
  0.3, seed 0); served@cap143 band across gate-passing traces is
  0.767-0.897 (T7) -> treat single rows as ~+/-1 tps bands at 143.

usage: python3 experiments/fidelity_tier_predictions.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import sim_paging as sp  # noqa: E402

# shipped README operating points: tier -> (cap, README claim)
POINTS = {
    "24GB-M4":   dict(caps=(64,), readme="11.8 tok/s ceiling (bandwidth model)"),
    "32GB-M4P":  dict(caps=(128,), readme="23.6 -> 25.7 (bandwidth model, LRU->pinned)"),
    "36GB-M4M36": dict(caps=(143,), readme="13.0 MEASURED (oMLX, 0.28 residency)"),
    "48GB-M4M":  dict(caps=(128, 192),
                      readme="29.1/54.8 (bandwidth model) — 48GB milestone target"),
    "64GB-M4X":  dict(caps=(192,), readme="bandwidth model row"),
}


def main():
    bf, bl, bT = sp.load_split(ROOT / "results/traces_synth/build.npz")
    hf, hl, hT = sp.load_split(ROOT / "results/traces_synth/holdout.npz")
    rng = np.random.default_rng(0)
    hs = np.sort(rng.choice(hT, size=min(8000, hT), replace=False))
    gold = np.stack([hl[li][hs] for li in range(sp.L)], axis=1).astype(np.int16)
    prior_rank = {li: np.argsort(-np.bincount(bl[li].ravel(), minlength=sp.E))
                  for li in range(sp.L)}
    picks = np.zeros((len(gold), sp.L, 1), np.int16)

    rows = []
    for tier, info in POINTS.items():
        if tier not in sp.TIERS:
            continue
        spec = sp.TIERS[tier]
        for cap in info["caps"]:
            # bandwidth model (shipped solve_policy, legacy throttle)
            for mode in ("lru", "prior"):
                srv_b, s_mb, a_mb, t_b, c_b, st_b = sp.solve_policy(
                    gold, picks, prior_rank, cap, mode,
                    0 if mode != "prior" else 0, spec, throttle="legacy")
                ser = sp.solve_policy_serial(gold, prior_rank, cap, mode,
                                             spec, hit_refresh=True)
                rows.append(dict(
                    tier=tier, cap=cap, mode=mode,
                    bandwidth_model_tps=round(float(t_b), 1),
                    serial_model_tps=ser["tps"],
                    serial_misses_per_tok=ser["misses_per_tok"],
                    serial_compute_ms=ser["compute_ms"],
                    readme_claim=info["readme"],
                    extrapolated_serial_constants=(tier != "36GB-M4M36"),
                ))
    out = dict(kind="simulated",
               note="serial constants measured on Santa Cruz only; "
                    "non-36GB rows are extrapolations of the oMLX "
                    "serial-resolve discipline, not of the drive. "
                    "compute_ms per tier = bandwidth-model DRAM term.",
               rows=rows)
    p = ROOT / "results/fidelity_tier_predictions.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    for r in rows:
        print(f"{r['tier']:>10} cap={r['cap']:<3} {r['mode']:<5} "
              f"bw={r['bandwidth_model_tps']:>5} serial={r['serial_model_tps']:>5} "
              f"(miss {r['serial_misses_per_tok']}/tok, comp {r['serial_compute_ms']}ms)"
              f"{'  [extrapolated]' if r['extrapolated_serial_constants'] else ''}")
    print("wrote", p)


if __name__ == "__main__":
    main()
