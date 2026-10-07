"""T4: pre-registered ON/OFF silicon prediction for the double-buffer /
staged-install design (for silicon_dbuf T9 to test on Santa Cruz).

Fixed BEFORE any T9 run executes (same discipline as T3's protocol):
the expected gain band, the measurement protocol, and the decision
rules are committed first; T9's measured result is then scored against
this sheet, not post-hoc rationalized.

All inputs are MEASURED values from the lab; the two structural
brackets (missing-step fraction of misses, install candidate) are named
bounds, not a tuning grid.

usage: python3 experiments/dbuf_silicon_prereg.py
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# MEASURED inputs (no invention):
# - n=1024 idle baseline on Santa Cruz @0.28 (design_inventor a853292):
#   15.55 tok/s median, 102.6 MB/tok physical SSD.
BASE_TPS = 15.55
BASE_MB_PER_TOK = 102.6
BASE_MS = 1000 / BASE_TPS            # 64.31 ms/tok
EXPERT_MB = 2.765
# - install constants (measured): eval_each 0.18, batched 0.28
#   (T2 silicon probe), older microbench 0.30 (lead_silicon).
INSTALL_CANDIDATES = {"eval_each_0.18": 0.18, "batched_0.28": 0.28,
                      "microbench_0.30": 0.30}
# - full-expert-equivalent misses from physical bytes; absorption 0-6%
#   (T3-corrected interval) bounds logical misses above it.
MISSES_FLOOR = BASE_MB_PER_TOK / EXPERT_MB       # 37.1 (absorption 0)
MISSES_CEIL = MISSES_FLOOR / (1 - 0.06)          # 39.5 (absorption 6%)
# - structural bracket: fraction of misses that are SINGLE-miss steps
#   (synth cap143: missing_steps/misses = 0.569; real decode may
#   differ - bracket [0.4, 0.8] as named structural bounds).
STEPS_FRAC_BOUNDS = (0.4, 0.8)

out = dict(
    kind="pre-registered prediction", design="within-layer double-buffer / "
    "staged expert install (T4, commits 56fdb43/4475d05)",
    owner="T4 glm_instrumentation; silicon execution T9 silicon_dbuf",
    measured_inputs=dict(base_tps_n1024=BASE_TPS,
                         base_ms_per_tok=round(BASE_MS, 2),
                         base_mb_per_tok=BASE_MB_PER_TOK,
                         expert_mb=EXPERT_MB,
                         logical_misses_interval=[round(MISSES_FLOOR, 1),
                                                   round(MISSES_CEIL, 1)],
                         steps_frac_bounds=STEPS_FRAC_BOUNDS,
                         note="all inputs measured on Santa Cruz; the "
                              "steps-fraction bracket is structural "
                              "(synth 0.569), not tuned"),
    mechanism_delta=("ON (staged): exposed install = inst * missing-"
                     "steps; OFF (stock oMLX): inst * logical misses. "
                     "Predicted saving = inst * misses * "
                     "(1 - steps_frac)."),
    predictions={},
    protocol=dict(
        runs="A/B/A matched pairs, >=3 pairs, SAME prompt each pair "
             "(prompt variance is 10.2-13.1 tok/s at n=256 - the "
             "matched design cancels it)",
        n="n>=1024 per run (T8 ramp: n=256 understates; anti-rule: "
           "n<1024 FORBIDS scoring)",
        blobs="collector blobs with runs[*].tokens + finish_reason + "
              "stream_integrity (score_silicon_run.py gates)",
        idle="idle box; report GPU util; no oMLX restart between arms",
        cite="every number cites prompt + residency"),
    decision_rules=[
        "ON beats OFF by >= 0.30 tps (median over >=3 matched pairs, "
        "same-prompt deltas) -> staged install TRANSFERS to silicon.",
        "|delta| < 0.30 tps -> WASH: consistent with the small-install "
        "regime (0.18 ms/expert makes the theoretical band +0.3..+1.1); "
        "NOT a falsification of the mechanism - repeat at a colder "
        "cache state where k/step is larger before concluding.",
        "ON < OFF by > 0.30 tps -> the install stream CONTENTS with "
        "compute/demand IO: falsifies the zero-contention assumption "
        "of the staged model (T2's contention probe measured 0.08-0.2 "
        "ms/expert interference for background IO - the realized cost "
        "may eat the saving).",
        "Any arm with n<1024, missing counts, or integrity problems: "
        "SCORING REFUSED (the fail-closed scorer enforces this "
        "mechanically).",
    ],
)
for iname, inst in INSTALL_CANDIDATES.items():
    s_lo = inst * MISSES_FLOOR * (1 - STEPS_FRAC_BOUNDS[1])
    s_hi = inst * MISSES_CEIL * (1 - STEPS_FRAC_BOUNDS[0])
    t_lo, t_hi = 1000 / (BASE_MS - s_lo), 1000 / (BASE_MS - s_hi)
    out["predictions"][iname] = dict(
        saving_ms_lo=round(s_lo, 2), saving_ms_hi=round(s_hi, 2),
        on_tps_band=[round(t_lo, 2), round(t_hi, 2)],
        delta_tps_band=[round(t_lo - BASE_TPS, 2),
                        round(t_hi - BASE_TPS, 2)])
out["notes"] = [
    "Prediction band at the best-measured install (0.18): +0.3..+1.1 "
    "tok/s on a 15.55 baseline - SMALL; a decisive ON/OFF test needs "
    "matched A/B/A pairs, and a null result there does not falsify "
    "the mechanism (the same patch under sidecar-prefetch bursts is "
    "predicted worth +7.6 tps, commit 4475d05).",
    "Honest caveat: the steps-fraction on real n=1024 decode is "
    "unmeasured; the [0.4, 0.8] bracket is the band over which the "
    "prediction holds, and a measured ON/OFF delta outside the band "
    "constrains the steps-fraction itself - either outcome is "
    "informative.",
    "No silicon run of this design exists; all numbers above are "
    "predictions from measured constants.",
]
print(json.dumps(out, indent=1))
p = ROOT / "results/t4_dbuf_silicon_prereg.json"
p.write_text(json.dumps(out, indent=1) + "\n")
print("wrote", p, file=sys.stderr)
