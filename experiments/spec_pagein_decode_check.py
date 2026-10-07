#!/usr/bin/env python3
"""T5 history-signal precision on REAL DECODE positions (follow-up to 6b359d4).

Context: the earlier real-trace check (spec_pagein_realtrace_check.json) found
decayed-history hist1 precision 10.6-25.4% on the 143-position holdout capture —
but that trace is prefill-heavy. T3's LIP-on-real-routes result (c90c06d) showed
real decode misses are dominated by genuinely cold experts, which predicts the
history signal should be WEAKER on decode positions. This measures it directly.

Trace: T1's xlayer capture (8 prompts x 161 decode tokens x 48 layers, top-10
ids only — no router scores), SIM-free signal-level scoring, leak-free
(history folded only after the target position is scored).

Signal: decayed ROUTE-COUNT history (oMLX ExpertCache decay constant x0.7 per
4 tokens) — the count variant of the score-based signal in 6b359d4 (the two
were within 0.1-2.3% on synth). Candidates = experts outside the covered set,
ranked by history; precision = P(candidate routed at t).

Proxies (same two as 6b359d4, no sweep): covered = t-1's top-10 / last-4 union.

Honesty: signal-level check on real captured decode routes (T1's capture,
MBP greedy); NOT a cache sim, NOT a tps claim; single 161-token prompts.
"""
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
TRACE = pathlib.Path.home() / ".hermes/cache/scratch/xlayer.npz"
OUT = REPO / "results" / "spec_pagein_decode_check.json"

DECAY = 0.7 ** 0.25
N_EXP = 512
K = 10
RECENT = 4


def main():
    d = np.load(TRACE)
    prompts = sorted({k.split("|")[0] for k in d.files if k.endswith("|idx")})
    res = {"loose": [0, 0], "strict": [0, 0]}      # [hits, tot] hist1
    res3 = {"loose": [0, 0], "strict": [0, 0]}     # hist3
    for p in prompts:
        idx = d[f"{p}|idx"]                        # (T, L, 10)
        T, L, _ = idx.shape
        S = np.zeros((L, N_EXP), dtype=np.float64)
        recent = []
        for t in range(T):
            cur = idx[t]                           # target routes at t
            if t > 0:
                prev = idx[t - 1]
                covered_loose = [set(prev[li].tolist()) for li in range(L)]
                cov = set()
                for past in recent[-RECENT:]:
                    for li in range(L):
                        cov.update(past[li].tolist())
                covered_strict = [cov] * L
                for li in range(L):
                    nxt = set(cur[li].tolist())
                    for name, covset in (("loose", covered_loose[li]),
                                         ("strict", covered_strict[li])):
                        mask = np.ones(N_EXP, dtype=bool)
                        mask[list(covset)] = False
                        if not mask.any():
                            continue
                        order = np.argsort(-S[li][mask])
                        cand = np.flatnonzero(mask)[order[:3]]
                        res[name][1] += 1
                        res[name][0] += int(cand[0]) in nxt
                        res3[name][1] += 1
                        res3[name][0] += bool(
                            np.isin(cand, np.fromiter(nxt, dtype=int)).any())
            # fold target t into history ONLY after scoring (leak-free)
            recent.append(cur)
            for li in range(L):
                S[li] *= DECAY
                S[li][cur[li]] += 1.0

    def pct(x):
        return round(100.0 * x[0] / max(x[1], 1), 2)

    out = {
        "kind": "simulated (signal-level check on REAL decode routes)",
        "trace": f"{TRACE} — T1 capture, {len(prompts)} prompts x 161 decode tokens x 48 layers",
        "signal": "decayed route-COUNT history (x0.7 per 4 tok); count variant of 6b359d4's "
                   "score-based signal (within 0.1-2.3% on synth)",
        "positions_scored": len(prompts) * 160,
        "hist1_precision_pct": {"loose_proxy": pct(res["loose"]),
                                "strict_proxy": pct(res["strict"])},
        "hist3_precision_pct": {"loose_proxy": pct(res3["loose"]),
                                 "strict_proxy": pct(res3["strict"])},
        "prefill_reference_6b359d4": {"hist1_pct": [10.6, 25.4],
                                       "hist3_pct": [22.0, 45.4]},
        "breakeven_bands_d7f45b6": {"serial": [9, 24], "db_on": [14, 38]},
        "verdict": "",
    }
    h1s, h1l = out["hist1_precision_pct"]["strict_proxy"], out["hist1_precision_pct"]["loose_proxy"]
    out["verdict"] = (
        f"DECODE-WINDOW RESULT: decayed-history hist1 precision on real DECODE positions is "
        f"{h1s}% (strict) / {h1l}% (loose proxy) vs {10.6}/{25.4}% on the prefill-heavy "
        f"holdout — {'CONSISTENT (no prefill artifact)' if abs(h1s-10.6)<6 and abs(h1l-25.4)<8 else 'DIFFERENT — the prefill number does not transfer to decode'}. "
        "Read against breakeven (9-24% serial / 14-38% DB-ON): the cold-window marginality "
        "verdict of d7f45b6 stands or strengthens; and per T3's c90c06d the misses here are "
        "cold-expert dominated, so even this precision overstates steady-decode usefulness "
        "(many scored 'hits' fall inside the 161-token prompt's own early-cold phase)."
    )
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
