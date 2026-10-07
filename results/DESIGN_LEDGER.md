# Design ledger: paging/hot-set mechanisms, with numbers (2026-10-07)

Durable ledger for design experiments. Every entry states its cost model,
baseline, and label (SIM = simulated on synthetic traces; MEASURED = real
silicon). Verdicts are keep/drop with numbers — no "follow-up" hedging.

Cost model (adopted from lead_silicon's measured serial-latency model,
`results/gap_santa_cruz.json`, commit 463c856): per token
`compute + 0.20·(layer steps with a miss) + (0.52+0.30)·misses/token +
0.122·48` ms, with compute assumed (18–23 ms). Validated against three
measured points (12.1 vs 12.7–13.0 tok/s @cap143; 8.9 vs 7.8 @cap92; cold
warmup 5.0 vs 6.43) — see `results/MEASURED_VS_SIM_36GB.md`.
Key implication: under this model, miss COUNT is worth ~0.82 ms each and
miss-free layer steps are worth ~0.32 ms each; bandwidth-model tok/s is not
a valid ranking metric on SSD-bound tiers.

## Entry 1 — admission/eviction policy study (SIM)

`experiments/design_admission_cache.py` → `results/design_admission_cache.json`
(untracked when first folded here; Bélády-style "opt" is clairvoyant).
LRU replay baseline under the serial cost model, matched constants:

| cap | LRU baseline (SIM tps) | oracle (clairvoyant) | oracle gain | ARC | TinyLFU | LFUDA |
|---|---|---|---|---|---|---|
| 92 | 8.88 | 13.39 | **+51%** | +4.6% | +2.6% | −14% |
| 143 | 12.08 | 17.02 | **+41%** | +0.0% | −6.6% | −18% |
| 192 | 14.69 | 19.91 | **+36%** | −1.5% | −12.5% | −16% |

Verdicts:

- **Eviction heuristics: DROP.** Every tested online policy (ARC, TinyLFU,
  LFUDA) is at or below the plain-LRU baseline at cap ≥ 143. There is no
  implementable win left in eviction-order tuning on this workload.
- **The headroom is real but needs prediction.** The clairvoyant bound says
  36–51% is available, and it comes from knowing *which* experts the next
  tokens need — i.e. route knowledge (sidecar/probe-style prediction),
  which also feeds cross-layer pipelining. Effort belongs there, not in
  cache-replacement heuristics.
- **Prerequisite (retro item 2, now quantified):** the shipped LRU does not
  refresh hits; under the serial model the fix alone is worth
  10.04 → 11.99 tps @cap143 (comp 23.2 ms) / 12.77 (comp 18.1 ms) vs
  measured 12.7–13.0 (`results/t8_policy_arbitration_cap92_143.json`, SIM).
  All future policy comparisons must run on the hit-refresh-fixed LRU —
  gains measured against the buggy baseline are partly the bugfix, not the
  design.
- **Prefetch-credit note (T8, SIM): while serial miss-resolution dominates,
  prefetching extra bytes alone does nothing; only designs that remove
  misses from the serial path (or overlap layers) can win.**

## Method notes for all design rows

- Traces: `results/traces_synth` (synthetic; served@143 spans 0.767–0.897
  across calibration-matching variants — quote bands, not points). Real
  router traces are absent from the lab checkout.
- Residency sweeps on silicon need n ≥ 1024 tokens per run (short benches
  understate steady state more as cap rises: n128/steady ≈ 0.92 @143,
  0.69 @220).
- Silicon ceilings for context (SIM, measured constants): ideal cross-layer
  overlap ≈ 18.6 tok/s @cap143; compute floor alone ≈ 55 tok/s @36 GB tier.

## Provenance

- Entry 1 numbers read directly from
  `results/design_admission_cache.json` (owner: design_inventor/T2, SIM,
  serial-latency cost model, synth holdout, sha f8262a526f8c); oracle row is
  the clairvoyant farthest-next-use bound, not an implementable policy.
- Arbitration rows from `results/t8_policy_arbitration_cap92_143.json`
  (sheryl_local_exp/T8, SIM).
- Cost-model validation and measured anchors: see
  `results/MEASURED_VS_SIM_36GB.md` and notebook entries 2026-10-06/07.
