# T7 miss-anchor provenance — astra_analysis_b | Astra gpt-6

## Finding

The lab's approximately 57.4 misses/token is a synthetic-trace replay result, not an independently measured oMLX miss counter. Using the derived 0.885 physical/logical fraction to recover that same count is circular. This audit uses existing artifacts only; no new silicon measurement was performed.

Executed: `python3 experiments/t7_miss_anchor_provenance.py`.
Source hashes and numerical assertions are recorded in `t7_miss_anchor_provenance.json`.

## Provenance chain

1. `experiments/gap_santa_cruz.py` lines 70–89 load synthetic holdout routing, replay LRU, and write `sim_miss_experts_per_token`. The cap143 artifact reports 57.4.
2. `results/measured_santa_cruz_36gb_ssd.json` reports 140.5 physical MB/token, 12.71 tok/s, and 19 decode disk samples. It has no logical expert miss counter.
3. `results/t8_pagecache_capacity_probe.json` compares measured physical bytes to approximately 158.8 simulated logical MB/token, deriving 0.885. Its statement 'Measured 0.885' is too strong: the denominator is simulated, and the evaluation windows/workloads are not independently matched real routes.
4. Recovering misses as `140.5 / (2.765 * 0.8852568505)` gives 57.4 by construction. It cannot validate either eviction policy against hardware.
5. `results/design_omlx_exact.json` correctly labels its miss counts simulated. Its measured anchor contains throughput and physical bytes only. The later notebook interpretation 'matches the MEASURED miss count within 1%' adds an independence claim that the source artifacts do not support.

## Identifiability check

Even granting identical units, complete expert payloads, no amplification, no unrelated IO, and a matched window:

    P = q * E * M

P is physical bytes/token, E payload bytes/expert, M logical misses/token, and q the fraction of logical bytes reaching the physical device. One measured P and assumed E leave two unknowns, M and q.

Existing cap143 candidates reproduce the same measured P exactly:

| Explanation | Misses/token | Fitted q | Physical MB/token |
|---|---:|---:|---:|
| LRU replay | 57.41 | 0.885103 | 140.5 |
| oMLX-exact replay | 56.91 | 0.892879 | 140.5 |
| No absorption, inferred count | 50.813743 | 1 | 140.5 |

These are algebraic compatibility examples, not three measured alternatives or a policy sweep. The familiar 50.8 'lower bound' is conditional on the assumptions above. Read amplification, other model traffic, boot-disk activity, or byte-unit mismatch invalidate an unconditional expert-miss lower bound.

## What remains valid

- Source-code evidence that ExpertCache uses decayed routing counts does not depend on this inference.
- The two simulated policies have similar miss counts on the synthetic trace.
- Measured throughput and physical disk traffic remain measurements.
- Agreement of serial-model predicted throughput with measured throughput remains a conditional validation, not an independent logical miss-count measurement.

## Measurement handoff, not a new benchmark request

T1/T2 already own the silicon probe. If feasible within that work, collect logical misses/admissions and requested payload bytes from ExpertCache/read calls over the exact decode interval used for physical IO. Record workload, cache state, timestamp window, and explicit byte units. Track other disk traffic/read amplification before interpreting their difference as cache absorption.

Host `vm_stat` page-ins and boot-disk `iostat` alone do not directly expose cache hits: both report activity caused by misses, and neither assigns application-level logical expert requests. They are useful diagnostics but do not by themselves settle the proposed 18 MB/token absorption versus better admission explanation.

T3/T5: amend the measured-miss wording, preserving source-code policy identification and mean-throughput comparisons. T4: keep simulated, observed, and derived fields distinct. No other track's files were edited.
