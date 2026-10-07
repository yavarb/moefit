# T6 cycle 2 | astra_analysis_a | Astra gpt-6

Fixed-cap143 local analysis. No new silicon measurements, parameter grid, or policy invention.

## Independent evidence versus circular inference

The 57.4 logical misses/token cited as measured in recent notebook entries is a synthetic LRU replay statistic, not an independent hardware counter in the cited SSD record.

Evidence chain:

- `results/measured_santa_cruz_36gb_ssd.json` measures 140.5 physical MB/token, 12.71 tok/s, disk IO counts and sizes. It contains no logical expert-miss counter.
- `experiments/gap_santa_cruz.py` generates an LRU miss matrix from synthetic holdout routes and discards the first 200 tokens.
- `results/design_omlx_exact.json` explicitly labels its LRU 57.41 and decayed-count 56.91 misses/token as simulated.
- `results/t8_pagecache_capacity_probe.json` derives the approximate 0.885 physical/logical ratio using simulated 57.4 misses/token and expert size 2.764892578125 MB.

Numerical identity verified by the audit:

    ratio = 140.5 / (57.4 * 2.764892578125) = 0.885291244587
    recovered = 140.5 / ratio / 2.764892578125 = 57.4

Recovering 57.4 by inverting that ratio is circular. It cannot validate the decayed-count replay against independent measured logical misses. The source-read policy identification and independently measured throughput/microbench observations remain useful; only the claimed independent miss-count agreement is unsupported by these cited artifacts.

Physical reads correspond to 50.815717 full-expert equivalents/token. This is not a logical miss measurement: page-cache hits, read amplification, other IO, and byte attribution must be resolved before such a conversion can be interpreted. This audit does not claim no other counter could exist; it establishes what the cited evidence actually contains.

## Matched replay windows

All following rows replay the same SHA256-locked holdout, cap143, using T2's existing implementations. Values are misses/token, not silicon measurements.

| Cache lifecycle | Evaluation window | Tokens | LRU | Decayed count | Miss reduction |
|---|---|---:|---:|---:|---:|
| Retained across prompts | All tokens | 2882 | 58.2811 | 57.7873 | 0.847% |
| Retained across prompts | Drop first200 globally (T2) | 2682 | 57.4072 | 56.9109 | 0.864% |
| Retained across prompts | Drop first128 of each prompt | 1218 | 47.7791 | 46.9007 | 1.839% |
| Reset each prompt | All tokens | 2882 | 64.3237 | 63.7790 | 0.847% |
| Reset each prompt | Drop first200 globally | 2682 | 63.6003 | 63.0444 | 0.874% |
| Reset each prompt | Drop first128 of each prompt | 1218 | 47.7800 | 46.9146 | 1.811% |

The often-quoted pair 58.3 versus 56.9 mixes different evaluation windows. The correct T2 pair is 57.4072 versus 56.9109. On all 13 individual prompt suffixes, decayed-count has fewer misses than LRU; reductions range from 1.255% to 2.319%. This is a descriptive result on these traces, not a significance claim or a guarantee on real routes. '<1%' applies to the aggregate global-window comparison, not uniformly to prompt-local steady windows.

The lifecycle convergence from cycle1 largely survives the policy refinement: matched suffix decayed-count means are 46.9007 retained versus 46.9146 reset. No inference about untested policies or silicon ranking is made here.

## Reproduction and ownership

Run `python3 experiments/analysis_t6_policy_windows.py`.

`results/analysis_t6_policy_windows.json` records input/source SHA256, exact window names, token counts, per-prompt results, and the evidence identity. Checks pass: LRU miss-mask cross-check against independent replay on all48 layers; T2 rounded anchors 57.41/56.91; ratio-inversion identity; input/source hash stability.

Dependency caveat: `experiments/design_omlx_exact.py` is T2-owned uncommitted WIP in the shared tree at execution time. This audit records its hash and uses it read-only; it does not commit or modify another agent's work. The audit validates agreement with that implementation, not independent fidelity of every detail to deployed oMLX source. T2 should commit the dependency when its silicon priority permits.

Handoff T3/T5: replace measured-miss language with simulated logical misses; distinguish matched global and per-prompt windows. Handoff T1/T2/T4: paired logical expert-miss counters and physical read bytes are still required for an independent absorption estimate. No changes requested to the measured throughput anchors.
