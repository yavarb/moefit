# T6 residency-weighting audit | astra_analysis_a | Astra gpt-6

Offline logical counterexamples and fixed-cap143 synthetic traffic analysis. No new silicon measurement; no operating-system cache simulation.

## What the measured snapshot establishes

`results/silicon_sc36/pagecache_expert_residency.json` reports 2.29 GB cached of 71.57 GB expert tables, 683 fully cached expert groups and 1,416 partially cached groups out of 24,576, with a 1.3-second scan. These are measured residency statistics and remain valid evidence of a small cached footprint at scan time.

The source `experiments/pagecache_expert_residency.py` maps shards and obtains mincore residency vectors sequentially. It does not record expert-request frequencies, cache-hit events, or per-request residency at the time of decode. The saved JSON does not contain the resident identities. This is not even a single instantaneous global snapshot: shard observations occur over the scan interval.

It therefore does NOT establish that at most 3.200% of requested bytes were served from the page cache. Footprint and request coverage have different weights:

    footprint fraction = sum(address resident bytes) / sum(all address bytes)
    absorption fraction = sum(requested resident bytes at request time)
                          / sum(all requested bytes)

The quantities coincide under special assumptions, such as uniform requests over addresses and suitable time alignment, not in general. Even uniform random resident-set selection only gives footprint as an EXPECTATION, not an upper bound.

## Executed counterexamples

Using the measured fully-cached count 683 of 24,576 as an equal-expert static-set budget:

- The same resident set covers 100% when all requests target that set, and 0% when requests target its complement. Both fixtures execute and assert their results. They are logical counterexamples, not claims about Santa Cruz behavior.
- Fixed cap143 LRU miss traffic on the locked synthetic holdout contains 167,966 requests. Equal-sized static sets yield:

| Set selection | Fraction of all experts | Heldout miss requests covered |
|---|---:|---:|
| Most missed experts in BUILD trace | 2.779% | 3.829% |
| Least missed experts in BUILD trace | 2.779% | 2.096% |
| Most missed experts in HOLDOUT (oracle) | 2.779% | 5.855% |
| Least missed experts in HOLDOUT (oracle) | 2.779% | 0.497% |

The build-selected set uses no holdout fitting. The oracle rows are only analytical extremes on this finite traffic. Static membership here is not a proposed paging mechanism and does not model UBC eviction, evolving membership, partial-page residency, or overlap with the expert cache. The fixture uses equal-size expert abstractions, not a reconstruction of the measured per-layer resident identities. It establishes that request weighting matters, not that actual page-cache absorption is any particular value.

## Consequences and limits

1. The claimed logical-miss interval [50.8,52.5]/token does not follow from mincore. It is a conditional scenario obtained by imposing absorption <=3.2%, which the footprint observation does not justify.
2. The mincore observation alone cannot choose between routing differences and cache absorption as explanations of lower physical traffic. Conversely, this audit does NOT prove a large hidden cache tier or rehabilitate prior quantitative L2 predictions.
3. Physical full-expert equivalents remain a descriptive conversion; interpreting them as logical misses also requires payload size, physical-byte attribution, amplification, and consistent time windows.
4. The compute-term conclusion cannot cite this footprint-derived interval as a measured bound. T7 separately owns regression statistics; this audit does not re-fit the regression or declare either compute value correct.
5. To estimate absorption, collect logical miss/read payload counters and physical IO in the same window. Request-aligned residency or cache-hit tracing can help; a post-run whole-table footprint cannot replace access-weighted measurements. No extra silicon run is initiated or requested outside T1/T2's existing ownership.

## Reproduction

Run `python3 experiments/analysis_t6_residency_weighting.py`.
Output: `results/analysis_t6_residency_weighting.json`.

SHA256 pins both build/holdout traces and the measured snapshot. Assertions check both endpoint counterexamples, the independent 167,966 holdout-miss baseline, resident-set cardinalities, total miss accounting, source stability, and an exhaustive five-address example showing that uniform random-set EXPECTATION equals footprint. No shared simulator or measurement code changed.
