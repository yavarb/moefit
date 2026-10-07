# T6 audit | astra_analysis_a | Astra gpt-6

SIMULATED trace replay, fixed cap143. No new silicon measurements.

Reproduce: `python3 experiments/analysis_t6_censoring.py`.
Machine-readable result: `results/analysis_t6_censoring.json`, including SHA256 of the input trace and exact prompt ordering.

## Correction to predecessor T6 interpretation

The previous 28,821 p99 stack-distance value is a first-touch sentinel, not a measured reuse distance. First references occupy 1.7705% of all references, so p99 necessarily includes this sentinel. With cold references excluded, finite distance p50/p90/p95/p99 = 23/149/255/434 and maximum = 511 (512 experts).

The original cap143 baseline remains correct: 58.2811 misses/token, hit fraction 0.878581. However, 78.9493% of those misses have both a previous reference and an observed later reference. Another 14.4726% are first references with a later observed use. End-of-trace nonrecurrence is right-censored, not proof of no future reuse.

Retract the predecessor claim that the stack-distance curve proves structurally zero eviction-policy headroom or an unexploitable tail. This analysis establishes LRU behavior only; it cannot establish an optimal-replacement bound. T2's separately measured offline Belady headroom is not contradicted, and neither does this audit establish that an online predictor can realize it. Static-pinning measurements remain unchanged.

## Cache lifecycle and matched windows

The standard loader concatenates 13 prompts (2,882 tokens), retaining cache state across prompt boundaries. Resetting the cache at each prompt changes the aggregate misses/token from 58.2811 to 64.3237. These are lifecycle scenarios, not a claim about the server's actual lifecycle.

On the SAME 1,218-token subset, selected by removing the first 128 tokens of EACH prompt, retained-cache and reset-cache replay give 47.779146 and 47.779967 misses/token respectively. The difference is only one miss over that subset. Thus the aggregate lifecycle difference is concentrated in the initial prompt windows, not a persistent steady-state difference.

Removing only the first 128 tokens of the entire concatenated trace instead gives 57.5269 misses/token. Comparing that number to the per-prompt-trimmed 47.7800 would confound window selection with cache policy. Future steady-state reports should record both reset policy and window selection explicitly.

## Verification

A 400-reference randomized fixture validates cold-reference classification, finite distances against direct distinct-count computation, distance-derived misses against OrderedDict LRU, and next-use masks against brute force. Full-trace distance-derived and replay miss masks agree at all 48 layers. Taxonomy counts sum exactly to all misses. SHA256 is checked unchanged before/after analysis. Only the audit script and its result/report are owned by this cycle; no shared simulator edits.
