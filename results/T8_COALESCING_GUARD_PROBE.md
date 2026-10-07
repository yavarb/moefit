# T8 guard challenge — astra_local_exp | Astra gpt-6

Reproduce: python3 experiments/t8_coalescing_guard_probe.py

No new silicon run or network call. Source hashes are in
results/t8_coalescing_guard_probe.json. Four deliberately coalesced synthetic
inputs all escape refusal; one clean control also remains unflagged.

## Failure paths

1. Direct counts: 129 chunks each carrying 2 tokens (258 total), gaps 160 ms.
   The normalizer correctly sets per-run coalesced=True, then recomputes the
   aggregate using gaps only, replacing direct evidence with coalesced=False.
   runs_flagged=[0] remains but the consumer ignores it when False.
2. Usage mismatch: 258 authoritative tokens vs 129 deltas, each delta recorded
   as 1 by the collector's assumption. The normalizer never compares these
   counts. Constant 160 ms gaps look clean.
3. Equal-count buffered pairs: 257 authoritative tokens and 257 deltas with
   alternating 160/0 ms gaps. Median is 80 ms, so 160 is not >5*median.
   Uniform 80 ms generation becomes p95/mean=2 without crossing the detector.
4. Aggregation dilution: a definitely flagged 100-gap run (90 near-zero, 10
   500 ms) pooled with 1000 clean 80 ms gaps falls below both fractions.
   The aggregate overwrites the positive per-run evidence. runs_flagged also
   lists the clean run, because sorted(run_co) selects every run key.

These are not simply threshold calibration failures: evidence propagation is
incorrect, and a negative heuristic cannot establish unbuffered delivery.

## Existing measured artifact

Reanalyzed results/silicon_sc36/collect_silicon_run_256x2.json unchanged:

    run 0: 256 usage tokens / 86 deltas / 86 declared chunk tokens
    run 1: 256 usage tokens / 83 deltas / 83 declared chunk tokens

Normalizer emits no tok_gap_coalesced flag. Pooled p95/mean=1.094 reaches the
low-ratio "falsifies BOTH" branch. Current code appends transport-provisional,
so this is NOT an unqualified mechanism verdict; nonetheless, the claimed
coalescing refusal does not occur. The claim that the guard already covers this
measured file needs correction. These are chunk gaps, not token-gap evidence.

## Fixes verified

The collector's last-token-bearing-event endpoint and control exclusion now
pass three independent assertions through the real one_run parser:

- Both ordinary and 5s-delayed-usage fixtures report 12.50 tok/s.
- Both have exactly 256 gaps for 257 single-token text events.
- Ordinary observed mean gap is exactly 80 ms.

The old cycle-1 probe's main routine asserts the trailer bug remains, so it is
now a historical counterexample, not a valid post-fix acceptance suite. This
new probe imports only its reusable run_case fixture and checks fixed behavior.

## T4 handoff

Preserve any positive direct/per-run evidence with a logical OR; do not replace
it with pooled heuristics. Record only actually flagged run IDs. Compare actual
usage completion count against text events where the collection contract is
one event per token; mismatch invalidates that contract, even if it does not
uniquely diagnose transport batching versus suppressed events. Unknown counts
remain unknown. A negative burst detector never verifies backend timing.

No production files or protocol thresholds changed by T8.
