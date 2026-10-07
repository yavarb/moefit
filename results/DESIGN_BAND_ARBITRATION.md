# Design-band arbitration: why t7_design_band.json diverges from the
# committed T2/T8 anchors (T3, 2026-10-07)

Question (T5 cycle 9, deliberate non-fold): which numbers are canonical —
committed anchors (design_idle_prefetch.json, t8_pipelining_predictor
probes) or T7's re-implementation (t7_design_band.json)?

## Verdict

BOTH are internally consistent; they encode DIFFERENT overlap
assumptions. The committed T2/T8 anchors stay canonical for the design
ledger (mechanism-level cache replay, LRU baseline validated against
measured silicon 12.12 vs 12.7). T7's numbers are the PESSIMISTIC BOUND
of the overlap assumption, and T7's own docstring already disclaims
absolute reproduction ("used for RELATIVE bands across traces, not
absolute design tps"). Quote the pair as a bracket; the silicon
sidecar-pread test (T2's proposal, pending) decides which bound is real.

## The four accounting choices that make up the whole gap

At cap143 sidecar_pess (committed 19.02 tok/s = 52.6 ms/tok vs T7 12.46
= 80.3 ms/tok; gap 27.7 ms/tok):

1. IDLE-WINDOW SIZE (dominant). T2 hides background reads inside the
   compute+sync window: 23.2+5.9 = 29.1 ms at 4.85 GB/s = the 50-expert
   budget exactly. T7 hides only inside compute: 18.1 ms at 4.65 GB/s
   (~32 experts), charging the remaining ~18 experts of read time on
   the critical path. Difference ~10-14 ms/tok.
2. COMPUTE TERM. T2 23.2 ms (BW-scaled 128 GB) vs T7 18.1 ms (best-fit).
   5.1 ms/tok, flagged in both files.
3. A-TERM. T7's prefetch regime drops the 0.20 ms per-layer-step fixed
   cost for layers with sync misses; T2 charges it. ~2-8 ms/tok
   depending on layer-step occupancy.
4. Drive cap 4.85 vs 4.65 GB/s (microbench k=4 vs bulk). ~2-4%.

Items 1-3 sum to the observed 27.7 ms/tok. No numerical bug on either
side; t7_design_band.py's `fit_hit_frac` saturating at 1.0 for the
sidecar anchors is the signature of item 1: its model cannot reach the
committed tps even at perfect prediction because its idle window is
smaller.

## What each file is for (keep both, labeled)

- design_idle_prefetch.json + t8 probes: ABSOLUTE design tps under the
  stated optimistic overlap assumption ("background IO confined to the
  compute+sync window, never delays demand reads" — flagged in its own
  assumptions list). Ledger rows keep these.
- t7_design_band.json: RELATIVE trace-to-trace spread of those designs
  (its per_trace gated entries) and the pessimistic overlap bound.
  Apply its per-trace multipliers to the committed anchors; do not
  quote its absolute tps as design expectations.

## Silicon decider (already pre-registered in spirit)

The T2-proposed oMLX sidecar-pread-thread run measures the real bound:
if background preads during decode leave demand reads untouched, the
committed numbers stand (minus any measured contention); if they
contend, T7's bracket is closer. Score it against BOTH bounds, not one.
