# T3 second-chance admission — silicon ON/OFF pre-registration

Written BEFORE any silicon ON/OFF run. Claim owner: glm_fidelity (T3).
Baseline: Santa Cruz, cap143, n=1024 greedy, median 15.55 tok/s (design_inventor a853292).

## What the SIM claim actually is (corrected for the silicon policy)

The cycle-21 WIN (a601dea) beat DECAYED-COUNT at caps 92/143. The silicon box
runs oMLX 0.7.0 = TRUE LRU with hit-refresh (design_inventor source read,
md5 d420b305). Re-measured on the TRUE-LRU arm
(results/design_admission_lru_siliconarm.json, locked synth, cap143):

- OFF (true-LRU + current-call-protected eviction): 57.151 miss/tok global,
  47.504 suffix; serial 11.98 / 13.32 tps.
- ON  (second-chance admission + cache-fullness warmup bypass, same eviction):
  58.292 global (+1.14, WORSE), 46.922 suffix (-0.58, better); serial
  11.84 / 13.41 tps.

## Honest silicon prediction

The n=1024 single-prompt greedy bench has NO cross-prompt transitions — the
regime where admission gating pays (steady multi-prompt suffix window,
-0.58 miss/tok) is not what the bench measures, and the regime it does
resemble (early-window/global) is where ON is slightly WORSE (+1.14 miss/tok).

PREDICTED measured delta at cap143 n=1024: **+0.03 .. +0.10 tps** (from the
suffix-window gain scaled to the measured 15.55 baseline, 13.32->13.41 tps
= +0.67% => 15.55 -> ~15.62), with the early-window cost possibly making the
first ~200 tokens slightly slower.

## Pre-committed decision rules

- ON - OFF >= +0.30 tps (median over >=3 same-prompt A/B/A pairs, n>=1024,
  collector blobs, fail-closed scorer gates): TRANSFERS (would be a surprise —
  outside the sim band; would constrain the real one-shot-miss fraction).
- |ON - OFF| < 0.30 tps: **WASH — the expected outcome.** The sim claim
  (+0.5% steady-state, multi-prompt) does NOT scale to a measurable
  single-prompt effect. The honest verdict for the Chief is: the second-chance
  gain is a cross-workload effect that the standard bench cannot resolve.
- ON < OFF by > 0.30 tps: the admission gate's re-fetch cost is realized
  worse than sim — falsifies the single-prompt transfer outright.

## ON-arm patch shape (installed oMLX 0.7.0, TRUE-LRU ExpertCache)

In ExpertCache._ensure_ids (installed 0.7.0: slot_of is an LRU-ordered dict,
hits re-insert, victim = next(iter(slot_of))):

- Track per-expert lifetime miss count in a dict alongside the cache
  (self._miss_hist, allocated in __init__, cleared in _allocate).
- On a miss: increment _miss_hist[e]. If _miss_hist[e] >= 2 OR the cache is
  not yet full (len(slot_of) < capacity): install as today (read + slot
  write + LRU insertion). If _miss_hist[e] == 1 AND cache full: read the
  expert's tensors and USE them for this call but do NOT insert into
  slot_of (no eviction, no install) — the read result is consumed directly
  by the call path and dropped.
- Env-gated: OMLX_ADMISSION=1 enables; default OFF preserves stock behavior
  bit-for-bit.

Caveat: written against design_inventor's source read of the installed file;
the exact call path for "use without installing" needs verification against
the installed moe_expert_offload.py at apply time (whoever holds the restart
window: T9/silicon_dbuf + T2). The patch is ~15 lines and orthogonal to the
T9 double-buffer toggle and the T2 sidecar flag.

## Cost of running it

3 A/B/A pairs x 1024 tokens at ~15.5 tok/s ≈ 6 x 66s ≈ 7 minutes of decode,
inside the already-approved shared restart window. The counter log line T2
requested (per-request hits/misses) should land in the same patch pass.
