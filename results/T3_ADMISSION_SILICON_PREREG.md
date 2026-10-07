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

## ON-arm patch — REVISED after reading the installed source (v1-feasible)

The prereg's original patch shape ("read and use without installing") is
INFEASIBLE in installed v1: `ensure` must write expert bytes into a slot
before the forward gathers via `map`; there is no transient-buffer compute
path. Verified by reading /opt/homebrew/Cellar/omlx/0.7.0/.../
moe_expert_offload.py on Santa Cruz (md5 d420b305, matches
design_inventor's read).

The v1-implementable equivalent is LIP insertion (Qureshi ISCA'07
LRU-Insertion-Policy): a first-lifetime-miss expert still installs, but is
inserted at the LRU position (next eviction victim) instead of MRU; recurrent
misses insert MRU. Same pollution protection (one-shots get evicted first,
the resident core is protected), purely a slot_of ordering change.

SIM re-measure of the LIP arm (results/design_admission_lip.json, locked
synth, cap143, both windows):
- OFF (true-LRU): 57.400 global / 47.751 suffix; serial 11.96 / 13.28 tps.
- ON (LIP): 57.588 global (+0.19) / 46.460 suffix (-1.29); serial
  11.93 / 13.42 tps. Best T3 eviction result to date, and the global-window
  penalty of the use-without-install forms (+1.14) is nearly eliminated.

### Honest silicon prediction (LIP arm, supersedes the number above)

Single-prompt n=1024 regime: global-window semantics dominate → predicted
measured delta **-0.05 .. +0.25 tps** on the 15.55 baseline. WASH remains the
pre-committed expected outcome; the decision rules below are unchanged and
apply verbatim to the LIP arm.

### The silicon diff (3 lines in ExpertCache, installed v1 file)

In `__init__` add: `self._miss_hist = {}` (and the same line in any
cache-reset path if present).
In `_ensure_ids`'s miss branch, after `self.misses += 1` and before/after the
`_install(e, ...)` call, wrap with (env-gated, default OFF):

    if _ADMISSION := os.environ.get("OMLX_ADMISSION") == "1":
        first = self._miss_hist.get(e, 0) == 0
        self._miss_hist[e] = self._miss_hist.get(e, 0) + 1
    ...
    slot = self._install(e, ...)   # unchanged
    if _ADMISSION and first and not self.free:
        # demote to LRU position: move slot_of's newest entry to the front
        self.slot_of[e] = self.slot_of.pop(e)  # (already newest) — instead:
        # OrderedDict semantics: delete any e-entry and re-insert FIRST:
        self.slot_of.pop(e, None); items = [(e, slot)] + list(self.slot_of.items()); self.slot_of.clear(); self.slot_of.update(items)

(slot_of is a plain dict in v1; insertion order IS the LRU order because
hits re-insert via pop+assign, so the demotion is a front-insert — the exact
expression to be finalized against the live miss-loop by the patch owner;
T9/T2 hold the restart window.)

## Cost of running it

3 A/B/A pairs x 1024 tokens at ~15.5 tok/s ≈ 6 x 66s ≈ 7 minutes of decode,
inside the already-approved shared restart window. The counter log line T2
requested (per-request hits/misses) should land in the same patch pass.
