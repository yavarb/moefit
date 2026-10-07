# T3 → M1 handoff: LIP autopsy for DIP implementation
Owner: glm_fidelity (GLM-5.3), T3. Written 2026-10-08 under the LIP freeze — this is the
closed-lane autopsy M1 ("Reproduce T3 KeyError") needs. No silicon was touched writing this.

## 1. The KeyError, reproduced and explained (3 independent confirmations: T3, T9, T4)

Installed silicon file family: oMLX 0.7.0 `moe_expert_offload.py` (stock md5 d420b305).
Cache = `ExpertCache` with `slot_of` (LRU-ordered dict: head = next victim, `next(iter(slot_of))`
in `_reserve`), hit-refresh on access, 48-expert read-ahead window per `_ensure_ids` call.

- v1 (LIP insert-at-LRU-head inside `_install`, Qureshi letter): a first-lifetime miss installs
  its expert at the slot_of HEAD. If the same layer-call misses MORE experts than fit (or the
  call's later installs trigger `_reserve`), the just-installed first-lifetime expert is evicted
  BEFORE `_glu_routes` gathers it → `KeyError: <expert_id>` in `slot_of`, HTTP 500, stream dies
  at 4–7 tokens.
- Decisive pair test (silicon): default env completes the prompt; `OMLX_ADMISSION=1` dies.
- Only bites at full cache (the `not self.free` guard) → steady decode, not prefill.
- v2 (margin-64 insert): survived the verify but WITHDRAWN as unsafe — any call installing
  >64 experts (long prefill) can still evict in-use entries. Never scored.
- v3 (deployed, measured): count first-lifetime installs per call in `_lip_pend` during pass 1
  of `_ensure_ids`; demote them ONCE at the TAIL of `_ensure_ids`, after the last install, when
  nothing evicts before the gathers. Composed md5 d9850d99. Stable: 3×1024 tokens, all
  finish=length.

**Invariant DIP must preserve (the whole lesson): no admission/demotion ordering may make an
expert evictable between its install and its gather within the same `_ensure_ids` call —
including entries installed by the 48-expert read-ahead window. Post-call-tail is the safe
mutation point; per-install mutation is not.**

## 2. Measured silicon numbers DIP should design against (all from committed arms)

- Baseline (stock, DB-ON default env): 15.55–15.72 tok/s median at cap143, n=1024, prompt
  family sha 87eb8e913d26cca9. Prompt-to-prompt variance is real: 10.2–13.1 on other prompts.
- LIP v3 ON vs OFF (only valid arm): +0.34 tps median-of-3 — TRANSFERS by prereg letter,
  INCONCLUSIVE statistically (bootstrap CI95 [−1.16, +0.80]; observed sd ~0.43 → 3-run
  resolution floor ~0.7 tps; ≥8 runs/arm to settle). Do NOT treat LIP as a proven win.
- LIP on REAL short-prompt decode routes (T1 capture, SIM): WASH everywhere (66.44→66.29
  miss/tok baseline). The one-shot-vs-recurrent distinction pays only in warm multi-prompt
  steady state. Expect DIP's single-prompt A/B to be a WASH by regime, not by bug.
- Per-miss value under DB-ON: 0.52 ms/miss registered + MEASURED install exposure 0.124–0.134
  ms/miss (T9's corrected BATCH1 window b6bd72f: D=4.74 ms/tok over m=38.2 miss/tok) →
  **S ≈ 0.65 ms/miss**, inside and settling T5's consistency bound [0.58, 0.71]. A
  1 miss/tok saving ≈ +0.10 tps warm-suffix (LIP's own −1.291 miss/tok × 0.65 ≈ +0.21 tps —
  right at the 3-run power floor); below the 3-run tps floor either way — the MISS/TOK
  COUNTER (T2's per-request hits/misses patch) is the decisive readout, tps is the coarse check.
- Demotion bookkeeping overhead (T9 microbench + my counters): pop-all-reinsert ≈ 4.65 µs
  per demoted entry per call, flat k=1..10; expected k ≈ 0.76 first-lifetime misses per
  layer-call at cap143 (114,610 misses / 48 layers / ~3136 tok) → ~0.17 ms/tok ≈ −0.04 tps
  upper bound. DIP's set-duel leader buckets must stay within this class of cost or be cheaper.
- Cold window: ~97 miss/tok transient, k~2/layer-call → bookkeeping spikes ~0.45 ms/tok briefly.

## 3. v3 semantics DIP-SD can inherit or replace

v3 = "first-lifetime miss installs normally, then demotes to LRU position at call tail."
Qureshi DIP differs per the memo (probationary insert via set-duel: BIP vs LRU leader
buckets, PSEL policy selector). The set-duel machinery is NEW code — v3's `_lip_pend`
(first-lifetime identification via the `_miss_hist == 1` increment at ~line 590 of the
merged file) is directly reusable as DIP's "new install" signal, whatever insert policy the
duel picks. Metadata-create-before-insert (memo's fix) is only needed if DIP goes back to
per-install ordering — v3 shows tail-demotion needs no transient metadata at all.

## 4. Rules for the silicon A/B (from the freeze + runbook, all pre-registered)

- No `OMLX_ADMISSION=1` (or successor flag) retries until the DIP patch is code-green with
  unit tests covering cold/scan/thrash AND the same-call-eviction invariant above.
- Prefetch OFF (T2 sidecar flag off), T9 default IO env; restart only via a guarded staged
  script (SLOT_LOCK_* refusal + verify-gated + EXIT-trap default-env restore). Never inline.
- Same-prompt hash 87eb8e91 family, n≥1024×3/arm minimum; ≥8 runs/arm if you want a
  verdict on tps rather than miss/tok.
- Score with the committed fail-closed scorer pattern (`experiments/score_lip_aba.py` shape:
  actual `runs[*].tokens` gate, band + power reading, counter readout reported not verdict).
- Scoring the tps A/B against anything other than a pre-registered band is post-hoc; register
  first. Given §2, register WASH-expected on tps with miss/tok as primary.

## 5. Open T3 items M1 may absorb

- `_lip_pend`-size counter (replaces the derived k≈0.76 with a counted one) — one line in any
  future patched restart.
- 8-run/arm LIP resolving session — SUPERSEDED by M1: if DIP ships, score DIP instead;
  re-running LIP v3 at n=8 has no independent value once a strictly better admission policy
  is code-green.
