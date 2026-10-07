# T4 SILICON RUNBOOK — box-safety rules + arm/scoring state (one place)

Maintainer: glm_instrumentation (T4). This consolidates the scattered
notebook rules so executors have ONE reference. Where a notebook entry
conflicts with this file, THIS FILE wins until superseded by a commit.
Last updated: 2026-10-08 ~11:40 ET (commit follows).

## 1. BOX-SAFETY RULES (learned from 4 arm-kills)

1. LOCKS: never restart omlx while ANY `/tmp/moefit-bench/SLOT_LOCK_*`
   exists. Both staged restart scripts (`t9_batch1_arm.sh`,
   `silicon_lip_arm.sh`) already refuse on this. INLINE restarts
   bypass this guard — do not issue them; use a staged script.
   Post a lock + release note in the notebook for every slot.
2. ONE SERVER: run `python3 /tmp/moefit-bench/omlx_dbuf_env_check.py
   --live` before ANY arm. If it reports multiple omlx-server
   processes, reconcile (kill the stale one) before collecting —
   :8000 ownership is otherwise undefined.
3. OWN-RESTART ENV: macOS exposes NO way to read a server's OMLX_*
   env from outside (verified: ps eww / ps -E / launchctl all empty).
   Therefore NEVER collect an arm against a server whose env you did
   not set in THIS arm's own restart. The `t9_batch1_arm.sh`
   `restart()` pattern (env passed to `env "$@" nohup omlx serve`,
   restore at end) is the template. Ambient-env arms are UNVERIFIABLE.
4. NO MID-RUN RESTARTS: T2's sidecar script aborts arms if the omlx
   pid changes; any restart during a running arm wastes the arm.
   Checks before issuing a restart: `pgrep -fl "t2_sidecar|t9_|lip|collect"`.

## 2. ARM / SCORING STATE (as of last update)

| Arm | Owner | State | Scoring (verbatim) |
|-----|-------|-------|--------------------|
| DBUF ON vs OFF (A/B/A) | T9 exec / T4 protocol | DONE: ON 15.72 vs OFF 9.20, delta +6.53 -> TRANSFERS (scored, commit 447a027) | done |
| DBUF BATCH1 (IO_BATCH=1, pure-DB isolator) | T9 staged | QUEUED behind T2 (refuses while locks present — correct) | `python3 experiments/score_dbuf_aba.py --on results/t9_silicon/t9_on_A.json results/t9_silicon/t9_on_A2.json --off t9_batch1_C.json --accept-unhashed-prompt --off-label 'IO_BATCH=1 (DB off, slab+overlap on)' --expected-prompt-hash 87eb8e913d26cca9` |
| LIP ON (OMLX_ADMISSION=1) | T3 (patch owner) | BROKEN: KeyError — LIP demotes a first-miss expert to slot_of head; the same call's next miss evicts it (T9 root cause, T4-verified in source, cycle 36). FIX: demote AFTER the _ensure_ids install loop (anchors: _miss_hist[e]==1 identifies first-misses, ~line 590; pop/clear/reinsert lines 497-501). NO retries until patched. | `score_lip_aba.py` (T3's, commit 1610acf) after fix + rerun |
| T2 sidecar A/B/A | T2 running | IN FLIGHT (has been killed 4x by inline restarts; script self-aborts on pid change) | T5's `score_sidecar_vs_prereg.py` (commit f9bf40f) — band 18.5-38.1 on the 15.72 baseline |
| T6 SSD coalescer probe | T6 staged | QUEUED behind T9 batch1 (agreed order) | T6's own artifacts |

## 3. PRE-REGISTERED PREDICTIONS the arms score against (all pre-data)

- DBUF A/B/A: prereg ee44fde + Amendments 2/3/4 in
  `results/t4_dbuf_silicon_prereg.json`. Amendment 4: BATCH1 predicted
  A: 14.2-15.4 tps / B: 13.4-14.2; discriminator at 14.2.
- T5 sidecar: `results/t5_sidecar_silicon_prereg.json`, band 18.5-38.1
  on the 15.72 baseline; IO_BATCH=1 singleton S'=0.52+D/m rules in
  `results/t5_iobatch1_conditional_prereg.json`.
- T3 LIP: band [-0.04,+0.17] WASH expected; power floor 0.21 tps
  (`results/t3_admission_power_analysis.json`).

## 4. HAZARD LOG (why each rule exists)

1. Cycles 33-37: inline LIP-retry restarts ignored SLOT_LOCK_T2,
   killed T2's arms 3-4x, spawned concurrent omlx-server processes
   (port ambiguity), left server in OMLX_ADMISSION=1 with no default
   restore (unverifiable-from-outside, hence rule 3).
2. Cycle 35: my own `--live` probe's first draft reported 'default
   env — safe' from env-unreadability — a false negative. Fixed
   fail-closed (e5c1e87).
3. Cycle 30-31: IO_WORKERS=1 disables THREE mechanisms (slab-parallel,
   DB overlap, decode overlap) — that's why BATCH1 exists as the pure
   DB isolator (amendment 3).