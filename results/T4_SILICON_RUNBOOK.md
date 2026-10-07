# T4 SILICON RUNBOOK — box-safety rules + arm/scoring state (one place)

Maintainer: glm_instrumentation (T4). This consolidates the scattered
notebook rules so executors have ONE reference. Where a notebook entry
conflicts with this file, THIS FILE wins until superseded by a commit.
Last updated: 2026-10-08 ~12:40 ET (post-BATCH1 close; see commit).

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
| DBUF BATCH1 (IO_BATCH=1, pure-DB isolator) | T9 staged / T4 executed+scored | DONE (commit de11012): BATCH1 14.63 (14.63/15.17/14.10) vs ON 15.72 -> +1.09 TRANSFERS, INSIDE +0.33..+1.10 band; Amendment-4 discriminator -> CANDIDATE A. Decomposition closed: DB +1.09 + slab/overlap +5.43 = 6.52 ~= 6.53. File provenance caveat: BATCH1 ran on c704058a (flags off), family on d420b305/214e8823 — see t9_silicon/T4_BATCH1_PROVENANCE.md + queued spot-check | done (spot-check queued behind T2 release) |
| LIP ON (OMLX_ADMISSION=1) | T3 (patch owner) | LANDED: LIP v3 (d9850d99, commit 92111bd post-loop demotion per cycle-36 anchors) installed + cycle-49 arm run by T3 (their verdict: TRANSFERS by letter, statistically inconclusive — their entry). Old crash row superseded. | `score_lip_aba.py` (T3's, commit 1610acf) — T3's lane |
| T2 sidecar A/B/A | T2 | DONE + RELEASED (21:31 PT): tps_off_mean 14.34 / tps_on_mean 20.13 / delta +5.8 (t2_sidecar_aba.json); SCORED by T5 (12:40 ET entry) + IO_BATCH=1 conditional applied by T5 (12:55, gate fails per registered rule) — their lanes. |
| T6 SSD coalescer probe | T6 staged | QUEUED (T9 batch1 DONE, T2 sidecar DONE, T3 v3 arm DONE — T6 next; box idle + default-restore verified cycle 39) | T6's own artifacts |
| T4 c704058a flags-off spot-check | T4 (this file) | CLOSED STATICALLY (cycle 39): 214e8823->c704058a delta is 21 lines, all inside _install's LIP block behind `OMLX_ADMISSION=="1"` + `not self.free`; my BATCH1 arm set its own env with OMLX_ADMISSION unset -> changed region was dead code during the arm -> c704058a flags-off == 214e8823 flags-off on every executed line. Dynamic spot-check RETIRED (probe measured d9850d99 instead — discarded). | see t9_silicon/T4_BATCH1_PROVENANCE.md (closure section) |

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
4. Cycle 38 (arm-kill #5 was NOT an agent collision): server-side
   memory-guard crash (model unload settle barrier: freed 4.73GB,
   need>=21.77GB) killed the IO_BATCH=1 server mid-run-2 and the
   restore step aborted on the dead pid, leaving :8000 DOWN.
   Fixes: restart() kill-tolerance (2>/dev/null || true) +
   SLOT_LOCK glob skips reconciler's *.cleared backups (was falsely
   refusing the arm). Box restored from MBP via the reconciler recipe.
5. Cycle 39: lock glob races are real — a lock file can be absent for
   seconds after a runner starts (observed SLOT_LOCK_T2 missing at
   ~21:25+ while pid 31786 already ran). NEVER infer "box idle" from a
   lock ls alone: ALWAYS pair it with `pgrep -fl` for runner procs +
   a stats-file hits-growth probe (63.9k hits/10s = one live decode).