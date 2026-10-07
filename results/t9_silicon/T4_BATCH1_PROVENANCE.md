# T4 BATCH1 provenance addendum (companion to results/t9_dbuf_batch1_scored.json, commit de11012)

Registered 2026-10-08 cycle 39, AFTER the BATCH1 arm was scored, in
response to a file-provenance audit triggered by T2's runner md5 log.

## The issue

The three arms of the closed DB-term decomposition were NOT all
collected against the same moe_expert_offload.py file:

| Arm | Blob | Installed file (md5) | Flags |
|-----|------|----------------------|-------|
| ON (A) | results/t9_silicon/t9_on_A.json | d420b305 (stock oMLX 0.7.0) | default env (DB on) |
| OFF (B) | results/t9_silicon/t9_off_B.json | stock-era, IO_WORKERS=1 | script-set |
| ON2 (A) | results/t9_silicon/t9_on_A2.json | 214e8823 (T2 sidecar + T3 LIP merge) | default env, flags OFF |
| BATCH1 (C) | results/t9_silicon/t9_batch1_C.json | c704058a (lip_arm4 "margin-64" + sidecar) | IO_BATCH=1, flags OFF |

- 214e8823 flags-off was verified tps-indistinguishable from stock
  (cycle-30 composition check: 15.64 vs 15.75 medians).
- c704058a flags-off has NOT been verified equivalent to stock. It was
  installed ~21:17 PT by lip_arm4's cp while T9's first batch1 attempt
  was running (T2's 00:55 ET notebook entry flagged it), so the clean
  BATCH1 re-run (21:20-21:24 PT) and the final default restore
  (pid 31205) both ran on c704058a.

## Why this is a caveat, not a refutation

- Both LIP (OMLX_ADMISSION) and sidecar (flag file) are env/flag-gated
  and were OFF during the BATCH1 arm; stats snapshots confirm
  sidecar_on=false and sc_* = 0 throughout (t9_b1_stats0/1.json are
  live deltas: hits 929,408 -> 1,381,873 across the arm window —
  dumper verified LIVE, cycle 39).
- The scored delta (+1.09) sits INSIDE the pre-registered band
  (+0.33..+1.10) and the Amendment-4 discriminator (BATCH1 14.63 >=
  14.2 -> Candidate A) is a threshold call with 0.43 tps of margin on
  the A/B boundary; a plausible flags-off file delta would have to
  exceed that margin to flip the verdict.

## Queued closer (T4, behind T2's sidecar release — do NOT run concurrently)

ONE default-env n=1024 run on the current server (c704058a flags-off,
same prompt family, hash 87eb8e913d26cca9), labeled
PROVENANCE-SPOTCHECK:

    /usr/bin/python3 /tmp/moefit-bench/collect_silicon_run.py \
      --url http://127.0.0.1:8000/v1/chat/completions \
      --model Qwen3.8-Flash-Next-oQ4e-mtp --max-tokens 1024 --runs 1 \
      --label 'T4 provenance spot-check: c704058a flags-off vs ON-family band' \
      --host santacruz --ram-gib 36 \
      --out /tmp/moefit-bench/t4_c704058a_spotcheck.json

Decision rule (pre-registered): tps within [15.52, 16.63] (the ON-family
band) -> c704058a flags-off ~= stock at tps level; caveat closes.
Below band by more than run noise (sd ~0.16-0.8) -> re-examine the
BATCH1 arm against a stock-file rerun. NOT an arm; NOT scored by
score_dbuf_aba.py; single-run supporting evidence only.

## CLOSURE (cycle 39, later same day): SUPERSEDED BY STATIC VERIFICATION — caveat CLOSED

The dynamic spot-check became moot before it ran: while I was probing,
T3 landed LIP v3 (d9850d99, commit 92111bd, demotion-after-loop per the
cycle-36 anchors) and ran their cycle-49 arm; the 21:36:11 PT restart I
observed was their post-arm default restore. My probe (15.78 tps,
1 run, 1024 tokens) therefore measured d9850d99-default, NOT
c704058a — DISCARDED as the registered closer (also: ambient env by
my own runbook rule 3; though T3's restore created it, I did not).

The stronger instrument ran instead — a STATIC gate-diff:

- c704058a content recovered from box staging /tmp/omlx_t2_sidecar_plus_lip.py
  (byte-identical to git 468a6ac) vs 214e8823 (git 7ea2d1e).
- The ENTIRE 214e8823 -> c704058a delta is 21 diff lines, all inside
  _install's LIP insertion block (installed-file lines 494-509), which
  is guarded by `self._admit and self._miss_hist.get(e,0)==1 and not
  self.free`, with `self._admit = os.environ.get("OMLX_ADMISSION","0")=="1"`
  (line 435).
- My BATCH1 arm restart set its own env (`env OMLX_MOE_OFFLOAD_IO_BATCH=1
  nohup omlx serve ...`) from a clean script environment: OMLX_ADMISSION
  unset -> _admit=False -> the entire changed region was DEAD CODE
  during the arm. c704058a flags-off is byte-for-byte 214e8823 flags-off
  ON EVERY EXECUTED LINE of my arm, and 214e8823 flags-off was already
  verified tps-indistinguishable from stock.
- Corroboration: no KeyError crash during the arm (LIP inactive — it
  crashes when active), warm-arm tps (14.63/15.17/14.10) consistent
  with the registered IO_BATCH=1 semantics.

VERDICT: the BATCH1 file-provenance caveat is CLOSED BY CONSTRUCTION.
The scored record (de11012) stands as scored; the ON/OFF family and the
BATCH1 arm are behaviorally same-file (flags-off) for every line each
arm executed. The queued dynamic spot-check is RETIRED.

Incidental (not the registered closer, single run, on T3's post-arm
default restore of d9850d99): 15.78 tps / 1024 tokens / ttft 7.94 s —
inside the ON-family band [15.52, 16.63]. Handing to T3 as a free
flags-off data point for their v3 file (their arm, their
interpretation).