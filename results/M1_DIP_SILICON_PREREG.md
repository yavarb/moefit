# M1 silicon A/B prereg — LRU-insert vs BIP vs DIP (silicon_dbuf, T9)

Registered 2026-10-07 00:45 ET, BEFORE any DIP code or data exists. Executor: silicon_dbuf.
Code owner: design_inventor. This file is the scoring contract; it doesn't change once data lands.

## Gate (no box touch until all hold)
1. design_inventor posts DIP code-green: unit tests for cold, scan, thrash AND the
   same-call-eviction invariant (results/T3_DIP_HANDOFF.md §1), plus the deploy-file md5.
2. The patch reads policy from `/tmp/omlx_insert_policy` (`lru` | `bip` | `dip`; absent = lru =
   stock order) and echoes it in `/tmp/omlx_moe_stats.json` as `insert_policy`. If the
   implementation uses a different toggle, adapt `experiments/m1_dip_aba.py` before running, not after.
3. No SLOT_LOCK_*; queue order respected (current: T2 A3/B3 -> T1 xlayer -> M1).

## Protocol
- `experiments/m1_dip_arm.sh` (guarded): ONE restart, default env (T9 IO pool default,
  OMLX_ADMISSION unset, no sidecar/xlayer flag); policy toggled by the flag file only.
- `experiments/m1_dip_aba.py`: prompt family sha 87eb8e913d26cca9, greedy, n=1024,
  3 rounds of rotated L/B/D (9 runs). Counter diffs are taken per run inside the one process,
  so the BATCH1 cross-process stats bug (b6bd72f) cannot recur.
- Validity per run: tokens >= 1024, finish=length, live policy == requested, server pid unchanged.
  Bit-identical output text across all arms is expected (eviction policy does not change math);
  a text_sha mismatch is reported as a BLOCKER, not scored.

## Readouts and decision rules
PRIMARY = logical misses/token (incl. prefill), median per arm, from same-process counter diffs.
- DIP works as a duel if miss(DIP) <= min(miss(LRU), miss(BIP)) + 0.5.
- BIP/DIP reduces misses if the median is >= 1.0 miss/tok below LRU AND all 3 paired rounds agree in sign.
- Expected (T3's real-route LIP evidence, single-prompt regime): |delta miss| < 1.0 → WASH on misses.
SECONDARY = decode tok/s median per arm.
- 3-run resolution floor ~0.7 tok/s (observed sd ~0.43, T3). |d| < 0.7 = WASH, not a falsification.
- d <= -0.7 for DIP vs LRU = REGRESSION (bookkeeping/duel overhead on the critical path).
- At S ≈ 0.65 ms/miss, 1 miss/tok ≈ 0.10 tok/s, so a tok/s win would need >= ~7 miss/tok saved;
  not expected.

## Not merge-worthy unless
PRIMARY shows >= 1.0 miss/tok reduction AND SECONDARY is not a regression.
