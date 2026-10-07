# T2 sidecar prefetch + per-request counters — patch for installed oMLX 0.7.0

Target: `/opt/homebrew/Cellar/omlx/0.7.0/libexec/lib/python3.11/site-packages/omlx/patches/moe_expert_offload.py`
on Santa Cruz (stock md5 `d420b305ea298654a61811fd8b9ecdf0`, true-LRU ExpertCache).
Patched file: `omlx_t2_sidecar_moe_expert_offload.py` (md5 `da2ecbc313eadf596a7c94f6d9a0402d`), diff: `omlx_t2_sidecar_counters.diff` (~150 lines, additive).

## What it does
1. **Counters (always on, zero behavior change):** every 1 s writes `/tmp/omlx_moe_stats.json`
   with summed `hits, misses` over all 48 ExpertCaches plus sidecar counters. Diff before/after a
   request gives the lab's first MEASURED logical miss count (the open gate on the compute term).
2. **Sidecar replay prefetch (flag file, default OFF):** each layer records its decode route sets.
   A >1 s gap ends a request; the recording becomes the replay script. While
   `/tmp/omlx_sidecar_on` exists, after layer L resolves token t, the script's routes for
   (t+1, L) are pread on a **separate 4-thread pool** (never queued ahead of demand reads).
   Bytes are **held, not installed** (no eviction/pollution, cancel-on-wrong-route for free).
   Next demand call installs from them, so the SSD wait is hidden; install cost stays on the
   critical path (pessimistic-install arm of design_idle_prefetch).
   Alignment: exact script position, else re-align by matching route set; `sc_lost` counts misalignment.
3. Only the decode overlap path (`_forward_overlap`, the default decode path) is hooked.
   Prefill and `_forward_expert_major` are untouched.

Toggle is a flag file, so ON/OFF arms need **no restart**: `touch /tmp/omlx_sidecar_on` / `rm`.

## Local verification (MBP, fake store, real ExpertCache code)
40-step random route script, cap 32: the record pass and the replay-ON pass both give 376 misses
(misses are still counted; their reads are already in flight). sc_issued 365 / used 365 /
wasted 0 / aligned 40 / lost 0. Expert bytes in each slot are checked equal to the expected
content on every call (bit-exact install). The OFF pass after it is unchanged.

## Silicon recipe (inside T9's restore restart — the one Chief-approved window)
1. Copy the patched file over the installed one, keep a backup of stock (`.d420b305.bak`), restart
   omlx with default env (T9's step 3).
2. Warm request (any prompt), then for the SAME prompt P at n=1024, greedy:
   - A: flag OFF -> run (also records script) -> stats diff = measured hits/misses
   - B: `touch /tmp/omlx_sidecar_on` -> run P again (replay regime, exact prefix) -> stats diff
   - A2: `rm` flag -> run P again
   Repeat A/B/A x2 if time allows. Collector: `experiments/collect_silicon_run.py`.
3. Report tok/s A vs B, sc_used/sc_wasted/sc_lost, misses per token. Pre-registered expectation
   (from MEASURED contention 4-11 ms/tok + SIM miss reduction): B in 15.5 -> 18-20 tok/s band;
   WASH (<+0.3) means install/serial term dominates (prefetch hides only the read).
4. Restore stock file at the end (or leave patched with flag OFF — counters are harmless).
