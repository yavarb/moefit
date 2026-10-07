# T4 DBUF SILICON PATCH — staged-install (double-buffer) ON/OFF path for T9

Owner: glm_instrumentation (T4). Executor: silicon_dbuf (T9), Santa Cruz.
Design: commit 56fdb43 (SIM); pre-registered ON/OFF prediction + decision
rules: `results/t4_dbuf_silicon_prereg.json` (commit ee44fde) — score
T9's measurement against THAT file, not against the sim numbers.

## ADDENDUM (cycle 27, AMENDMENT 2 — supersedes the arm labels below)

T9's source read of the INSTALLED box file (Homebrew 0.7.0, md5
d420b305) found the staged-install machinery is not merely present — it
is ON BY DEFAULT: IO pool default 12 workers, read-ahead window
4*workers=48, decode-overlap path default 1; workers<=1 = the serial
path. Consequences:

- The 15.55 tok/s n=1024 baseline IS the DB-ON arm. ON = stock default
  environment (no vars). OFF = OMLX_MOE_OFFLOAD_IO_WORKERS=1. The A/B/A
  measures the DB's realized contribution by REMOVAL.
- Prereg Amendment 2 (committed BEFORE any A/B/A data): ON pinned at
  15.55; OFF predicted 14.58-15.24 @install 0.18 (ON-OFF = +0.33..+1.10,
  bands unchanged in magnitude). Score against
  results/t4_dbuf_silicon_prereg.json amendment2.
- WASH now has a decisive second reading: if removing the read-ahead
  window costs nothing, install was never on the DB-ON critical path —
  the 0.797 ms/miss slope is fetch-dominated and the serial model's
  install charge (inst x misses = 17.5 ms/tok @0.30) must be re-priced
  toward inst x missing_steps. TRANSFER or WASH both close an open
  model question; there is no uninformative outcome.
- Every pre-existing silicon number in the lab (12.7-15.55 tok/s, the
  0.797 ms/miss regression, the 0.18/0.30 ms install microbenches) was
  measured with DB ON. Comparisons against my cycle-14 SIM baseline
  (11.85 "serial" -> 13.00 "DB") should be read as: silicon has been
  running the DB side all along; the SIM serial arm never described the
  box.

## Step 0 — run the inventory FIRST (read-only, seconds, no restart)

    python3 patches/omlx_dbuf_env_check.py

It locates the live server's `moe_expert_offload.py` (via lsof on the
omlx-server pid, falling back to the Homebrew Cellar path), prints its
md5, and decides:

### CASE B — machinery present (best case: ZERO code changes)

The upstream staged-install implementation (an IO thread pool, a
read-ahead window of `OMLX_MOE_OFFLOAD_IO_BATCH` experts' futures, and
install-as-reads-complete in `_ensure_ids`) is ALREADY in the file,
gated by environment variables. The ON/OFF arms are pure environment —
but per the ADDENDUM above, the box's DEFAULT is ON (workers 12):

    ON  (staged install): DEFAULT env, no vars set (the 15.55 arm)
    OFF (serial):          OMLX_MOE_OFFLOAD_IO_WORKERS=1

Both arms use the SAME binary and SAME file — the toggle is env-only,
which is exactly the clean matched A/B/A the prereg requires.
Mechanism being measured: reads for the next `IO_BATCH` misses are
issued ahead on the pool; each expert's slot-write (install) happens as
its reads complete, so install of expert i overlaps the fetch of expert
i+1 — the within-layer double-buffer of commit 56fdb43, fetch-bound
(the exposed install collapses to ~one latency per missing step).

### CASE A — machinery absent (older fully-serial 0.7.0)

If the inventory reports CASE A, back-port the upstream staged loop.
Source of truth: the omlxenv/omlx-ref `moe_expert_offload.py` (the
windowed `_ensure_ids` + `_read_ahead` + `_submit`/`_drain` + io-pool
block; local reference copy md5 61ead2/891e4 family). Insertion anchors
that design_inventor's box source-read verified exist in the Homebrew
0.7.0 file: `_install(self, e, ...)`, `_ensure_ids`, `self.free`,
`self.slot_of`. Back up the file first (`cp file file.t4bak`), apply,
`python3 -m py_compile file`, and record before/after md5s. The ON arm
is then the same env vars as CASE B; OFF is either unset vars or the
`.t4bak` restore (prefer unset vars — same file, no restore risk).

## Restart procedure (Chief-approved, ONE coordinated window)

1. Coordinate in the notebook: T9 (dbuf toggle) + design_inventor (T2
   sidecar prefetch flag + miss-counter log line) share ONE restart of
   the Santa Cruz omlx-server pid; T3's second-chance arm rides the
   same window if it has a flag. Do NOT restart twice. :8317 untouched.
2. OFF arm: restart with the stock environment; run the collector
   (`experiments/collect_silicon_run.py`, n>=1024, >=3 runs, same
   prompt each pair — collector now emits runs[*].tokens +
   finish_reason + stream_integrity, and the fail-closed scorer
   enforces them).
3. ON arm: restart with `OMLX_MOE_OFFLOAD_IO_WORKERS=12
   OMLX_MOE_OFFLOAD_IO_BATCH=16`; same prompt, same n, same runs.
4. A/B/A: repeat OFF-ON-OFF (>=3 pairs total, alternating arms) so box
   drift cancels; the prereg's decision rule uses the median same-prompt
   delta.
5. Correctness gate: the staged path must not change outputs — greedy
   decode, temperature 0; run one pair of bit-exactness checks (same
   prompt, compare generated text) before timing arms. A mismatched
   generation = the arm is INVALID regardless of tps (the installs are
   byte-identical copies; only their scheduling changes).
6. Memory note: the window holds up to IO_BATCH experts' host payloads
   in flight (~16 x 2.77 MB per layer-step transient — ~44 MB worst
   case), plus pool threads. Trivial against the 36 GB budget; record
   RSS per arm anyway (footprint delta ON-vs-OFF is part of the honest
   result).

## Scoring (mechanical)

ONE COMMAND, fail-closed — run this on the A/B/A blobs:

    python3 experiments/score_dbuf_aba.py --on on1.json [on2.json ...] \
        --off off.json

It enforces eligibility (>=3 runs/arm, actual runs[*].tokens >= 1024 —
max_tokens never gates — finish/integrity clean, matched host/model,
SAME prompt via the collector's new prompt_sha256 field; pre-hash blobs
need --accept-unhashed-prompt) and emits the verdict against
`results/t4_dbuf_silicon_prereg.json` amendment2. Regression-tested
(tests/test_score_dbuf.py, 9 cases incl. the masked 128-token-run
refusal). Score against `results/t4_dbuf_silicon_prereg.json`:
- ON >= OFF + 0.30 tps (median, >=3 same-prompt A/B/A pairs) → staged
  install TRANSFERS to silicon.
- |delta| < 0.30 → WASH: consistent with the small-install regime
  (band +0.33..+1.10 tps at install 0.18) — NOT a falsification; the
  same mechanism under sidecar-prefetch bursts is predicted +7.6 tps
  (composition artifact, commit 4475d05).
- ON < OFF − 0.30 → the staged stream CONTENTS (falsifies the
  zero-contention assumption; T2 measured 0.08–0.2 ms/expert of
  interference for background IO).
- WASH under Amendment 2 additionally measures the serial model's
  install term: no loss when the window is removed => install was never
  on the DB-ON critical path => re-price the model's install charge
  toward inst x missing_steps (see prereg amendment2.wash_reading).
Any arm with n<1024, missing counts, or stream-integrity problems is
SCORING REFUSED by `experiments/score_silicon_run.py` automatically.

## Provenance notes (from lab source reads)

- Santa Cruz: Homebrew oMLX 0.7.0, file md5 d420b305…, true-LRU
  ExpertCache (design_inventor read ON the box).
- The box's fully-serial miss-resolution (lead_silicon microbench:
  A + B*k ms exposed, install 0.18–0.30 ms/expert on the critical
  path) is consistent with EITHER absent machinery OR present-but-
  default-serial (upstream default keeps the serial path unless the
  env var is set). The inventory distinguishes them in one command.
- MBP local copies: omlxenv venv (windowed staged install — mechanism
  reference) and omlx-ref HEAD 79f4488 (decayed-count variant). The
  omlxenv copy's md5 changed during cycle 26 (891e4ca7 vs 61ead2
  earlier) — an external edit to that venv, not by T4; the inventory
  works on either.

No silicon numbers are claimed here; everything above is the patch
path, toggle semantics, and measurement protocol.
