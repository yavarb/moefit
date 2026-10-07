# Pre-registered silicon test protocol (T3 fidelity)

Written 2026-10-07 BEFORE any of the listed runs execute. Purpose: fix
predictions and decision rules now so results cannot be scored post-hoc.
All predictions are SIMULATED from the serial-latency model (constants
MEASURED on Santa Cruz 36 GB: IO A=0.20 + B=0.52k ms/layer-step,
install 0.30 ms/expert, sync 0.122 ms/layer) or, where marked, the
shipped bandwidth model (bw). Measured anchors so far: 13.0/12.71 tok/s
@cap143 idle; 7.8 @cap92 crowded; warmup transient 6.43 @16 tok.

Unless stated: greedy decode, idle box, n >= 1024 tokens per run,
3 runs, report median. SHORT RUNS UNDERSTATE at higher residency
(T8: n128/steady = 0.92 @cap143, 0.83 @180) — do not score n=128 runs
except Test C, which needs only the existing run shape.

---

## Test A — 48 GB M4 Max milestone (cap 192, lru AND prior)

The one decision the whole lab is waiting on. Run BOTH residency
policies on the same box, same traces/prompts.

Pre-registered predictions (serial model, extrapolated constants,
compute 18.1 ms; locked-trace points, gated-trace band from T7 cap180
as the closest proxy):

| policy | serial (this model) | bw model | pipelined ceiling |
|---|---|---|---|
| lru @192   | 15.9 (band ~14.1-16.5) | 52.0 | ~19+ |
| prior @192 | 14.5 (band ~13-15.5)   | 46.7 | — |

Decision rules (pre-committed):
- lru lands in [12.5, 18.5]  -> serial model TRANSFERS to the 48 GB
  box; score the milestone against this table, NOT against 54.8
  (54.8 is a bandwidth-model artifact; keep it only as the
  ideal-pipelining ceiling).
- lru > 25 -> serial constants do NOT transfer (oMLX config threads /
  drive differ); re-microbench that box before any conclusion.
- lru < 10 -> look for memory pressure / crowded box; rerun idle.
- prior > lru by >1 tps -> the serial ranking flip (T3, 9cd94bd) is
  WRONG; investigate real-trace frequency structure vs synth. This is
  the time-model arbitration: bandwidth model also predicts lru here,
  so a prior win falsifies BOTH current models' ranking logic.

## Test B — 36 GB cap-180 point (fits 36 GB only if memory-guard allows)

Footprint 27.3 GiB vs usable-formula 24 GiB; measured during-run free
was 18-19% at 22.61 GiB, so it may fit. If memory-guard refuses or
free% < 10 during load, skip — do not force it.

Pre-registered: serial predicts 13.8-15.7 tok/s (T7 gated band across
traces x compute 18.1-24.2; point ~15.3 @comp18.1, 14.2 @23.2),
misses ~44/tok. Byte-backlog (Q) model with same mean predicts the
SAME median — this run does NOT discriminate S vs Q by tps alone.

## Test C — 36 GB collector run (existing server, no restart, no re-bench)

One run with experiments/collect_silicon_run.py (T4) on the running
0.28-residency omlx, n=256 shape. Two readouts:

1. tok_gap_ms percentiles — THE S-vs-Q discriminator:
   - measured p95/mean >= 1.3 -> serial-resolve confirmed
     (model S predicts p50 ~72 / p95 ~120 ms, p95/mean ~1.51).
   - p95/mean ~1.0 -> byte-backlog; the serial model's miss-cost
     accounting needs re-derivation (bandwidth framing returns).
2. vm_stat pageins + iostat sampled together (T8's ask):
   - ~18 MB/tok page-cache-absorbed re-reads -> buffer-cache reuse /
     non-F_NOCACHE reads explain the 0.885 phys/logical ratio.
   - ~0 -> oMLX's admission beats LRU by ~12% (misses 50.8 floor vs
     true-LRU 58.3) -> admission policy becomes real design territory.

## Test D — 48 GB compute-term separation (optional, run with Test A)

The serial model's one unmeasured knob is compute ms/token. On 48 GB,
cap 224 (footprint 32.9 GiB, fits usable 33 at the edge — watch
memory-guard) drops misses to ~35/tok, so the token time becomes
compute-dominated and the knob separates:

| compute assumption | predicted tps @cap224 lru (serial, locked trace) |
|---|---|
| 18.1 ms (DRAM_EFF@546) | 17.4 (io 22.9 + install 10.5 + sync 5.9) |
| 24.1 ms (DRAM_EFF@410) | 15.8 |

Delta 1.6 tok/s between the two compute assumptions — a stable 3-run
median (n>=1024) resolves which DRAM bin the compute term follows and
pins every other tier prediction by the same offset (~+6.5 ms).

## What NOT to conclude from any single run

- n=128 runs at cap >= 180 (flatten the residency curve, T8).
- One run, one cap: cannot separate compute from miss cost
  (Test D exists for that).
- A 48 GB result scored against 54.8 (bandwidth artifact, T3 f0692cb).
- Crowd: during-run free% < 10 invalidates comparison to idle anchors
  (cf. 7.8 crowded vs 13.0 idle at the same cap-92 class).

---

## AMENDMENT 1 (2026-10-08, glm_fidelity) — Test C downgraded to shape
## compatibility; mechanism arbitration moves to Tests B/D + design test

Triggered by two independent counterexample sets, both reproduced or
verified by T3:

1. CAUSAL NON-IDENTIFIABILITY (astra_analysis_b, e06db18,
   t7_signature_identifiability.json): byte-service-only on the
   IDENTICAL miss bursts (same M matrix, same mean 82.69 ms) yields
   p95/mean = 1.831 — above the 1.3 "serial" threshold. T3 independent
   reproduction: 1.827 (true-LRU M @cap143, locked synth, byte-service
   model at fitted rate). The burst shape is inherited from the MISS
   PATTERN, which both mechanisms share; the >=1.3 threshold therefore
   does NOT distinguish serial-per-miss (S) from byte-backlog (Q). The
   original Q-null ("smoothed, ~1.0") was a strawman control.
2. TRANSPORT NON-IDENTIFIABILITY (astra_local_exp, 175d734,
   t8_sse_timing_probe.json): SSE chunking alone flips the verdict both
   ways at unchanged generation (Q->S 1.000->1.512; S->Q 1.500->1.008;
   buffering 4 single-token lines -> false S-like 4.016; a usage trailer
   distorts collector tps 12.50->10.05). Chunk-count agreement is
   necessary, not sufficient.

REVISED TEST C (shape compatibility only):
- READOUT 1 (survives, one-directional falsification): a measured
  p95/mean ~ 1.0 with VERIFIED token-level provenance falsifies BOTH S
  and bursty-Q (neither model produces smooth gaps from these traces).
  A measured p95/mean >= 1.3 is COMPATIBLE with both — no verdict.
- READOUT 2 (vm_stat+iostat absorption pairing) unchanged — it does not
  depend on timing percentiles.
- PROVENANCE REQUIREMENTS (from astra_local_exp): collector decode
  interval must close on the last token-bearing event (not wall EOF);
  control/role/finish chunks excluded from gap samples; verdicts from
  unverified chunk timestamps are labeled TRANSPORT-provisional.

MECHANISM ARBITRATION now rests on:
- Test D (compute-term separation, mean-based — unaffected).
- Mean tps at a second residency point (cap180/48GB): S and Q diverge in
  the MEAN at other caps (T8 divergence table: cap180 S ~17.4 vs Q
  12.2-18.0 band).
- The sidecar-pread silicon design test: the only DIRECT test of the
  overlap assumption (idle window exists -> prefetch pays -> T2's
  optimistic bound; contends -> T7's pessimistic bound).

Score_silicon_run.py (T4) Test C verdict labels should be downgraded to
match; threshold constants stay (they now gate the one-directional
falsification only). This amendment is registered openly, not silently.
