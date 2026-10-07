# Design ledger: paging/hot-set mechanisms, with numbers (2026-10-07)

Durable ledger for design experiments. Every entry states its cost model,
baseline, and label (SIM = simulated on synthetic traces; MEASURED = real
silicon). Verdicts are keep/drop with numbers — no "follow-up" hedging.

Cost model (adopted from lead_silicon's measured serial-latency model,
`results/gap_santa_cruz.json`, commit 463c856): per token
`compute + 0.20·(layer steps with a miss) + (0.52+0.30)·misses/token +
0.122·48` ms, with compute assumed (18–23 ms). Validated against three
measured points (12.1 vs 12.7–13.0 tok/s @cap143; 8.9 vs 7.8 @cap92; cold
warmup 5.0 vs 6.43) — see `results/MEASURED_VS_SIM_36GB.md`.
Key implication: under this model, miss COUNT is worth ~0.82 ms each and
miss-free layer steps are worth ~0.32 ms each; bandwidth-model tok/s is not
a valid ranking metric on SSD-bound tiers.

## Entry 1 — admission/eviction policy study (SIM)

`experiments/design_admission_cache.py` → `results/design_admission_cache.json`
(untracked when first folded here; Bélády-style "opt" is clairvoyant).
LRU replay baseline under the serial cost model, matched constants:

| cap | LRU baseline (SIM tps) | oracle (clairvoyant) | oracle gain | ARC | TinyLFU | LFUDA |
|---|---|---|---|---|---|---|
| 92 | 8.88 | 13.39 | **+51%** | +4.6% | +2.6% | −14% |
| 143 | 12.08 | 17.02 | **+41%** | +0.0% | −6.6% | −18% |
| 192 | 14.69 | 19.91 | **+36%** | −1.5% | −12.5% | −16% |

Verdicts:

- **Eviction heuristics: DROP.** Every tested online policy (ARC, TinyLFU,
  LFUDA) is at or below the plain-LRU baseline at cap ≥ 143. There is no
  implementable win left in eviction-order tuning on this workload.
- **The headroom is real but needs prediction.** The clairvoyant bound says
  36–51% is available, and it comes from knowing *which* experts the next
  tokens need — i.e. route knowledge (sidecar/probe-style prediction),
  which also feeds cross-layer pipelining. Effort belongs there, not in
  cache-replacement heuristics.
- **Prerequisite (retro item 2, now quantified):** the shipped LRU does not
  refresh hits; under the serial model the fix alone is worth
  10.04 → 11.99 tps @cap143 (comp 23.2 ms) / 12.77 (comp 18.1 ms) vs
  measured 12.7–13.0 (`results/t8_policy_arbitration_cap92_143.json`, SIM).
  All future policy comparisons must run on the hit-refresh-fixed LRU —
  gains measured against the buggy baseline are partly the bugfix, not the
  design.
- **Prefetch-credit note (T8, SIM): while serial miss-resolution dominates,
  prefetching extra bytes alone does nothing; only designs that remove
  misses from the serial path (or overlap layers) can win.**

## Entry 2 — mechanism headroom at cap 143 (SIM, exact stack-distance sweep)

`experiments/analysis_t6_traffic_headroom.py` →
`results/analysis_t6_traffic_headroom.json` (owner: sheryl_analysis_a/T6,
committed c358684). Exact reuse-distance sweep on the synth holdout —
served fraction is analytically determined by the reuse-distance
distribution (p50 D=24, p90 170, p99 28821; cross-check: analytic 0.879 =
true-LRU replay 0.879, shipped FIFO-LRU 0.84 undercounts by 3.9pp):

| mechanism (true-LRU basis, cap 143) | served | verdict |
|---|---|---|
| plain true-LRU | 0.879 | baseline |

Baseline note: the true-LRU basis approximates silicon — oMLX 0.7.0's
actual policy is DECAYED-COUNT eviction (source read,
`experiments/design_omlx_exact.py`). Provenance correction (astra T7/T6
audits, `results/T7_MISS_ANCHOR_PROVENANCE.md`): the often-quoted "~57.4
misses/tok" is a SIM replay value, not a measured counter — deriving
0.885 = 140.5/(57.4×2.765) from iostat bytes and inverting it recovers
57.4 circularly. What IS measured: 140.5 MB/tok physical reads, which
correspond to **50.8–54.1 full-expert misses** (absorption 0–5.9%; the
mincore footprint fraction is NOT a request-weighted bound — static sets
of that size cover 0.5–5.9% of misses by selection. Both synth policies
overstate by 6–15%; mincore
confirms no hidden page-cache tier). The policy identification stands on
the source read and throughput agreement, not on a measured miss count;
direct logical-miss counters are still needed. Window caveat (astra T6):
omlx-exact vs true-LRU agreement is <1% on the global replay window but
−1.8% (decayed-count fewer) on matched per-prompt suffixes — fix the
window when comparing.
| static hot-set pinning (0 → 50% of cap) | −2.9pp at any pin fraction | **pinning HURTS** — persist-pin-style designs lose at this cap |
| perfect next-token prefetcher | +12.1pp | **upper bound for ANY prefetcher** — the served-fraction lever caps here |
| sidecar (exact prefix knowledge) | 1.000 | removes all sync misses; only helps if it feeds pipelining |

Miss dynamics for design sizing: 58.3 misses/token mean (p50/p90/p99 =
50/81/188, max 480); SSD MB/token p99 506.7, max 1291, burst p99/mean 3.23 —
bursty, so queue-aware designs matter more than mean-rate ones.

Verdicts:

- **Static pinning: DROP at cap 143** — every pin fraction 0–50% is below
  plain LRU (−0.1 to −2.9pp served). Same story as T8's prior-pin-half
  (+15% misses at both caps): frequency-prior hot-sets do not pay on this
  traffic. Persist-pin / layer-persist-mix WIP designs must be re-based on
  the TRUE-LRU baseline or their comparisons are inflated ~20%.
- **Prefetch: bounded.** Even a perfect next-token predictor buys only
  +12.1pp served — the large wins are in *pipelining the misses that
  remain* (serial model: ~0.82 ms/miss), not in eliminating them.
- Served-vs-cap curve (analytic): 0.734@64, 0.879@143, 0.912@192, 0.975@448
  — diminishing returns past cap ~192 argue against footprint growth as a
  strategy on its own.

**Correction to the stack-distance interpretation** (astra_analysis_a/T6,
`results/analysis_t6_censoring.json`, 2026-10-07): the original reading
of this sweep ("tail unexploitable; eviction-policy headroom structurally
zero") was over-strong. The p99 reuse distance of 28,821 was a COLD
SENTINEL (1.77% compulsory first-touches), not a measured reuse distance —
the true tail is p99=434, max=511, and 78.95% of LRU misses recur later in
the trace. The drop-eviction-heuristics verdict STANDS, but on the entry-1
numbers (every online policy ≤ LRU; Belady bound +41–51% reachable only
with future knowledge), not on any proof that the tail is unexploitable.
Also from that reanalysis: cache lifecycle and evaluation window matter —
a concatenated 13-prompt replay gives 58.28 miss/tok vs 64.32 with
reset-per-prompt caches at cap 143, but a matched-window check shows the
gap is INITIAL-WINDOW dominated, not a steady-state policy effect: on
identical suffixes (first 128 tokens of each prompt trimmed, 1218 tokens)
retained-vs-reset misses differ by exactly 1 (47.779 vs 47.780/tok).
Report the cache-reset policy and window mask with every steady-state
comparison; window selection must not masquerade as a policy effect
(`results/T6_CENSORING_AUDIT.md`, astra_analysis_a).

## Entry 3 — idle-window sidecar prefetch (SIM; first designs to beat baseline)

`experiments/design_idle_prefetch.py` → `results/design_idle_prefetch.json`
(owner: design_inventor/T2, commit 377eb7c). Mechanism: during the
~29.1 ms/token the SSD sits idle (compute+sync), background-read next-token
routes from the routing sidecar — budget 50 experts/token at the measured
4.85 GB/s — inserted as MRU into the same capped LRU, behind a break-even
precision gate (0.366 pessimistic / 0.05 with async install). Serial
cost model, compute 23.2 ms assumed. All numbers SIM on synth holdout
(sha f8262a526f8c), cap 143 unless noted:

| variant | misses/tok | tok/s (install on critical path / async) | vs LRU |
|---|---|---|---|
| LRU baseline | 57.15 | 12.12 | 1.0 |
| sidecar prefetch | 10.12 | 19.02 / 26.0 | +57% / +114% |
| sidecar + Belady eviction | 3.5 | 24.54 / 30.94 | +103% / +155% |

Verdicts:

- **KEEP — this is the first design family to beat the baseline** (the
  Belady-eliminated eviction alone cannot; the win comes from *prediction
  feeding pipelining*, exactly where entries 1–2 said the headroom lives).
  Cap 92: 8.9 → 12.2/14.9 (sidecar), 21.0/28.7 (+Belady); cap 192:
  14.7 → 22.2/29.1, 27.4/33.5.
- **Online predictors: DROP, safely.** decay-freq / co-activation precision
  is 1–3% on synth — far below the 36.6% break-even — so the gate shuts
  them off and they cost 0.998–1.00×. Zero-risk by construction, but also
  zero gain on synth; real traces may differ.
- **Install cost is the second-order lever**: at cap143 sidecar+async,
  pessimistic install is 9.5 ms of a 40.7 ms token (19.0 vs 26.0 tps).
- Standing assumptions to check on silicon: background preads must not
  slow GPU compute or demand reads; prefix-replay regime only (the sidecar
  knows exact routes because the prefix was seen before). Proposed first
  silicon test (needs an oMLX patch, not started): sidecar-driven pread
  thread vs the measured 12.7 tok/s.

**Bracket, not points** (`results/DESIGN_BAND_ARBITRATION.md`, T3, commit
57e825d): every tok/s in this entry is an OPTIMISTIC-overlap bound — T2
hides background reads in the full compute+sync idle window (29.1 ms @
4.85 GB/s). T7's independent re-implementation is the PESSIMISTIC bound
(compute-only window, A-term charged): cap143 sidecar_pess 12.46, sidecar
opt 15.13, Belady pess/opt 19.57/23.28, pipelining-only 14.09. The
committed anchors stay canonical (their LRU baseline is validated against
measured silicon, 12.12 vs 12.7), but quote designs as a bracket
[T7 pessimistic … T2 optimistic] — e.g. sidecar prefetch @cap143 is
12.5–19.0 (pess install) / 15.1–26.0 (async), and the pending silicon
sidecar-pread run decides which bound is real.

## Cross-track synthesis: the design ordering (all SIM, measured constants)

Three independent analyses (T2 idle-window, T8 mechanism probes, T6
stack-distance) converge on one ranking for where the next tok/s comes
from, at cap 143, true-LRU baseline 12.0–12.1:

1. **Install-cost reduction — the gating lever** (T8): every prefetched
   expert pays 0.30 ms install regardless of hit; a real coact predictor
   is net-NEGATIVE until install ≤ ~0.18 ms/expert (16.0–15.4 vs 16.6
   pipelining-only baseline; at install 0 the same predictor yields 22.6).
   Worth ~17 ms/tok directly (0.30 × 58 misses).
2. **Cross-layer pipelining** — +38% (12.0 → 16.6; matches the ~18.6
   ideal-overlap ceiling). Oracle prediction on top breaks that ceiling:
   25.1 tps drive-capped at 4.65 GB/s / 39.1 uncapped, because the
   constraint shifts to the 18.1 ms compute floor.
3. **Prediction** — the largest prize (oracle: served 1.0, misses → 0,
   compute-bound ceiling ~43–55 tok/s) but pays only after 1–2.
4. **Eviction/pinning/admission heuristics — DEAD** (entries 1–2; T6
   pin scan; T8 prior warning).

Also folded this cycle:

- **Policy ranking flip** (glm_fidelity/T3, commit 9cd94bd): the serial
  model is now IN sim_paging (`solve_policy_serial`, `hit_refresh=True` =
  oMLX ExpertCache semantics, `want_misses` hooks). Under it, true-LRU
  beats static-pinned prior at EVERY cap (11.9 vs 10.4 @143; 14.5 vs 13.3
  @192) because prior's misses (72/tok) exceed true-LRU's (57.4) at
  ~0.82 ms each. The carry-forward "prior@≥128 for peak tps (54.8@192)"
  is a bandwidth-model artifact. Discriminator for the 48 GB silicon run:
  race lru@192 vs prior@192 — if prior wins on silicon the serial ranking
  is wrong.
- **Shipped-sim bias table** (T6): FIFO-vs-true-LRU served bias is
  +7.0/+6.4/+4.3/+3.9/+2.4 pp at caps 32/64/128/143/192 — design sims
  below cap 128 on the shipped (no-refresh) sim understate baselines
  most.
- **Probe numerics: KEEP (research-only), policy still loses** (updated):
  T8's "matmul overflow/NaN" was spurious BLAS noise (reproduces in float64
  on healthy data; picks finite end-to-end). Hardened anyway (float64 +
  finiteness guard, tests). Policy verdict unchanged: probe served 0.830
  vs true-LRU 0.879 at cap143. If T2 wants a predictor on synth: coact
  matrix (precision@10 = 0.157, 8.1× uniform lift) beats the PLE probe
  (precision@6 = 0.078, 6.6× lift) — per the retro, probe stays
  research-only. Equivalence guardrail note: the long-failing
  test_sim_equivalence was stale reference constants (EXPERT_MIB 1.46 vs
  2.69), not algorithm drift — restored, 45/45 pass (4a7ebb8).

## Entry 4 — prediction quality on traffic + IO coalescing/layout bounds (SIM)

`experiments/analysis_t6_predict_coalesce.py` +
`results/analysis_t6_predict_coalesce.json` +
`results/analysis_t6_packed_layout.json` (owner: sheryl_analysis_a/T6,
commit 969ec54, SIM on locked synth, measured serial constants, compute
18.1 assumed; notebook note pending at fold time).

| question | number | verdict |
|---|---|---|
| PLE ridge-probe as a next-token predictor | precision 0.002 overall, 1.1% on new installs (coact matrix: 15.7% — **14× better**); 83% of picks re-install already-resident hot experts; serial net −1.4/−2.6 tps at budgets 6/12 | **DROP the probe as a prefetch predictor** — it re-installs the hot core instead of catching misses |
| co-locating co-missed experts (pack for A-batch merging) | clusters/step 1.15; A saving **−0.97 ms/tok** (packing pays A more often than it saves); 12.59 vs 12.74 tps | **DROP packing-for-A-merging at cap143** — only 13.3% of miss steps have ≥2 clusters, so the A*(g−1) saving is structurally tiny |
| perfect A-merging bound (all co-misses in one IO) | max +3.14 ms/tok → 13.42 tps (+4%) | ceiling too small to chase |
| sequential read packing (duty cycle 2×) | 12.74 → **16.83 tps (+31%)**; bounded ~3× on miss bytes (drive microbench 3.8–5.6 GB/s vs 1.79 duty) | **KEEP as the read-path lever** — the win is duty-cycle/sequential-merge, not batch-A |

Context that sharpens earlier entries: T8's "install ≤ 0.18 ms/expert
break-even" assumed a useful predictor; on synth the only candidate with
signal is the coact matrix (15.7% precision@10), so prediction work should
target coact-class signals, real traces (synth has no t→t+1 structure),
and only after install cost drops. The read-path lever is confirmed but
its paying form is sequential/duty-cycle, not the A-batch form I
speculatively ranked in the cycle-2 gap doc — corrected here.

## Method notes for all design rows

- Traces: `results/traces_synth` (synthetic). Trace-shape uncertainty on
  gate-passing variants is small (served@143 0.837–0.846; only gate-failing
  variants spread 0.767–0.897) — quote the gated band. Real
  router traces are absent from the lab checkout.
- Residency sweeps on silicon need n ≥ 1024 tokens per run (short benches
  understate steady state more as cap rises: n128/steady ≈ 0.92 @143,
  0.69 @220).
- Silicon ceilings for context (SIM, measured constants): ideal cross-layer
  overlap ≈ 18.6 tok/s @cap143; compute floor alone ≈ 55 tok/s @36 GB tier.

## Provenance

- Entry 1 numbers read directly from
  `results/design_admission_cache.json` (owner: design_inventor/T2, SIM,
  serial-latency cost model, synth holdout, sha f8262a526f8c); oracle row is
  the clairvoyant farthest-next-use bound, not an implementable policy.
- Entry 2 numbers read from
  `results/analysis_t6_traffic_headroom.json` (owner: sheryl_analysis_a/T6,
  commit c358684, SIM, exact stack-distance sweep + true-LRU/pin/prefetch
  replay at cap 143).
- Entry 3 numbers read from
  `results/design_idle_prefetch.json` (owner: design_inventor/T2, commit
  377eb7c, SIM, serial cost model, sidecar/sidecar_opt/Belady rows).
- Entry 4 numbers read from
  `results/analysis_t6_predict_coalesce.json` and
  `results/analysis_t6_packed_layout.json` (owner: sheryl_analysis_a/T6,
  commit 969ec54, SIM, measured serial constants).
- Cross-track synthesis draws on T8
  `results/t8_pipelining_predictor_probes.json` (SIM), T3
  `results/sim_serial_policy_ranking.json` (commit 9cd94bd), T6 notebook
  bias table, and T4 `experiments/serial_predict.py` (commit 9b89eab).
- Arbitration rows from `results/t8_policy_arbitration_cap92_143.json`
  (sheryl_local_exp/T8, SIM).
- Cost-model validation and measured anchors: see
  `results/MEASURED_VS_SIM_36GB.md` and notebook entries 2026-10-06/07.
