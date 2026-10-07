# Measured vs simulated: Santa Cruz 36 GB paging (updated 2026-10-07, cycle 4)

Honest ledger of what is **measured on silicon** and what is **simulated**, at
one matched configuration, plus the current best reconciliation of the gap.
"Measured" = real runs on the Santa Cruz host (M4 Max, 36 GB, macOS 27.0,
oMLX 0.7.0, PLE on SSD, Qwen3.8-Flash-Next-oQ4e-mtp). "Sim" =
`experiments/sim_paging.py` on synthetic traces (`results/traces_synth`;
real router traces are not in the lab checkout).

## The matched configuration

| quantity | value | source |
|---|---|---|
| resident experts per layer | 143 (fraction 0.28) | measured (`omlx` settings, actual load) |
| resident footprint | 22.61 GB (18.98 GB experts) | measured (free ~18–19% during decode) |
| decode throughput | **13.0 tok/s median** (12.23–13.07, 3 runs, idle box; 76.92 ms/token) | measured — `results/measured_santa_cruz_36gb.json`, commit c1ca86b |
| decode throughput, crowded box | 7.8 tok/s (cap 92, 0.18) | measured — commit f20bdd3 (crowding datapoint: +5.2 tok/s / +67% from idling alone) |
| sim tok/s, cap 143, LRU | **36.6 tok/s** (served 0.84, 207 SSD MB/token) | simulated — `results/sim_paging_matched_cap143.json` |
| sim tok/s, cap 143, prior-pin | **31.7 tok/s** (served 0.815, 239 MB/token) | simulated |
| sim tok/s, cap 143, sidecar | **36.9 tok/s** (served 0.841, 205 MB/token) | simulated |
| sim tok/s, static first-143 (no hot set) | **8.2 tok/s** (served 0.287, 921 MB/token) | simulated |

Sim tier assumed `dram=546 GB/s` peak (M4 Max), `ssd=7.4 GB/s`,
`usable=24 GiB`; DRAM efficiency 293 GB/s effective, calibrated on the 128 GB
full-fit measurement (57.4 tok/s).

## Geometry is validated; throughput is not

- Expert size: sim 2.69 MiB vs oMLX 2.637 MiB (−2.0%).
- Footprint at cap 143: sim 22.63 GiB vs measured 22.61 GB (+0.02).
- Resident experts: 18.98 GB both; residency matched to 0.001.

The sim's byte accounting is right; its time model is not.

One more honesty note on the headline "2.8× optimistic" ratio: how much
is trace uncertainty? T7's gated sweep
(`results/t7_gap_uncertainty.json`, SIMULATED) regenerated seven synth
variants and scored them against the shipped served-table gate: the five
that PASS the gate (mean |Δ| ≤ 0.03) all predict 35.9–38.0 tok/s at
cap 143 — a ratio band of **2.76–2.92×** (±0.1×). Only gate-FAILING traces
move it far (1.9–4.2×). So: the gap is not a synth-trace artifact — any
trace matching the shipped served stats leaves the bandwidth-model sim at
least 2.7× optimistic — but quote the ±0.1× band, not the bare point.
(T7's earlier hardware-knob "residency discriminator" predictions were
retracted as built on the superseded bandwidth framing; a replacement,
computed through the serial model, is in progress.)

## The gap, decomposed

Sim roofline at cap 143 LRU: compute 18.1 ms + SSD stream 27.3 ms → 36.6
tok/s, limiter = stream. Measured: 76.9 ms/token — **49.6 ms/token
unexplained by both roofline terms**
(`results/gap_report_santacruz36_cap143_lru.json`). Earlier candidate
reconciliations (`results/fidelity_santa_cruz.json`): (H1) effective
random-read SSD BW ≪ sequential spec; (H2) unmeasured PLE SSD traffic;
(H3) oMLX residency policy mismatch. What actually closed the gap is below.

## Resolution: measured SSD counters + expert-read microbench (2026-10-06/07)

Three silicon measurements (`results/measured_santa_cruz_36gb_ssd.json`,
`results/microbench_expert_reads_santa_cruz.json`,
`results/gap_santa_cruz.json`, commits 463c856 local):

1. **Decode-time SSD counters** (iostat -d, physical reads only, inside a
   256-token decode window, same oMLX process): **12.71 tok/s**, disk
   1785.7 MB/s, **140.5 MB/token**, 10,746 IOPS at ~171 KB/IO, idle 0.03 MB/s.
2. **Expert-read microbench** replaying oMLX's CheckpointExpertStore pattern
   (9 preads/expert, 2.765 MB/expert, 12-thread pool, F_NOCACHE cold): the
   drive sustains **3.8–4.9 GB/s at decode-sized batches** (per-layer-step
   latency ≈ **0.20 + 0.52·k ms** for k misses) and 5.6 GB/s bulk — never
   the 7.4 GB/s the tier assumes, but far above the 1.79 GB/s decode-average.
   Host→slot install costs **0.27–0.35 ms/expert**; per-layer `.tolist()`
   sync 0.12 ms.
3. **Serial-latency model** (LRU replay + the measured constants): at cap
   143 it predicts **12.1 tok/s** vs 13.0/12.7 measured; at cap 92 it
   predicts **8.9 tok/s** vs the 7.8 crowded measurement. The shipped
   bandwidth-overlap model predicted 40.9 / 30.3 — 3.2–3.9× over.

### Correction to the cycle-2 reading of this doc

The earlier claim here — "140.5 MB/tok ÷ 1785.7 MB/s = 78.7 ms/token =
measured token time, so decode is 100% SSD-transfer-bound" — is a
tautology: MB/token ÷ MB/s is *defined* as seconds/token for any run. The
1.79 GB/s disk average is a **duty-cycle** number, not a bandwidth ceiling;
the microbench shows the drive does 3.8–5.6 GB/s on the same pattern. The
real story is **serialization**: oMLX resolves each layer's misses serially
(sync → miss reads → install → compute), and nothing overlaps across
layers. IO is ~half the token; install + sync + compute is the rest.

### Per-token breakdown at cap 143 (serial model, measured constants)

| term | ms/token | basis |
|---|---|---|
| compute | 23.2 | assumed (128 GB measurement BW-scaled 546→410 GB/s) |
| expert IO (57.4 misses/tok, 159 MB/tok) | 36.5 | measured 0.20+0.52k model |
| install (host→slot) | 17.2 | measured 0.27–0.35 ms/expert |
| per-layer sync | 5.9 | measured 0.122 ms × 48 |
| total → 12.1 tok/s | 82.8 | vs 12.7–13.0 measured (−4 to −7%) |

Sim-replay traffic (159 MB/tok) vs measured physical reads (140.5 MB/tok):
physical reads are a lower bound; page-cache absorbs some re-misses.

### Model status (three validations, one verdict)

The serial-latency model is now validated at **three measured points across
two caps plus a cold-start transient**: 12.1 vs 12.7–13.0 tok/s @cap143;
8.9 vs 7.8 @cap92; cold-warmup 5.0 vs 6.43 tok/s
(`experiments/fidelity_serial_validate.py`, commit bb6cf9d). Its one
unmeasured knob is the compute term (spread 12.1–12.9 tps across 18–23 ms —
within run noise; needs a measured compute term on the 36 GB box to pin
down).

Verdict on the shipped bandwidth-overlap model on SSD-bound tiers:
**DROP for silicon prediction — KEEP only as the ideal-overlap
ceiling** (~18.6 tok/s @cap143). Three things the bandwidth model hid:

- The **LRU hit-refresh bug** (shipped LRU does not refresh on reuse): the
  fix moves served 0.840 → 0.879 and misses 76.9 → 58.3/token at cap143 —
  under the bandwidth model predicted tps jumps 36.6 → 48.2, i.e. the bug
  was masking part of the model error (post-fix bandwidth model is 3.8×
  optimistic). All policy comparisons must run on the fixed LRU.
- **Static-first residency is falsified**: keeping experts 0..142 gives
  921 MB/token — over the measured 76.9 ms/token budget even at 7.4 GB/s
  with zero overhead. oMLX's ExpertCache is recency/hotness-based.
- Residual accounting is now closed end-to-end: the gap-report tooling
  (commit 8e77497) pairs the measured iostat blob with sim rows —
  measured-SSD accounting gives 78.7 ms/tok disk vs 78.7 measured,
  **residual −0.0 ms**; traffic ratio measured/sim-logical 0.679.

Two latency models fit the cap-143 ramp equally well (lead_silicon's
serial-resolve vs T8's byte-backlog at fitted BW); the n=16 warmup weakly
favors serial-resolve, and a cap-180, n≥1024 silicon run discriminates
physical-vs-logical traffic accounting (predictions 12.2–18.0, SIM).

Policy reconciliation (T8, SIM under model S): only a **hit-refreshing
LRU** lands on the measured band — shipped no-refresh LRU predicts 10.04
tok/s vs measured 12.7–13.0 (−21%), true-LRU predicts 11.99–12.77. Two
consequences: (a) oMLX's runtime behaves like true-LRU, not the shipped
no-refresh variant (consistent with the 57.4 replay miss count); (b) the
hit-refresh fix is worth ~20% predicted tps — retro item 2 is now
quantitatively justified, and all policy baselines must use the fixed LRU.
Labeling flag to T1: the "plain LRU" replay in `gap_santa_cruz.json` is
true-LRU by behavior; name it accordingly.

**Capstone: the sim now reproduces silicon end-to-end.** The full chain —
policy replay (`hit_refresh=True`, the oMLX ExpertCache semantics) → miss
matrix → measured serial constants (compute 18.1 ms assumed, the one
unmeasured term) → tok/s — predicts **12.75 tok/s** at cap 143 vs measured
12.71–13.07 (`results/serial_lru143_hitrefresh.json`, commit 4b3e56d), and
two independent implementations agree exactly
(`sim_paging.solve_policy_serial` vs `moefit/metrics.serial_model_from_misses`,
breakdown identical 18.1/37.0/17.5/5.9 ms, 58.3 misses/token). With this,
the residual is closed: geometry, policy, and time model all reproduce the
silicon operating point from measured constants. Caveat that remains:
compute ms is assumed (18.1 fits best but is within run noise of 23.2),
and the constants are Santa-Cruz-specific — other machines need their own
microbench.

### Hypothesis verdicts (updated)

- **H1 partially confirmed**: the SSD never delivers 7.4 GB/s on this
  pattern (3.8–4.9 GB/s batch, 5.6 bulk), but the 1.79 GB/s decode average
  is duty cycle, not a bandwidth ceiling. Bandwidth alone does not explain
  the gap.
- **H2 rejected for this run**: measured traffic is *below* sim's, so extra
  PLE traffic is not needed to reconcile.
- **H3 rejected at this cap**: sim-LRU replay misses 57.4/token vs ≥50.8
  measured (physical-read lower bound); oMLX residency is at least
  sim-LRU-like here.
- **New: serialization is the dominant cause**, confirmed against two
  measured operating points (12.1 vs 12.7–13.0 at cap 143; 8.9 vs 7.8 at
  cap 92).

## What this changes

- The shipped sim needs **serial-latency terms** (per-layer-step 0.20+0.52k
  ms, install 0.3 ms/expert, sync 0.12 ms), not just bandwidth: a pure
  bandwidth-overlap model cannot reproduce either measured point, and any
  sim row for a paging tier is 3–4× optimistic until it carries them.
- The crowded 7.8 tok/s was not a paging result — idle-box crowding alone
  was +67%. Any bench that does not record free RAM before and during the
  run is not comparable.
- The 54.8 tok/s README row for 48 GB remains simulated and unvalidated;
  the serial model predicts the 48 GB run will land far below 54.8. Treat
  the 48 GB run as the calibration test of the serial model.
- **Design levers, ranked by the measured breakdown**: cross-layer read
  pipelining (issue layer L+1 reads during layer L compute) and cheaper
  install (batched writes) attack ~53 ms/token of the 83 ms. Hit-rate
  tuning alone caps the gain at ~1/0.88 ≈ 1.14×. Under ideal overlap
  (max of compute, IO+install) the cap-143 ceiling is **~18.6 tok/s
  (SIM, measured constants)** — 1.5× today's silicon.

## Highest-value next measurements

1. ~~SSD bytes/token + effective BW~~ — **done** (see above).
2. Per-layer timing instrumentation inside oMLX (needs a server restart —
   pending a decision; the running 0.28 server and :8317 stay untouched
   until then).
3. One instrumented Santa Cruz run with
   `experiments/collect_silicon_run.py` (emits per-token gap percentiles).
   T7's model-S signature makes this a hard discriminator, not just a
   check: at cap143 the serial model predicts tok-gap p50 71.6 / p95
   119.8 / p99 176.3 ms, **p95/mean ≈ 1.51** (cv 0.296); a byte-backlog
   model with the same mean predicts p95/mean ≈ 1.0. Measured p95/mean
   ≥ 1.3 → serial-resolve; ≈ 1.0 → byte-backlog (SIM predictions,
   `results/t7_serial_band.json`).
4. A cap-180 silicon point if memory allows (~27 GiB footprint; watch the
   memory-guard at 18–19% free). T7's gated band through model S predicts
   **14.0–15.7 tok/s** (n≥1024; SIM, measured constants).
5. One 48 GB M4 Max paging run — the serial model's out-of-sample test,
   and a policy-ranking discriminator: race **lru@192 vs prior@192** —
   the serial model says true-LRU wins (14.5 vs 13.3, SIM); the old
   bandwidth-model carry-forward said prior (54.8). Silicon picks the
   ranking.
6. Method note from the T8 queue model (SIM): residency sweeps on silicon
   need runs of n ≥ 1024 tokens; short 128-token benches understate
   steady-state more as residency rises (n128/steady ≈ 0.92 @cap143,
   0.69 @cap220).
7. Page-cache absorption probe (T8 ask, one 256-token run, no restart):
   measured physical/logical traffic ratio is 0.885 (140.5 vs 158.8
   MB/tok). A two-level replay fits 0.885 only with ~13–16 GiB of
   page-cache absorption — more than the ~5–7 GiB visible spare RAM.
   Pairing vm_stat page-in deltas with iostat during decode separates:
   buffer-cache reuse / non-F_NOCACHE reads (pageins ≈ 18 MB/tok) vs
   oMLX admission beating LRU (pageins ≈ 0, logical misses < 57.4).
   Design consequence: in the pipelined regime, evict-to-L2 alone is
   worth 16.8 → 21.9–24.9 tok/s (L2 = 110–200 experts/layer, SIM), and
   the drive stops binding at ~31 physical misses/token.

Tooling: `experiments/gap_report.py` pairs any sim row with any measured
record; `moefit/metrics.py` is the RunRecord schema (tok-gap p50/p90/p95/p99
supported); `experiments/gap_santa_cruz.py` and
`experiments/microbench_expert_reads.py` regenerate the tables above;
`experiments/fidelity_santa_cruz.py` regenerates the policy table.

## Provenance

- Measured: commits f20bdd3 (7.8 crowded), c1ca86b (13.0 idle), and 463c856
  (SSD counters + microbench + serial-latency model) on yavarb/moefit; raw
  runs in `results/measured_santa_cruz_36gb.json`,
  `results/measured_santa_cruz_36gb_ssd.json`,
  `results/microbench_expert_reads_santa_cruz.json`.
- Simulated: matched-config runs on synthetic traces (build 3747 / holdout
  2882 rows, 48 layers × 512 experts × top-10 routing). Caveats: synthetic
  traces may overstate served fraction at high caps; real router traces
  are absent from the lab checkout; oMLX's actual resident-selection policy
  is approximated by lru/prior/sidecar/static variants — none exact; the
  serial model's compute term is assumed (BW-scaled from the 128 GB
  measurement), all its IO/install/sync constants are measured.
- Gap analysis: notebook entries by glm_writeups (22:58, 23:55, 00:15),
  glm_fidelity (23:10, 00:20 — validation commits bb6cf9d),
  glm_instrumentation (23:25, 23:55, 00:25 — accounting commit 8e77497),
  lead_silicon (00:05, incl. the correction this cycle-3 revision
  incorporates), sheryl_local_exp (T8 model arbitration), 2026-10-06/07;
  consolidated here. Cycle-2 revision of this doc contained the
  tautological "100% SSD-bound" claim; corrected in cycle 3. Design-row
  writeups live in `results/DESIGN_LEDGER.md`.
