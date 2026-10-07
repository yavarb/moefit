# Measured vs simulated: Santa Cruz 36 GB paging (updated 2026-10-06, cycle 2)

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
| decode throughput, crowded box | 7.8 tok/s | measured — commit f20bdd3 (superseded as baseline; kept as the crowding datapoint: +5.2 tok/s / +67% from idling alone) |
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

## The gap, decomposed

Sim roofline at cap 143 LRU: compute 18.1 ms + SSD stream 27.3 ms → 36.6
tok/s, limiter = stream. Measured: 76.9 ms/token. **49.6 ms/token is
unexplained by both roofline terms** (`results/gap_report_santacruz36_cap143_lru.json`).
If silicon were compute-bound at the sim's byte traffic it would imply
68.9 GB/s effective DRAM vs the 293 GB/s the 128 GB measurement supports —
implausible, so the box is SSD-bound and the miss is in the SSD-side model.
Three candidate reconciliations (`results/fidelity_santa_cruz.json`):

| hypothesis | knob value that closes the gap | verdict so far |
|---|---|---|
| effective random-read SSD BW ≪ sequential spec | **~2.6–3.0 GB/s** (vs 7.4 assumed); at 3.0 GB/s the sim predicts 14.8 tok/s vs 13.0 measured | plausible — sim's 7.4 is a sequential-stream number; paging does random ~2.7 MiB reads. Not measured on Santa Cruz |
| unmeasured PLE SSD traffic (gathers/round-trips) | **~376 MB/token** of PLE reads (sim charges 0.3 MB/token) reconciles at 7.4 GB/s | plausible — competing explanation; needs a byte counter |
| oMLX residency is static top-frequency, not hot-set | static first-143 predicts **8.2 tok/s** — undershoots 13.0 by 37%, so measured behavior sits between static and LRU/hot-set | partial — policy mismatch alone cannot explain the gap |

They are not mutually exclusive; none is measured yet. One cheap silicon
probe disambiguates the first two: random 2.7 MiB chunk read bandwidth from
the expert file, plus per-token SSD bytes during decode.

## Resolution: the SSD counters landed (measured 2026-10-06 ~23:50)

`results/measured_santa_cruz_36gb_ssd.json` (iostat -d on the boot disk,
physical reads only, sampled inside a 256-token decode window, same oMLX
process) measures, on silicon:

| quantity | measured | sim assumed |
|---|---|---|
| decode throughput | **12.71 tok/s** (78.68 ms/token) | 36.6 predicted (LRU@143) |
| disk throughput during decode | **1785.7 MB/s** | 7400 MB/s (sequential spec) |
| physical SSD reads | **140.5 MB/token** | 207 MB/token (sync term) |
| IO pattern | **10,746 IOPS at ~171 KB/IO** | assumed ~2.7 MiB chunk streams |

Arithmetic: 140.5 MB/token ÷ 1785.7 MB/s = **78.7 ms/token of pure disk
time = the measured 78.68 ms/token exactly** (disk 100% busy across the
decode window). The compute term (18 ms) is fully hidden/overlapped.
**The gap is resolved: decode is 100% SSD-transfer-bound, and both SSD-side
sim knobs were wrong in opposite directions** —

- effective bandwidth: **1.79 GB/s**, 24% of the 7.4 GB/s the tier assumes;
- traffic: **140.5 MB/token**, 32% *below* the sim's 207 MB/token sync term.

The IO pattern explains the bandwidth: ~171 KB average IOs (not 2.7 MiB
streams) — each missed expert enters as ~16 page-sized random reads.
Measured IOs/token (845) ≈ implied misses/token (~51) × ~16 reads/expert;
the ~51 expert-misses/token implies an effective hit rate ~0.89 on this
workload, i.e. better than the sim LRU's served 0.84 at the same cap.

Hypothesis verdicts: **H1 confirmed and sharpened** (effective random-read
BW is 1.79 GB/s — even below the 2.6–3.0 GB/s single-knob estimate, because
the IO granularity is ~171 KB, not 2.7 MiB); **H2 rejected for this run**
(measured traffic is *lower* than the sim's, so extra PLE traffic is not
needed to reconcile — though physical-read counters under-count page-cache
hits, so the true miss rate may differ); **H3 partially rejected** (oMLX
residency outperforms sim-LRU at matched cap on this workload, not
undershoots it).

Calibration for the simulator, from measurement: replace the per-tier
sequential `ssd` GB/s with a measured effective random-read throughput
(~1.8 GB/s on Santa Cruz's boot disk at ~171 KB IO), and charge sync
traffic at the measured ~140 MB/token basis. Predicted LRU@143 under that
calibration: 140.5/1785.7 → 12.7 tok/s — matching silicon without any
per-miss overhead term. The earlier "49.6 ms/token unexplained" was an
artifact of the 7.4 GB/s assumption, not hidden runtime overhead.

Caveats: one 256-token run; physical reads only (page-cache hits invisible,
so true traffic may be higher than 140.5 MB/token and hit rate can only be
inferred); page-cache effects may inflate the apparent hit rate on a
re-decode of the same prompt.

## What this changes

- **The gap is measured, not mysterious**: decode is 100% SSD-bound at
  ~1.8 GB/s effective random-read throughput. Any sim row for a paging
  tier using the 7.4 GB/s sequential number is ~3–4× optimistic on the
  stream term; recalibrate before quoting sim tok/s.
- The crowded 7.8 tok/s was not a paging result — idle-box crowding alone
  was +67%. Any bench that does not record free RAM before and during the
  run is not comparable.
- The 128 GB-derived DRAM calibration does not apply to the SSD-bound
  regime (the compute term is fully overlapped here anyway).
- The 54.8 tok/s README row for 48 GB remains simulated and unvalidated;
  if the ~1.8 GB/s effective BW persists on the 48 GB run, it will land
  far below 54.8. Treat the 48 GB run as the calibration test.
- Even a perfect paging policy caps at ~55 tok/s on the 36 GB tier
  (compute floor alone); the tier cannot approach the shipped 48 GB sim
  rows.
- Design lever this exposes: **raising effective SSD throughput matters
  more than eviction-policy tuning** — larger contiguous read granularity
  (closer to whole-expert 2.7 MiB reads instead of ~171 KB page-ins) would
  directly raise the 1.8 GB/s ceiling. A design that batches/coalesces
  misses attacks the measured bottleneck; one that only raises hit rate
  from 0.89 toward 1.0 caps the gain at ~1/0.89 ≈ 1.12×.

## Highest-value next measurements

1. ~~Santa Cruz random 2.7 MiB chunk read bandwidth~~ and ~~per-token SSD
   bytes~~ — **done** (`results/measured_santa_cruz_36gb_ssd.json`):
   1785.7 MB/s at 171 KB IOs, 140.5 MB/token. Remaining question: is the
   1.8 GB/s ceiling inherent to the disk, or caused by oMLX's small
   page-size reads? A synthetic whole-expert-size random-read probe
   (2.7 MiB blocks) separates disk from software.
2. One 48 GB M4 Max paging run — turns the 54.8 row from simulated to
   measured and tests whether the ~1.8 GB/s effective BW (and thus the
   calibrated stream term) predicts it.
3. A run with a longer/different prompt to check the ~0.89 effective hit
   rate is not a page-cache artifact of re-decoding similar prefixes.

Tooling that consumes these: `experiments/gap_report.py` pairs any sim row
with any measured record and emits the decomposition above;
`moefit/metrics.py` defines the RunRecord schema for both sim rows and
measured blobs (per-token SSD `miss_count_per_tok` / `miss_ms_per_tok`
fields are proposed but not yet in the schema);
`experiments/fidelity_santa_cruz.py` regenerates the policy table.

## Provenance

- Measured: commits f20bdd3 (7.8 crowded) and c1ca86b (13.0 idle) on
  yavarb/moefit; raw runs in `results/measured_santa_cruz_36gb.json`.
- Simulated: matched-config runs on synthetic traces (build 3747 / holdout
  2882 rows, 48 layers × 512 experts × top-10 routing). Caveats: synthetic
  traces may overstate served fraction at high caps; real router traces
  are absent from the lab checkout; oMLX's actual resident-selection policy
  is approximated by lru/prior/sidecar/static variants — none exact.
- Gap analysis: notebook entries by glm_writeups (22:58), glm_fidelity
  (23:10), glm_instrumentation (23:25), 2026-10-06; consolidated here.
