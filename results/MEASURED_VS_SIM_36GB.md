# Measured vs simulated: Santa Cruz 36 GB paging (2026-10-06)

Honest ledger of what is **measured on silicon** and what is **simulated**,
at one matched configuration. Numbers with "measured" are real runs on the
Santa Cruz host (M4 Max, 36 GB, macOS 27.0, oMLX 0.7.0, PLE on SSD,
Qwen3.8-Flash-Next-oQ4e-mtp); numbers with "sim" come from
`experiments/sim_paging.py` on synthetic traces. Nothing here is extrapolated
silicon speed.

## The matched configuration

| quantity | value | source |
|---|---|---|
| resident experts per layer | 143 (fraction 0.28) | measured (`omlx` settings, actual load) |
| resident footprint | 22.61 GB | measured (actual omlx load; free ~18–19% during decode) |
| decode throughput | **13.0 tok/s median** (12.23–13.07, 3 runs, idle box) | measured — `results/measured_santa_cruz_36gb.json`, commit c1ca86b |
| decode throughput, crowded box | 7.8 tok/s | measured — commit f20bdd3 (superseded as baseline; keep as the crowding datapoint) |
| sim tok/s, cap 143, LRU | **36.6 tok/s** (served 0.84, 207 SSD MB/token) | simulated — one-off run, synthetic traces |
| sim tok/s, cap 143, prior-pin + LRU | **31.7 tok/s** (served 0.815, 239 SSD MB/token) | simulated — one-off run, synthetic traces |

Sim tier assumed `dram=546 GB/s` (M4 Max), `ssd=7.4 GB/s`, `usable=24 GiB`.
The one-off runner lives outside the repo (lab scratch, `sim36_matched.py`);
reproduce with `solve_policy(gold, None, prior_rank, 143, mode, 0, spec)` on
`results/traces_synth`.

## The gap, stated plainly

At the same residency, the simulator predicts 31.7–36.6 tok/s and silicon
delivers 13.0. The sim is **2.4–2.8× optimistic** on this tier. Decomposition
per token (sim, LRU row): compute 18.1 ms, SSD stream 27.3 ms → 36.6 tok/s.
Measured 13.0 tok/s = 77 ms/token, i.e. ~50 ms/token unexplained by the sim's
two-term model. Candidate causes, in the order we would instrument them:

1. **Effective SSD bandwidth.** If the run were purely SSD-stream-bound at the
   sim's 207 MB/token, 77 ms/token implies ~2.7 GB/s effective — not the
   7.4 GB/s the tier assumes. Santa Cruz SSD read bandwidth under PLE load
   has not been measured.
2. **Overlap.** The sim assumes async prefetch hides behind compute; oMLX
   0.7.0 may serialize part of the PLE traffic with compute.
3. **Traffic.** Real SSD MB/token may exceed 207 (PLE read amplification,
   re-reads, paging granularity) — needs a byte counter at the SSD layer.
4. **Usable memory.** Free RAM was 18–19% during decode; the sim's 24 GiB
   "usable" tier is optimistic for a 36 GB box also running the OS + server.

Also note the sim's 18.1 ms compute term alone caps the tier at ~55 tok/s —
so even a perfect paging policy cannot approach the shipped 48 GB sim rows
on 36 GB silicon. The 54.8 tok/s README row for 48 GB remains simulated and
unvalidated; this 13.0 measurement is the only paging number on real
silicon so far, and it does not support extrapolating the sim table to 36 GB.

## What this changes

- The prior crowded 7.8 tok/s was not a paging-policy result; idle-box
  crowding alone was worth +5.2 tok/s (+67%). Any bench that does not record
  free RAM before and during the run is not comparable.
- Simulator calibration against the 128 GB full-fit point (57.4 tok/s,
  293 GB/s effective DRAM) does not transfer to the SSD-bound regime; the
  fidelity work must calibrate the SSD term against a measured 36 GB run.
- Highest-value next measurements (each converts a sim assumption to a
  number): SSD effective GB/s during decode; SSD bytes/token via counters;
  residency sweep on silicon at fixed prompt to test the served-fraction
  curve.

## Provenance

- Measured: commits f20bdd3 (7.8 crowded) and c1ca86b (13.0 idle) on
  yavarb/moefit; raw runs in `results/measured_santa_cruz_36gb.json`.
- Simulated: matched-config one-off (this doc, 2026-10-06), synthetic traces
  from `results/traces_synth` (build 3747 / holdout 2882 rows, 48 layers ×
  512 experts × top-10 routing). oMLX's static top-frequency resident set is
  approximated by `lru` (hard capacity) and `prior` (half pinned); neither is
  an exact model of oMLX 0.7.0's policy.
