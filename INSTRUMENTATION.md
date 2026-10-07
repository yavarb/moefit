# Instrumentation reference (T4)

How measured and simulated numbers are produced, normalized, and compared
in this repo. Everything below is wired and tested; nothing requires new
plumbing. Units: tok/s = decode tokens/sec; ms/tok = milliseconds per token.

## Canonical schema

`moefit/metrics.py` defines the RunRecord: every sim or silicon result
reduces to `kind` ("sim"|"silicon"), `tps`, `residency`, `footprint_gib`,
`expert_gib_resident`, per-term ms, `source` provenance. `validate_record`
checks it. Measured constants used by the serial model live in
`SERIAL_CONSTANTS_MEASURED` (io A=0.20 + B=0.52*k ms per layer-step,
install 0.30 ms/expert, sync 0.122 ms/layer — measured on Santa Cruz by
lead_silicon, results/gap_santa_cruz.json; compute_ms is the model's one
ASSUMED term).

## Producing measured data (silicon)

- `experiments/collect_silicon_run.py` — stdlib streaming bench against
  any OpenAI-style chat-completions endpoint. Emits the canonical blob
  with per-chunk timestamps + per-token gaps (chunk-coalescing servers
  distort per-token gaps; omlx streams ~1 token/chunk). Handles
  reasoning_content streams.
- `experiments/measure_ssd_per_token.py` (lead_silicon) — iostat
  physical-bytes blob: `decode_disk_MBps`, `decode_disk_MB_per_token`,
  `decode_iops`, `decode_avg_KB_per_io`. NOTE: measured MB/s is a
  duty-cycle average under oMLX serial miss resolution (drive does
  3.8-5.6 GB/s on the same pattern) — not a drive ceiling.

Three blob shapes are accepted by `silicon_record_from_measured`:
runs-based (bench), chunk-timestamped (collector), iostat flat keys.

## Comparing sim vs silicon

```
# bandwidth-model row vs measured record (shows both accountings)
python3 experiments/gap_report.py --sim results/sim_paging_matched_cap143.json \
    --tier 48GB-M4M --mode lru --cap 143 \
    --measured results/measured_santa_cruz_36gb.json

# silicon-realistic serial-model sim side in one command
python3 experiments/gap_report.py --serial --cap 143 --mode lru \
    --tier 36GB-M4M36 --measured results/measured_santa_cruz_36gb_ssd.json
```

`gap_report` emits: tok/s ratio, residency/footprint alignment, sim
roofline terms, sim-spec "unexplained" (labeled as an accounting
artifact of the 7.4 GB/s sequential assumption), measured-SSD accounting
(disk ms/tok vs measured ms/tok, residual), and — when the measured run
carries per-token gaps — the tok-gap percentiles and the S-vs-Q
signature verdict (`serial_signature_check`: p95/mean >= 1.3 -> serial
miss resolution; <= 1.1 -> smoothed byte-backlog; T7 predicts ~1.51 for
S at cap143 true-LRU).

## Pricing designs in silicon-realistic units

```
python3 experiments/serial_predict.py --cap 143 --mode lru --hit-refresh
```

Runs the policy sim with `want_misses=True` on locked traces and prices
the miss matrix through the measured serial constants. Rule of thumb at
cap143: every miss/tok removed is worth ~0.82 ms/tok (0.52 io + 0.30
install). `--hit-refresh` = oMLX ExpertCache semantics (required to
reproduce measured 12.7-13.0 tok/s; no-refresh contrast: 10.56).
Equivalent implementation inside the sim: `sim_paging.solve_policy_serial`.

## Anchors (measured, Santa Cruz 36GB idle)

- 13.0 tok/s median @ residency 0.28 (cap143), footprint 22.61 GB
  (results/measured_santa_cruz_36gb.json, commit c1ca86b).
- 12.71 tok/s n=256, 140.5 MB/tok physical SSD reads @ 1785.7 MB/s,
  10.7k IOPS @ 171 KB/IO (results/measured_santa_cruz_36gb_ssd.json).
- Sim reproduction: serial model + hit-refresh LRU + compute 18.1 ms →
  12.75 tok/s; with the 36GB-bin compute term 24.1 ms → 11.9.
  Compute term on the 36GB box is the remaining unmeasured knob.

## Tests

tests/test_metrics.py (schema, normalizers, both accountings, serial
model math, signature check), tests/test_prior_mode.py,
tests/test_probe_numerics.py.
