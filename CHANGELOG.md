# Changelog

Newest first. Every fix lists the test that reproduces the original
behaviour.

## Unreleased (review branch `fable-review`)

Measured paging run and agent setup path:

- Idle re-run on M4 Max 36 GB (“Santa Cruz”): oMLX 0.7.0 expert offload at
  0.28 residency (~143 experts/layer), **13.0 tok/s** median decode
  (`results/measured_santa_cruz_36gb.json`, `results/estimate_santa_cruz_36gb.json`).
  Prior crowded run was 7.8 tok/s at 0.18 when other apps held ~16 GB.
  README speed table labels every row measured or simulated.
- `scripts/configure_omlx_paging.py` turns an `estimate.py` verdict into
  oMLX `model_settings.json` (expert offload fraction clamped to the Metal
  cap and, with `--ceiling-gb`, to a live 507 ceiling), links the
  checkpoint into `~/.omlx/models`; `scripts/serve_paging.sh` starts and
  waits for the server; `scripts/bench_decode.py` measures wall-clock
  decode tok/s over the OpenAI endpoint.
- SETUP.md rewritten end to end around `omlx serve` with verify blocks.
- `check_docs.py` ties README measured rows to `results/measured_*.json`
  and fails on an unlabelled speed row. FAQ SSD-traffic sentence restored
  to the JSON values (it had drifted to 341 MB/tok; JSON says 629).

Fixes with runnable reproductions under `tests/`:

- `estimate.py` sized tensors as shape times a per-dtype byte table with a
  2-byte default. MLX 4-bit checkpoints store packed weights as U32, which
  the table lacked, so every packed weight was counted at half its size.
  Tensors are now sized from the header's `data_offsets`.
  (`tests/test_estimate_sizes.py`)
- `moefit_prefetch.py` sidecar: entries were recorded under
  `<hash>:<layer>` but reloaded under the bare hash, so nothing replayed
  after a restart, and the default path's directory was never created.
  (`tests/test_sidecar.py`)
- `moefit_prefetch.py` warming: a multi-token input warmed only the last
  token's rows, and a row crossing a 16 KiB page boundary left its tail
  cold. Every token and every page is now warmed; single-file
  `model.safetensors` checkpoints are recognised.
  (`tests/test_prefetch_warm.py`)
- `experiments/sim_paging.py`: the prefetch allowance solver could not
  throttle once the SSD was the bottleneck (the spare-time formula
  reduced to "allow what you just issued"). The shipped behaviour is kept
  as `--throttle legacy` so `results/sim_paging.json` stays reproducible;
  `--throttle compute` models an adaptive prefetcher.
  (`tests/test_sim_throttle.py`)
- `experiments/sim_paging.py`: LRU bookkeeping moved from an O(capacity)
  list scan per access to a lazy heap with identical eviction order,
  verified against a frozen copy of the original on 45 cases.
  (`tests/test_sim_equivalence.py`) Per-layer capacities and a
  per-token capacity audit were added.

Added: `--alloc hetero` (greedy per-layer capacity from build-trace LRU
hit curves), `experiments/synth_trace.py` and `synth_validate.py`
(synthetic traces calibrated to `results/*.json` for running the
simulator without the real traces), `check_docs.py`, `FAQ.md`,
`REVIEW.md`. Removed macOS AppleDouble files (`._*`).

Docs rewritten for scanning. The README path `tools/estimate.py` was
wrong, since both tools are at the repo root.

## Simulation table v2 (shipped in `results/sim_paging.json`)

An earlier version of the paging simulator inserted prefetched experts
without evicting, so the resident set drifted from 32 to 456 experts per
layer over a run and the table it produced overstated every prefetch
policy, and that table was retracted. The shipped v2 simulator enforces
capacity per layer, and the v2 table is the only one quoted in these
docs.
Capacity accounting is audited every token in the current code and the
simulator raises if any layer exceeds its cap.

## Geometry correction (post-review, all numbers re-simmed)

`estimate.py` counted U32-packed 4-bit tensors at 2 bytes/element, and
the original geometry constants inherited the same halving: the model
is 99.0 GiB (64.6 routed / 29.8 PLE / 4.6 floor), not 57 GiB, and
experts are 2.69 MiB, not 1.46. DRAM calibration was redone against
actual bytes read (57.4 tok/s x 4.75 GiB/token => 293 GB/s effective,
54% of peak). The table in SETUP.md and `results/sim_paging.json` were
regenerated on the real traces with the corrected geometry; ceilings
drop to 11.8 / 23.6 / 45.1 tok/s (LRU) and a 48 GB M4 Max reaches 51%
of a 128 GB Mac at cap 128 (96% at cap 192 pinned). All qualitative
conclusions are unchanged: LRU reaches the chip ceiling when capacity
allows, the pinned hot-set is the cheapest win, n-gram speculative
prefetch loses, and the sidecar wins tight capacity on repeated
prefixes.

## Renamed to moefit

Earlier names were `specexp` and `pagepilot`. The package, prefetch script,
and Application Support path are now `moefit`. Older changelog entries may
still mention the previous names.

