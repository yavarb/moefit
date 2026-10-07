# Changelog

Newest first. Every fix lists the test that reproduces the original
behaviour.

## Unreleased (review branch `fable-review`)

Fixes with runnable reproductions under `tests/`:

- `estimate.py` sized tensors as shape times a per-dtype byte table with a
  2-byte default. MLX 4-bit checkpoints store packed weights as U32, which
  the table lacked, so every packed weight was counted at half its size.
  Tensors are now sized from the header's `data_offsets`.
  (`tests/test_estimate_sizes.py`)
- `specexp_prefetch.py` sidecar: entries were recorded under
  `<hash>:<layer>` but reloaded under the bare hash, so nothing replayed
  after a restart, and the default path's directory was never created.
  (`tests/test_sidecar.py`)
- `specexp_prefetch.py` warming: a multi-token input warmed only the last
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
