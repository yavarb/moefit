# specexp

Run a 99 GB mixture-of-experts model (Qwen3.8-Flash-Next class) on a
24 to 64 GB Apple Silicon Mac. The model's always-needed part (4.6 GiB)
stays in RAM. The 64.6 GiB of routed experts are paged from SSD, and the
29.8 GiB n-gram embedding table streams row by row.

## Verdict

Decode speed on these models is set by DRAM bandwidth, because every
token reads about 1.26 GB of expert weights from unified memory whether
those experts were cached or just arrived from SSD. A paging policy can
only decide whether the bytes are already in RAM when the token needs
them. In simulation at measured bandwidths, plain LRU paging already
reaches that chip-bound ceiling at most RAM sizes. A 48 GB Mac holding
192 experts per layer with a pinned hot-set decodes at about 55 tok/s
against a 128 GB Mac's measured 57.4 — the pinned set reaches ~95%
there, while LRU alone sits at 51 to 79%. A 32 GB Mac reaches 14 to
26 tok/s. A 24 GB Mac is capped near 11.8 tok/s by its chip and leaves little room for the OS.

## Should I use this?

| Your Mac | Verdict |
|---|---|
| 128 GB or more | No. The model fits. Load it normally. |
| 64 GB | Yes. 192+ experts per layer, pinned hot-set; ~55 tok/s on M4 Max. |
| 48 GB | Yes. 128 to 192 experts per layer; 29 to 55 tok/s simulated on M4 Max (pinned hot-set). |
| 32 GB | Yes, with care. 64 to 128 experts per layer; 14 to 26 tok/s simulated on M4 Pro. Prefer the pinned hot-set to keep SSD traffic low. |
| 24 GB | Marginal. The chip caps decode at 11.8 tok/s, and the OS may push on the 2.9 GiB floor. 32 GB is the comfortable minimum. |

Run the estimator before anything else. It reads the checkpoint and this
machine and prints FULL, PAGING, or NO-GO:

```
python3 estimate.py /path/to/model-dir
```

## What is in the repo

- `estimate.py`: stdlib-only go/no-go for a model directory on this Mac.
- `specexp_prefetch.py`: warms the n-gram table rows each token needs and
  stores routing traces for exact replay of repeated prompts (the sidecar).
- `check_docs.py`: verifies every number in these docs against
  `results/sim_paging.json`.
- `experiments/`: trace collection (needs the model and a 128 GB Mac),
  the paging simulator, the probe and n-gram studies, and a synthetic
  trace generator for running the simulator without the real traces.
- `results/`: the shipped simulation table and study outputs.
- `tests/`: runnable reproductions for every bug fixed in this repo.

Both tools are at the repo root, not under `tools/`. Setup steps with
verify blocks are in [SETUP.md](SETUP.md), buyer questions are in
[FAQ.md](FAQ.md), and the history of retracted numbers is in
[CHANGELOG.md](CHANGELOG.md).

## For AI agents

Read this file, then SETUP.md, and run every `verify` block after each
step. Rules:

- Never delete user files, never resize swap, and never disable SIP.
- Do not proceed past a failing verify. Report it to the user instead.
- Report measured numbers (tok/s, RSS) back to the user, not expectations.

## What NOT to believe

- No predictor raises the DRAM-bound tok/s ceiling. Paging decides stall
  or no stall, and the chip decides the ceiling.
- Expert-skipping speculation (run only predicted experts) fails here.
  The best probe covers 0.44 of the true top-10 set with 14 candidates,
  so exact per-token hits are vanishingly rare.
- n-gram speculative prefetch is net-negative at tight RAM in the shipped
  simulation: about three quarters of prefetches are wasted and they
  consume the SSD time that hides unavoidable misses.
- 24 GB "works" only at the chip's ceiling with the OS pressing on the
  floor. Treat it as marginal.

## Numbers and where they come from

| Quantity | Value | Source |
|---|---|---|
| Non-expert floor | 2.9 GiB | checkpoint header |
| Routed experts | 35.9 GiB (512/layer x 48 layers, 1.46 MiB each, top-10) | checkpoint header |
| PLE n-gram table | 17.9 GiB | checkpoint header |
| Decode, 128 GB M4 Max | 57.4 tok/s | measured, `experiments/bench_baseline.py` |
| Prefill, 128 GB M4 Max | 586 tok/s | measured |
| Simulator DRAM model | within 4% of measured decode | calibration |
| Router determinism | 8208 of 8208 rows identical across greedy re-runs | `experiments/det_test.py` |
| PLE-probe coverage at 14 candidates | 0.44 (mass 0.46); random 0.025 | `results/ple_probe.json` |

Everything labelled "simulated" comes from `experiments/sim_paging.py`
replaying the holdout routing trace at measured tier bandwidths, with
per-layer capacity audited every token. Check the docs against the
shipped table at any time:

```
python3 check_docs.py
```

## Prior art

Strata (stratallm.org), oMLX (github.com/jundot/omlx), mlx-vlm. This
repo's code is clean-room. The vendored architecture under
`vendor_mlx_vlm/` is Apache-2.0 with attribution.
