# moefit

**Run Qwen3.8-Flash-Next on a Mac that does not have enough RAM for the full model.**

Point your coding agent at this repo, give it the setup prompt below, and it can install and run a ~125B MoE model on Apple Silicon with **24–64 GB** of unified memory — less than the checkpoint needs to sit fully in RAM.

Apple Silicon · macOS 14+ · MIT · free and open source

---

## What it does

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is a large mixture-of-experts (MoE) model. Only a small slice of its weights is used on every token; the rest are “experts” that the router picks per token.

On a Mac with enough RAM (about 128 GB for the quant we test), you load the whole model and you are done. On a Mac with **less** RAM, you normally cannot run it at all.

**moefit** keeps the always-needed weights in RAM, pages the experts in from SSD as they are needed, and remembers which experts a prompt actually used so agent loops don’t thrash the disk.

**Validated on:** [Jundot’s Qwen3.8-Flash-Next oQ4e+MTP MLX](https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp) (the checkpoint used for the numbers below). It should work on any other **MLX** build of the same Qwen3.8-Flash-Next MoE — 4-bit or 8-bit — as long as `estimate.py` reports `PAGING`. Bigger quants leave less room for resident experts and will be slower; dense (non-MoE) models are out of scope.

If you have a coding agent on your Mac (Claude Code, Cursor, Codex, Hermes, …), paste this:

```
Set up moefit on this Mac for me: https://github.com/yavarb/moefit
— read README.md and SETUP.md, run every verify block after each step,
never delete user files, never resize swap, never disable SIP, and
report measured numbers back to me.
```

---

## Why this approach is different

1. **Exact routing, not speculation.** For a fixed prompt prefix, the MoE router always chooses the same experts (we checked 8,208/8,208 decisions bit-exact across greedy re-runs). Agent loops that resend a growing prefix can look up the next experts instead of predicting them.
2. **A pinned hot-set.** Which experts are “frequent” is stable across prompts. Pin the top ones from a short routing trace; let LRU handle the long tail.
3. **Page experts, keep the floor.** Attention, shared expert, and the head (~4.6 GiB) stay resident. The ~64.6 GiB of routed experts live mostly on SSD and enter RAM only when needed.

---

## How fast is it?

| Your Mac | Experts kept in RAM per layer | Decode | Notes |
|---|---|---|---|
| M4 Max, 128 GB | all (model fits) | **57.4 tok/s — measured** | baseline; you don’t need moefit |
| M4 Max Studio, 36 GB | 143 (fraction 0.28) | **15.55 tok/s — measured** ([results/measured_m4max_36gb_n1024.json](results/measured_m4max_36gb_n1024.json)) | oMLX 0.7.0 expert offload, default IO pool; n=1024 steady. Short-run n=128 was 13.0 |
| M4 Max, 48 GB | 192 | **~15.9 tok/s — simulated** | serial miss model |
| M4 Max, 48 GB | 128 | **~11.8 tok/s — simulated** | serial miss model |
| M4 Pro, 32 GB | 128 | **~9.7 tok/s — simulated** | usable for patient agent loops |
| M4, 24 GB | 64 | **~4.9 tok/s — simulated** | likely too slow for agents |

Exact routing-replay on **repeated** prompts is a separate measured KEEP (~+40%); it is not a general speedup on fresh prompts. Simulated rows use a serial miss model calibrated to measured silicon; details in [SETUP.md](SETUP.md#what-the-simulation-says).

---

## Measured on M4 Max 128 GB (model fits)

| | |
|---|---|
| Decode | **57.4 tok/s** (`experiments/bench_baseline.py`) |
| Prefill | **586 tok/s** |
| Paging simulator vs that decode baseline | within **4%** (real router traces) |
| Router determinism (greedy re-runs) | **8208 / 8208** identical (`experiments/det_test.py`) |

## Measured on M4 Max Studio 36 GB (paging)

| | |
|---|---|
| Decode | **15.55 tok/s** steady (n≥1024, median of 15.73 / 15.42 / 15.55); short-run **13.0** (n=128) |
| Prefill | not separately published for this config (decode-focused paging runs) |
| Paging simulator vs short-run decode | within **~1%** (12.9 sim vs 13.0 measured at cap 143) |
| Router determinism (greedy re-runs) | **8208 / 8208** identical (same model property as above) |
| Exact-replay on repeated prompts | **~+40%** decode (e.g. 14.3 → 20.1 tok/s) — repeat prefixes only |

Engine for the 36 GB row: oMLX 0.7.0, `moe_expert_offload_enabled`, resident fraction 0.28 (143/512 experts per layer), PLE on SSD, MTP off, default IO pool. Raw JSON: [results/measured_m4max_36gb_n1024.json](results/measured_m4max_36gb_n1024.json) (steady), [results/measured_m4max_36gb.json](results/measured_m4max_36gb.json) (short-run).

---

## Will it work on my Mac?

| Memory | Verdict |
|---|---|
| **128 GB+** | Don’t use this. Load the model normally. |
| **64 GB / 48 GB** | Works. ~12–16 tok/s simulated at 128–192 experts per layer. |
| **36 GB (M4 Max)** | Runs. **15.55 tok/s** measured steady with stock oMLX paging at 143 experts/layer. |
| **32 GB** | Usable for patient agent loops at ~10 tok/s simulated. |
| **24 GB** | Not recommended: ~5 tok/s simulated plus OS memory pressure. Prefer 32 GB+. |

```bash
python3 estimate.py /path/to/model-dir
```

Prints `FULL`, `PAGING` (capacity per layer), or `NO-GO`.

---

## How it works

Apple Silicon has one memory pool for CPU and GPU. On each token the model touches a fixed floor plus about 10 of 512 experts per layer. moefit:

1. Keeps the floor in RAM.
2. Pins a learned hot-set of experts in RAM.
3. Pages the rest from SSD when the routing sidecar (or hot-set) says they are needed.

Setup with verify blocks: [SETUP.md](SETUP.md). FAQ: [FAQ.md](FAQ.md).

```bash
python3 check_docs.py
```

---

## Credits and license

Model: [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (Qwen team). Validated checkpoint: [Jundot oQ4e+MTP MLX](https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp). Architecture pieces from [oMLX](https://github.com/jundot/omlx) keep their Apache-2.0 attribution. **MIT.**
