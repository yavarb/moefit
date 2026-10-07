# moefit

**Run Qwen3.8-Flash-Next on a Mac that does not have enough RAM for the full model.**

Point your coding agent at this repo, give it the setup prompt below, and it can install and run a ~125B MoE model on Apple Silicon with **24–64 GB** of unified memory — less than the checkpoint needs to sit fully in RAM.

Apple Silicon · macOS 14+ · MIT · free and open source

---

## What it does

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is a large mixture-of-experts (MoE) model. Only a small slice of its weights is used on every token; the rest are “experts” that the router picks per token.

On a Mac with enough RAM (about 128 GB for the quant we test), you load the whole model and you are done. On a Mac with **less** RAM, you normally cannot run it at all.

**moefit** keeps the always-needed weights in RAM, pages the experts in from SSD as they are needed, and remembers which experts a prompt actually used so agent loops don’t thrash the disk.

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
2. **A pinned hot-set.** Which experts are “frequent” is stable across prompts. Pin the top ones from a short routing trace; let LRU handle the long tail. That was more reliable than runtime predictors in our measurements, and it does not spend bandwidth on wrong predictions.
3. **Page experts, keep the floor.** Attention, shared expert, and the head (~4.6 GiB) stay resident. The ~64.6 GiB of routed experts live mostly on SSD and enter RAM only when needed.

---

## How fast is it?

| Your Mac | Experts kept in RAM per layer | Decode (LRU → pinned hot-set) | Notes |
|---|---|---|---|
| M4 Max, 128 GB | all (model fits) | **57.4 tok/s — measured** | baseline; you don’t need moefit |
| M4 Max, 48 GB | 192 | 45.1 → **54.8 tok/s** | ~96% of full-fit when pinned |
| M4 Max, 48 GB | 128 | 29.1 → 31.7 tok/s | |
| M4 Pro, 32 GB | 128 | 23.6 → 25.7 tok/s | usable for agents |
| M4, 24 GB | 64 | 11.8 tok/s | chip bandwidth ceiling; tight |

The limit is your chip’s memory bandwidth, not clever paging math: each token still reads about 1.3 GB of weights through unified RAM. Paging only decides whether you stall waiting for SSD. Full table and methods: [SETUP.md](SETUP.md#what-the-simulation-says).

**Status:** 128 GB numbers are measured on silicon. Paging rows are from a simulator calibrated within ~4% of that measured baseline, replaying real router traces. Next step is a real 48 GB M4 Max paging run to turn the 54.8 tok/s row from simulated into measured.

---

## Will it work on my Mac?

| Memory | Verdict |
|---|---|
| **128 GB+** | Don’t use this. Load the model normally. |
| **64 GB / 48 GB** | Good fit. Pinned hot-set ≈ within 4% of full-resident speed in sim. |
| **32 GB** | Yes, with care (~14–26 tok/s on M4 Pro). Prefer the pinned hot-set. |
| **24 GB** | Marginal. OS wants the same RAM. Prefer 32 GB+. |

Check first (stdlib only; no heavy install):

```bash
python3 estimate.py /path/to/model-dir
```

It prints `FULL` (you don’t need moefit), `PAGING` (capacity per layer), or `NO-GO`.

---

## How it works

Apple Silicon has one memory pool for CPU and GPU — there is no separate “VRAM” — but ~99 GB still does not fit in 48 GB, and SSD is roughly 70× slower than the RAM the GPU reads from.

On each token the model touches a fixed “floor” plus about 10 of 512 experts per layer. moefit:

1. Keeps the floor in RAM.
2. Pins a learned hot-set of experts in RAM.
3. Pages the rest from SSD in the background when the routing sidecar (or hot-set) says they are needed for the next token.

```
token t emitted  →  sidecar / hot-set says which experts token t+1 needs
                         │
              already in RAM? ──yes──► compute t+1
                         │
                        no → start SSD read now (background)
                              if it lands before t+1, no stall
```

Longer write-up: [REVIEW.md](REVIEW.md). History of number changes: [CHANGELOG.md](CHANGELOG.md).

---

## What’s in the repo

| File / dir | Role |
|---|---|
| `estimate.py` | Go / no-go for a Hugging Face checkpoint on this Mac |
| `moefit_prefetch.py` | Routing sidecar: store and replay traces; warm what each token needs |
| `check_docs.py` | Fails if README numbers disagree with `results/sim_paging.json` |
| `experiments/` | Trace collection, paging simulator, probe studies |
| `results/` | Simulation tables and study outputs |
| `tests/` | Repros for bugs this repo has fixed |

Setup with verify blocks: [SETUP.md](SETUP.md). FAQ: [FAQ.md](FAQ.md).

---

## Model footprint (checkpoint)

| Piece | Size |
|---|---|
| Always-resident floor (attention, shared expert, head) | 4.6 GiB |
| Routed experts (512 per layer × 48 layers, top-10 active) | 64.6 GiB |
| PLE n-gram table | 29.8 GiB |

Sizes come from the safetensors header.

## Measured on M4 Max 128 GB (model fits)

| | |
|---|---|
| Decode | **57.4 tok/s** (`experiments/bench_baseline.py`) |
| Prefill | **586 tok/s** |
| Paging simulator vs that decode baseline | within **4%** (real router traces) |
| Router determinism (greedy re-runs) | **8208 / 8208** identical (`experiments/det_test.py`) |

Re-check docs against the shipped table anytime:

```bash
python3 check_docs.py
```

---

## Credits and license

Model: [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (Qwen team). Checkpoint used in tests: Jundot’s oQ4e+MTP MLX quant. Architecture pieces from [oMLX](https://github.com/jundot/omlx) keep their Apache-2.0 attribution. **MIT.**
