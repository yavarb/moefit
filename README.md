# pagepilot

**Run a 125-billion-parameter MoE model on a Mac that can't hold it.**

Apple Silicon, 24 to 64 GB unified memory · macOS 14+ · free and open source

pagepilot runs **Qwen3.8-Flash-Next** (and MoE models shaped like it) on
Macs that don't have room for the whole model. The always-hot part
(4.6 GiB) stays in RAM, the 64.6 GiB of experts page in from the SSD,
and a routing trace remembers which experts each prompt used — so
repeated prompts and agent loops never guess.

*Renamed from "specexp": the speculative-experts idea turned out not to
pay for itself; what survives is expert paging, a pinned hot-set, and
an exact routing sidecar. See
[what we tested and rejected](#what-we-tested-and-rejected).*

## How fast is it?

A token is about ¾ of a word. 60 tokens/s is faster than you can read.

| Your Mac | Experts resident per layer | Decode, LRU → pinned | vs full-fit |
|---|---|---|---|
| M4 Max, 128 GB | all (the model fits) | **57.4 tok/s — measured** | the comparison |
| M4 Max, 48 GB | 192 | 45.1 → **54.8 tok/s** | 96% of full-fit pinned |
| M4 Max, 48 GB | 128 | 29.1 → 31.7 tok/s | 55% |
| M4 Pro, 32 GB | 128 | 23.6 → 25.7 tok/s | — |
| M4, 24 GB | 64 | 11.8 tok/s (the chip's ceiling) | — |

The ceiling is your chip's memory bandwidth, not the paging: every
token reads about 1.3 GB of weights through the same unified RAM
whether they were cached or just arrived from SSD. Paging decides
whether you pay a stall on the way there. Full table and sources:
[SETUP.md](SETUP.md#what-the-simulation-says).

## Which Macs, which verdicts

| | |
|---|---|
| **128 GB or more** | You don't need this. The model fits. Load it normally. |
| **64 GB / 48 GB** | Yes. 192 experts per layer with the pinned hot-set: within ~4% of a fully-resident M4 Max. |
| **32 GB** | Yes, with care: 14–26 tok/s on an M4 Pro. Prefer the pinned hot-set; it also halves SSD traffic. |
| **24 GB** | Marginal: 11.8 tok/s chip ceiling, and the OS competes for the same RAM. 32 GB is the comfortable minimum. |
| **Not a Mac?** | [Strata](https://github.com/Niko1221/Strata) does this job on Windows/Linux PCs with NVIDIA/AMD GPUs. |

Before anything else, run the estimator on your Mac:

```
python3 estimate.py /path/to/model-dir
```

It reads the checkpoint and your machine and prints FULL (you don't
need us), PAGING (here's your per-layer capacity), or NO-GO.

## Let your AI set it up

Use an AI coding assistant (Claude Code, Codex, Cursor, Hermes...)?
Paste this into it:

```
Set up pagepilot on this Mac for me: https://github.com/yavarb/pagepilot
— read README.md and SETUP.md, run every verify block after each step,
never delete user files, never resize swap, never disable SIP, and
report measured numbers back to me.
```

## How it works (90 seconds)

Apple Silicon shares one pool of memory between CPU and GPU, so there
is no "out of VRAM" — but 99 GB still doesn't fit in 48 GB, and the SSD
is ~70× slower than the RAM the GPU reads from. Three ideas carry the
design:

1. **Page the experts, keep the floor.** Attention, shared expert and
   head (4.6 GiB) are read every token; the 512 routed experts per
   layer are not — a token touches 10 of 512. Keep the frequent ones
   resident, stream the rest.
2. **Pin a learned hot-set.** "Frequent" is stable across prompts: pin
   the trace-fitted top experts, let LRU carry the tail. This beat
   every runtime predictor we measured and costs zero speculative
   bandwidth.
3. **Remember routing exactly (the sidecar).** Given the same prefix,
   an MoE model routes identically every time — verified 8,208/8,208
   router decisions bit-exact across greedy re-runs. For agent loops
   that resend a growing prefix, the next token's experts are a
   lookup, not a prediction.

```
 token t emitted ──► sidecar lookup / hot-set for t+1
                           │
           resident? ─yes──┤── no: SSD read queued NOW (background)
                           │          └─ lands before t+1? costs nothing
 token t+1 computed ◄──────┘   late = the stall you see in the table
```

The longer story, including everything that failed:
[REVIEW.md](REVIEW.md) and [CHANGELOG.md](CHANGELOG.md).

## What we tested and rejected

Three plausible ideas died in measurement. They're in the front matter
because a project that buries its failures will quote their numbers to
you:

- **Expert-skipping speculation** (run only predicted experts): the
  best trained probe covers 0.44 of the true top-10 set at 14
  candidates; exact per-token hits are vanishingly rare. Dead.
- **n-gram speculative prefetch**: ~75% of prefetches are wasted bytes
  that spend exactly the SSD time hiding the misses you can't avoid.
  Net-negative at every capacity tested. Dead.
- **Heterogeneous per-layer capacity**: an oracle allocator with
  hindsight ties with uniform allocation. Dead.

## What is in this repo

- `estimate.py` — stdlib-only go/no-go for any HF checkpoint on your Mac.
- `pagepilot_prefetch.py` — warms the n-gram table rows each token needs;
  stores and replays routing traces (the sidecar).
- `check_docs.py` — fails if any number in these docs disagrees with
  `results/sim_paging.json`.
- `experiments/` — router trace collection, the paging simulator
  (capacity audited every token), the probe studies, and a synthetic
  trace generator so the simulator runs without the 99 GB checkpoint.
- `results/` — the simulation table and study outputs.
- `tests/` — a runnable reproduction for every bug this repo has fixed.

Setup with verify blocks: [SETUP.md](SETUP.md). Buyer questions:
[FAQ.md](FAQ.md). Where every number comes from and which ones we
revised: [CHANGELOG.md](CHANGELOG.md).

## Numbers and where they come from

| Quantity | Value | Source |
|---|---|---|
| Non-expert floor | 4.6 GiB | checkpoint header (data_offsets) |
| Routed experts | 64.6 GiB (512/layer × 48, 2.69 MiB each, top-10) | checkpoint header |
| PLE n-gram table | 29.8 GiB | checkpoint header |
| Decode, 128 GB M4 Max | 57.4 tok/s | measured, `experiments/bench_baseline.py` |
| Prefill, 128 GB M4 Max | 586 tok/s | measured |
| Simulator DRAM model | within 4% of measured decode | calibrated on real traces |
| Router determinism | 8208 of 8208 rows identical across greedy re-runs | `experiments/det_test.py` |
| PLE-probe coverage at 14 candidates | 0.44 (mass 0.46); random 0.025 | `results/ple_probe.json` |

Everything labelled "simulated" comes from `experiments/sim_paging.py`
replaying the holdout routing trace at measured tier bandwidths, with
per-layer capacity audited every token. Re-check these docs against the
shipped table at any time:

```
python3 check_docs.py
```

## Status

The simulator is calibrated within 4% of a measured fully-resident
run; the paging rows are simulated on real router traces and labeled as
such. Next on the list: the first paging run on real silicon — a 48 GB
M4 Max pilot converts the table's best row (54.8 tok/s, ~96% of
full-fit) from simulated to measured.

## Credits and license

The model is [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
by the Qwen team; the checkpoint tested here is Jundot's oQ4e+MTP MLX
quant. [Strata](https://github.com/Niko1221/Strata) and
[oMLX](https://github.com/jundot/omlx) are prior art — Strata for
expert streaming on consumer PCs, oMLX for the architecture code
(vendored pieces keep their Apache-2.0 attribution). Everything else is
clean-room. MIT license.
