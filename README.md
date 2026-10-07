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
2. **A pinned hot-set.** Which experts are “frequent” is stable across prompts. Pin the top ones from a short routing trace; let LRU handle the long tail. That was more reliable than runtime predictors in our measurements, and it does not spend bandwidth on wrong predictions.
3. **Page experts, keep the floor.** Attention, shared expert, and the head (~4.6 GiB) stay resident. The ~64.6 GiB of routed experts live mostly on SSD and enter RAM only when needed.

---

## How fast is it?

| Your Mac | Experts kept in RAM per layer | Decode (LRU) | Notes |
|---|---|---|---|
| M4 Max, 128 GB | all (model fits) | **57.4 tok/s — measured** | baseline; you don’t need moefit |
| M4 Max, 36 GB | 143 (fraction 0.28) | **15.55 tok/s — measured** ([results/measured_santa_cruz_36gb_n1024.json](results/measured_santa_cruz_36gb_n1024.json)) | oMLX 0.7.0 expert offload with **default IO pool** (IO_WORKERS=12), PLE on SSD, MTP off; n=1024 steady (median of 15.73/15.42/15.55). Prior n=128 short-run was 13.0 |
| M4 Max, 48 GB | 192 | **~15.9 tok/s — simulated (serial model)** | stock oMLX; bandwidth-model ideal-pipelining ceiling ~52–55 tok/s |
| M4 Max, 48 GB | 128 | **~11.8 tok/s — simulated (serial model)** | ceiling ~32 tok/s |
| M4 Pro, 32 GB | 128 | **~9.7 tok/s — simulated (serial model)** | usable for patient agent loops; ceiling ~26 tok/s |
| M4, 24 GB | 64 | **~4.9 tok/s — simulated (serial model)** | likely too slow for agents; ceiling ~12 tok/s |

Rows below 128 GB are re-priced under the serial-latency model (see Status); the older bandwidth-model numbers (45.1→54.8 etc.) remain in [SETUP.md](SETUP.md#what-the-simulation-says) and should be read as ideal-pipelining ceilings, not expectations. Simulated tier rows other than 36 GB extrapolate constants measured on Santa Cruz ([results/fidelity_tier_predictions.json](results/fidelity_tier_predictions.json)).

The limit is your chip’s memory bandwidth, not clever paging math: each token still reads about 1.3 GB of weights through unified RAM. Paging only decides whether you stall waiting for SSD. Full table and methods: [SETUP.md](SETUP.md#what-the-simulation-says).

**Status:** measured silicon baselines: the 128 GB full-fit row and the 36 GB paging run (M4 Max 36 GB “Santa Cruz”, oMLX 0.7.0 expert offload at 0.28 residency). **Exact-replay sidecar** on *repeated* prompts is a separate measured KEEP (~14→20 tok/s, +~40–44%; Silicon-proof) — not a non-repeat claim. Prefer **n≥1024** for the steady figure: **15.55 tok/s** median ([results/measured_santa_cruz_36gb_n1024.json](results/measured_santa_cruz_36gb_n1024.json)); earlier n=128/256 runs understated at ~12.7–13.0. That 15.55 number **already includes** oMLX 0.7.0’s default staged-IO / IO worker pool (DB-ON) — silicon A/B/A confirms keeping the default pool (see Silicon-proof below), not a new additive patch. The other paging rows are from a simulator calibrated within ~4% of the 128 GB baseline, replaying real router traces. At the matched 36 GB config (cap 143/layer), the default serial model predicts **12.9 tok/s** vs the short-run **13.0** band (~1%; [results/sim_paging_matched_cap143.json](results/sim_paging_matched_cap143.json)) — it does **not** yet describe the n=1024 DB-ON steady point. The old bandwidth/overlap model was ~2.8× optimistic; that path remains as `--time-model bandwidth`. DRAM full-fit calibration is unchanged (~55 sim / 57.4 measured). Details: [results/MEASURED_VS_SIM_36GB.md](results/MEASURED_VS_SIM_36GB.md). Next: a 48 GB M4 Max silicon run.

---

## Will it work on my Mac?

| Memory | Verdict |
|---|---|
| **128 GB+** | Don’t use this. Load the model normally. |
| **64 GB / 48 GB** | Works. ~12–16 tok/s simulated (serial model) at 128–192 experts per layer — slower than the older bandwidth-model table suggested; measured run pending. |
| **36 GB (M4 Max)** | Runs. **15.55 tok/s** measured idle steady (n=1024) with oMLX expert offload at 143 experts per layer (fraction 0.28); keep the default IO pool. |
| **32 GB** | Usable for patient agent loops at ~10 tok/s simulated (serial model); the older 14–26 range was bandwidth-model optimism. |
| **24 GB** | Not recommended: ~5 tok/s simulated (serial model) on top of OS memory pressure. Prefer 32 GB+. |

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
| `check_docs.py` | Fails if README/SETUP numbers disagree with `results/sim_paging.json` or `results/measured_*.json`, or a speed row is not labelled measured/simulated |
| `scripts/` | `configure_omlx_paging.py` (estimate → oMLX settings), `serve_paging.sh`, `bench_decode.py` |
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

## Measured on M4 Max 36 GB (Santa Cruz, paging)

| | |
|---|---|
| Engine | oMLX 0.7.0, `moe_expert_offload_enabled`, resident fraction 0.28 (143 of 512 experts per layer), `qwen4_ple_ssd_offload`, MTP off, **default IO pool** (`OMLX_MOE_OFFLOAD_IO_WORKERS=12`), `--memory-guard aggressive --max-concurrent-requests 1` |
| Decode (steady, prefer this) | **15.55 tok/s** median of 3× n=1024 greedy runs (15.73 / 15.42 / 15.55) |
| Decode (short-run, understated) | **13.0 tok/s** median of 3× n=128 (12.23–13.07); n=256 SSD-instrumented run 12.71 |
| Time to first token (n=128 set) | 0.68 s median on an 88-token prompt (prefix-cached) |
| Cold load (n=128 set) | 4.7 s to first token on warm box (only resident experts load; PLE rows are gathered through mmap) |
| Resident footprint | 22.6 GB actual (oMLX log: 19.0 GB of the 68 GB expert tables resident) |
| Memory during run | ~85% free before load; ~18–19% free during decode (idle: Photo Booth / prior omlx stopped; CLIProxy left on :8317) |
| Prior crowded run | 7.8 tok/s at fraction 0.18 when other apps held ~16 GB (live ceiling 20 GB); idle 0.28 target is the estimate.py default |

Raw JSON: [results/measured_santa_cruz_36gb_n1024.json](results/measured_santa_cruz_36gb_n1024.json) (steady); prior short-run [results/measured_santa_cruz_36gb.json](results/measured_santa_cruz_36gb.json); estimate: [results/estimate_santa_cruz_36gb.json](results/estimate_santa_cruz_36gb.json). Matched sim-vs-measured gap analysis (short-run calibration): [results/MEASURED_VS_SIM_36GB.md](results/MEASURED_VS_SIM_36GB.md); design experiments: [results/DESIGN_LEDGER.md](results/DESIGN_LEDGER.md).

### Silicon-proof (Santa Cruz, 2026-10-06–07 ET)

Lab claims scored on physical silicon. Artifacts: [results/silicon_proof_three/](results/silicon_proof_three/), [results/t2_silicon/](results/t2_silicon/), [results/t6_ssd_proof_clean.json](results/t6_ssd_proof_clean.json), [results/t3_lip_scored.json](results/t3_lip_scored.json).

**Repeat vs non-repeat.** Exact routing replay is a KEEP only when the same prompt prefix is decoded again (agent loops, retries). It is **not** a general speedup on fresh prompts. Learned correlators / Markov / cross-layer L→L+1 do not clear the bar under honest harnesses — do not market them as tip.

| Claim | Measured | Verdict | Merge? |
|---|---|---|---|
| **T2 exact routing-replay sidecar** (identical prompt, greedy; flag-file ON/OFF) | 3+3 runs: OFF median **13.95** → ON **20.05** tok/s (**+43.7%**); earlier pair mean **14.34 → 20.13** (**+40%**); **0** wasted prefetches; bit-identical text within each session ([t2_sidecar_3pair_summary.json](results/t2_silicon/t2_sidecar_3pair_summary.json), [t2_sidecar_aba_measured.json](results/t2_silicon/t2_sidecar_aba_measured.json)) | **KEEP — repeat-only** | **Yes for repeated prefixes.** Ship the sidecar patch (`patches/omlx_t2_sidecar_moe_expert_offload.py`); default OFF until a replay script exists. |
| **T9/T4 staged IO / double-buffer** (default IO pool vs `IO_WORKERS=1`) | ON **15.75** / ON2 **15.64** tok/s vs OFF **9.2**; paired Δ median **+6.54** | **TRANSFERS** | **Yes — keep the default IO pool.** Confirms the ~15.55 baseline (already in oMLX 0.7.0), not a new additive patch. |
| **T6 multi-expert coalescer** on real SSD | Contiguous expert-ID batching useful **5.39 GB/s** (beats **5.02** peak; contiguous ON/OFF **1.44×**). Random-order path still fails (4.67 / 4.44 vs 5.02). | **PASS** (contiguous only) | **Contiguous batching only.** Not a decode tok/s claim; no oMLX PR from this pass. Cached-file ~10.5 GB/s claims are **invalid as SSD proof**. |
| **T3 LIP v3** (`OMLX_ADMISSION=1`) | ON median **15.98** vs OFF **15.64** (**+0.34**, n=3) — CI crosses zero / below resolution floor | **No claimed win** | **Do not claim a silicon win.** |
| **M2c / non-repeat correlators** | Same-prompt “+32% / +37%” was route-replay leak; leak-free ON-first holdout: precision **3.0%**, Δ **−0.31** tok/s ([m2c_replay_off_verdict.json](results/t2_silicon/m2c_replay_off_verdict.json)) | **DROP_AS_REPLAY** | **No.** SP2 non-repeat +15% bar retired; best real paraphrase end-to-end ~**+5%**. |
| **Unguarded L→L+1 cross-layer prefetch** (`perf/moe-cross-layer-prefetch` on the oMLX fork) | Qwen hyper-connection linker NO-GO (OFF ≥ ON when linked) | **NO-GO** | **Do not merge to oMLX main.** Stock oMLX `main` remains the product tip. |

Do not treat random-order coalescer, LIP, correlators, or L→L+1 as shipped wins. Contiguous T6 is an SSD useful-bandwidth pass only.

Re-check docs against the shipped tables anytime:

```bash
python3 check_docs.py
```

---

## Credits and license

Model: [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (Qwen team). Validated checkpoint: [Jundot oQ4e+MTP MLX](https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp); other MLX quants of the same MoE should work via `estimate.py`. Architecture pieces from [oMLX](https://github.com/jundot/omlx) keep their Apache-2.0 attribution. **MIT.**
