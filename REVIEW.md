# Review of pagepilot (then "specexp", branch `fable-review`)

Scope: `experiments/sim_paging.py`, `estimate.py`, `specexp_prefetch.py`,
and the user-facing docs. The real routing traces
(`results/traces/*.npz`) and the model were not available, so every
simulation in this review ran on a synthetic trace calibrated to the
shipped per-layer statistics. Every number below is from a command that
was run, and the command and its output are quoted or named.

Test runner notes: the stdlib tests run with plain `python3`. The
simulator tests need numpy, as the simulator does, and were run with
`/tmp/specexp-venv/bin/python` (Python 3.14, numpy 2.5.3).

## A. Bugs found, with reproductions

Four defects in shipped code and one documentation claim that the data
files contradict. Each has a test under `tests/` whose failing output
before the fix is quoted.

### A1. `estimate.py` counted 4-bit packed weights at half their size

The estimator sized every tensor as element count times a per-dtype
byte table, with a 2-byte default for dtypes missing from the table.
MLX 4-bit affine checkpoints store packed weights as `U32`, which the
table lacked, so each packed weight tensor was counted as 2 bytes per
element instead of 4. On the target checkpoint the routed-expert total
and the per-expert size would both have come out at roughly half, which
flips PAGING verdicts toward FULL and doubles the resident capacity the
tool recommends.

Reproduction (`tests/test_estimate_sizes.py`) builds a real single-file
safetensors with a U32 tensor and compares the header's byte lengths to
what the estimator computed. Before the fix:

```
BAD model.layers.0.mlp.experts.gate_proj.weight        U32     expected=   134217728 got=    67108864
OK  model.layers.0.mlp.experts.gate_proj.scales        BF16    expected=     8388608 got=     8388608
total expected=146800640 got=79691776 ratio=0.543
FAIL: mis-sized tensors: ['model.layers.0.mlp.experts.gate_proj.weight']
```

Fix: size tensors from the header's `data_offsets`, which are exact for
every dtype, and fall back to a completed dtype table only when offsets
are missing. After the fix the test prints `ratio=1.000 PASS`.

### A2. The routing sidecar lost every entry on restart

`Sidecar.record()` stored entries under the key `<hash>:<layer>`, but
`Sidecar.__init__()` reloaded the JSONL file under the bare hash.
`lookup()` uses the composite key, so after a process restart no prefix
ever hit. The SETUP verify step ("second identical prompt logs sidecar
hits") passes within one process and fails across a restart, which is
the case that the persistence exists for. Separately, `record()`
appended to the default path under `~/Library/Application
Support/specexp/` without creating that directory, so the first write on
a fresh Mac raised `FileNotFoundError`.

Reproduction (`tests/test_sidecar.py`), before the fix:

```
same-process lookup: [1, 2, 3]
after-restart lookup: None
FAIL: sidecar entries do not survive restart
FAIL: second layer entry lost
FAIL: record() into a missing directory raised: [Errno 2] No such file or directory: '.../Library/Application Support/specexp/sidecar.jsonl'
```

Fix: one `_key()` helper used by load, record, and lookup; `mkdir -p` of
the parent before the first append. After the fix: `PASS`.

### A3. Multi-token input warmed only the last token's n-gram rows

The stdin loop appended all ids from a line and then computed rows for
the newest token only. SETUP tells agents to "feed JSON arrays of token
ids per turn", so a 200-token prompt fed as one array warmed 16 rows,
not 3,200. `warm_file_range` also touched only the first page of each
row, so a row that crosses a 16 KiB page boundary left its tail cold,
and `ModelGeom` silently found no table when the checkpoint is a single
`model.safetensors` without an index file.

Reproduction (`tests/test_prefetch_warm.py`) uses a fake two-shard table
and the unchanged hashing code. After the fix:

```
fed 8 tokens: warmed_rows=128 expected=128
pages touched for a straddling row: [0, 16384]
PASS
```

The hash replica (`NgramKeys`) was not changed. Its bit-exactness claim
could not be re-verified here without the model.

### A4. The simulator's prefetch throttle cannot throttle when the SSD is the bottleneck

`solve_policy()` sizes the per-token prefetch allowance from

```
spare = SSD_rate * s_ms - (sync + PLE),   s_ms = max(compute_ms, stream_ms)
```

When `stream_ms > compute_ms`, `SSD_rate * stream_ms` equals
`sync + async + PLE` by construction, so `spare == async`: the solver
allows exactly as much prefetch as was just issued and the fixed point
lands on the unthrottled volume. The docstring describes async reads as
"hidden by design", and the simulate() docstring describes the allowance
as "free SSD slots". In the SSD-bound regime there are no free slots,
and the code does not model that.

Reproduction (`tests/test_sim_throttle.py`), uniform random routing at
cap 32 on the 24 GB tier (SSD-bound for every policy):

```
LRU   : tps=7.7 compute_ms=67.7 ssd_ms=129.7 sync=664MB async=0MB
probe : tps=4.9 compute_ms=74.9 ssd_ms=205.3 sync=664MB async=387MB
legacy spare formula at the fixed point = 387.1 MB/token; async actually issued = 387.1 MB/token
legacy throttle is a no-op when SSD-bound: CONFIRMED
fixed (--throttle compute): tps=7.6 compute_ms=67.7 ssd_ms=131.1 sync=671MB async=0MB
```

What this does and does not change:

- LRU and pinned hot-set rows issue no prefetch, so the shipped numbers
  for those two policies are unaffected. The headline conclusions stand:
  LRU reaches the ceiling at most sizes, and the pinned set is the
  cheapest win.
- Sidecar rows: the sidecar prefetches exactly the next token's experts,
  so converting a hidden prefetch into a sync miss moves the same bytes
  at the same time. On the synthetic trace the fix leaves every sidecar
  tok/s and SSD MB/token unchanged to the printed precision (section
  B3), while the "served" fraction drops in SSD-bound rows because the
  reads are now counted as stalls.
- Probe rows: the shipped simulation modelled a prefetcher that keeps
  issuing when the SSD is saturated. That is a legitimate model of a
  naive prefetcher, and the conclusion "do not ship n-gram prefetch"
  survives under the fix, but the size of the loss shrinks (section B3).

The shipped behaviour is preserved as `--throttle legacy` (the default,
so `results/sim_paging.json` stays reproducible from the code) and the
corrected window is `--throttle compute`. I did not edit the shipped
table: it was produced on the real traces, and the fix can only be
re-applied there.

### A5. Documentation claim that could not be reconciled with the data files

SETUP.md said the table came from "8k holdout tokens". The simulator
evaluates `min(--eval-sub, holdout rows)` positions with `--eval-sub`
defaulting to 8000, so 8,000 is a cap, not a count. The shipped JSONs
record the holdout size: `results/ple_probe.json` has
`"holdout_rows": 2882`, and `results/trigger_eval.json` has
`"eval_positions": 138336`, which is 2,882 positions times 48 layers.
The corpus has 13 holdout prompts. Unless the traces were re-collected
with longer generations after those two files were written, the table
was computed on 2,882 positions. The rewritten docs say "the holdout
routing trace" and do not quote a token count. If you have the npz
files, `python experiments/sim_paging.py` now prints
`holdout rows=N evaluated=M` on its first line.

### A6. Smaller inconsistencies, documented and left alone

- The simulator's 24 GB tier uses 135 GB/s of DRAM bandwidth, while the
  estimator's chip table uses Apple's published 120 GB/s for a base M4.
  The estimator therefore prints about 13 tok/s for a 24 GB M4 where
  the table says 14.8. SETUP.md now says so. A 24 GB Mac mini can also
  be an M4 Pro (273 GB/s), which neither tool distinguishes by RAM.
- `estimate.py` reserves 12 GiB for the OS by default, while the
  simulator's tiers imply 9 GiB. The estimator is the conservative one.
- The DRAM term charges prefetched bytes landing in DRAM (`async_mb`)
  but not sync-fetched bytes landing in DRAM. Both are DMA writes and
  neither goes through the GPU path the 0.385 efficiency factor was
  calibrated on. The asymmetry is at most a few percent of the compute
  term and I left it, since changing it would move every row by an
  uncalibrated amount.
- The original docstring gave the compute term as
  `floor + served_frac * 701 MiB`, but the code uses the full 701 MiB,
  which is right because every used expert is read from DRAM. The
  docstring now matches the code.
- The simulator replays prefill positions as if they were decode tokens.
  Prefill routes a whole prompt in one pass and would page differently.
  This is a modelling simplification shared by the shipped table and
  this review.

## B. Modelling experiment: heterogeneous per-layer capacity

One modelling change was evaluated end to end: letting each layer hold a
different number of resident experts under the same total RAM.

### B1. Hypothesis

The shipped simulator gives every layer the same number of resident
experts. Layers differ in how concentrated their routing is: the
build-split top-14 prior covers 0.11 to 0.26 of holdout top-10 routing
depending on the layer (`results/ple_probe.json`), and previous-token
persistence ranges from 0.06 to 0.51 (`results/trigger_eval.json`). If
the marginal value of one more resident expert differs across layers,
moving capacity from flat layers to layers where it buys hits should
raise the served fraction at the same total RAM. For the 32 GB tier at
64 experts per layer, where the shipped LRU row is SSD-bound (33.5 ms
compute vs 38.4 ms stream), each point of served fraction is worth
about 0.4 tok/s, so a 5-point gain would have changed the 32 GB advice.

### B2. Method

1. Synthetic traces (`experiments/synth_trace.py`): per layer, a Zipf
   popularity with exponent fitted to that layer's prior@14, a
   persistence probability fitted to that layer's "prev", and two global
   knobs for per-prompt topic locality and build-to-holdout shift, chosen
   with `experiments/synth_validate.py` to match the shipped LRU and
   pinned-hot-set served fractions. Same prompt counts and row counts as
   the real splits (17 build prompts, 3,747 rows; 13 holdout prompts,
   2,882 rows). Fit quality at the chosen knobs (beta 2.0, shift 0.3):

   ```
    cap mode     synthetic  shipped   delta
     32 lru          0.501    0.513  -0.012
     32 prior        0.457    0.481  -0.024
     64 lru          0.669    0.664  +0.005
     64 prior        0.625    0.661  -0.036
    128 lru          0.820    0.798  +0.022
    128 prior        0.791    0.815  -0.024
    192 lru          0.887    0.870  +0.017
    192 prior        0.875    0.893  -0.018
   mean |delta| served = 0.020
   ```

   One real feature the synthetic trace does not reproduce: on the real
   traces the pinned hot-set overtakes LRU at 128 and 192 experts per
   layer; on the synthetic trace LRU stays ahead at every size. The
   per-layer heterogeneity that the experiment depends on is present:
   per-layer LRU hit rates at cap 32 span 0.44 to 0.69 on the synthetic
   holdout (standard deviation 0.06), and build-split and holdout-split
   per-layer hit rates correlate at 0.94, so an allocator fitted on build
   has the right ordering.

2. Allocator (`experiments/hetero_alloc.py`): measure each layer's LRU
   hit curve on the build split at 17 capacities, then hand out the total
   budget 48 x C in steps of 4 experts to the layer with the largest
   interpolated marginal gain, with a floor of 16 experts per layer so no
   pin budget can exceed a layer's capacity. An "oracle" variant fits the
   allocation on the holdout itself, bounding what any allocator could
   reach.

3. Replay the holdout under per-layer caps with the shipped simulator's
   `simulate()`, which now accepts a list of 48 capacities and asserts
   every token that no layer exceeds its cap. `tests/test_hetero_alloc.py`
   checks the plumbing is live (a deliberately skewed 16/48 allocation
   changes the served fraction), the budget is spent exactly, and the
   audit fires on a leaky configuration:

   ```
   uniform 32 served=0.4140  skewed 16/48 served=0.4016
   allocator: sum=1536 min=16 max=48
   audit fired as expected: capacity leak: layer 0 holds 14 > 8 at token 0
   PASS
   ```

### B3. Measured delta

`/tmp/specexp-venv/bin/python experiments/hetero_alloc.py --traces-dir results/traces_synth`
(32 GB tier tok/s; full output in `results/hetero_alloc_synth.json`):

| cap | policy | uniform served / tok/s / SSD MB | hetero (build-fit) served / tok/s / SSD MB | oracle (holdout-fit) served | caps range |
|---|---|---|---|---|---|
| 32 | lru | 0.5009 / 17.6 / 350 | 0.5008 / 17.6 / 350 | 0.5009 | 24..48 |
| 32 | prior | 0.4572 / 16.1 / 380 | 0.4571 / 16.1 / 380 | 0.4574 | 24..48 |
| 32 | sidecar | 0.9890 / 18.6 / 329 | 0.9890 / 18.7 / 329 | 0.9891 | 24..48 |
| 64 | lru | 0.6694 / 26.5 / 232 | 0.6696 / 26.5 / 232 | 0.6696 | 48..96 |
| 64 | prior | 0.6250 / 23.4 / 263 | 0.6253 / 23.4 / 263 | 0.6252 | 48..96 |
| 64 | sidecar | 0.9882 / 27.2 / 225 | 0.9883 / 27.2 / 225 | 0.9883 | 48..96 |
| 128 | lru | 0.8196 / 29.9 / 126 | 0.8201 / 29.9 / 126 | 0.8201 | 96..160 |
| 128 | prior | 0.7914 / 29.9 / 146 | 0.7927 / 29.9 / 145 | 0.7925 | 96..160 |
| 128 | sidecar | 0.9965 / 28.9 / 125 | 0.9965 / 28.9 / 125 | 0.9965 | 96..160 |
| 192 | lru | 0.8874 / 29.9 / 79 | 0.8877 / 29.9 / 79 | 0.8877 | 160..256 |
| 192 | prior | 0.8746 / 29.9 / 88 | 0.8756 / 29.9 / 87 | 0.8760 | 160..256 |
| 192 | sidecar | 0.9981 / 29.3 / 78 | 0.9981 / 29.3 / 78 | 0.9981 | 160..256 |

The allocator spreads capacity between 0.75x and 1.5x of uniform, and
the served fraction moves by at most 0.0014 (pinned hot-set, cap 192,
oracle). SSD traffic moves by under 1 MB per token, and tok/s by at most
0.02 at any cap. The oracle allocation, which knows the holdout, is no
better than the build-fitted one, so the ceiling for this idea on this
trace is the same zero.

Why: although layers differ in hit rate, their hit curves have nearly
the same slope at any given capacity, so the greedy allocator finds
equal marginal gains and stays close to uniform. What varies across
layers in the fitted model is mostly the level of the curve, not its
shape.

Throttle fix delta, for completeness (`experiments/throttle_delta.py`,
candidate lists built with the real probe's precision of 0.31 right
picks, about 70% wasted; `results/throttle_delta_synth.json`):

THROTTLE_TABLE_PLACEHOLDER

### B4. Verdict

Negative. Heterogeneous per-layer capacity does not beat the shipped
uniform allocation on a trace calibrated to the shipped statistics, and
an oracle allocation cannot either. The allocator was removed from the
shipped simulator and kept in `experiments/hetero_alloc.py` with its
test so the result can be re-run on the real traces, where the hit-curve
shapes may differ. The one change that survives in the simulator is
per-layer capacity support plus the per-token capacity audit.

## C. Documentation changes

- `README.md`: what the repo is, a one-paragraph verdict, a "Should I
  use this?" table by RAM size, what is in the repo, agent rules, the
  "what NOT to believe" list, a table of measured and simulated numbers
  with their sources. The `tools/estimate.py` path was wrong and is now
  `estimate.py`.
- `SETUP.md`: the results table and the policies it compares, then the
  three steps with their verify blocks. Verify blocks are unchanged in
  substance; Step 1's verify now states the per-token count for
  multi-token input (16 rows per token), and Step 2's verify adds the
  restart case the sidecar bug would have failed. The retraction story
  moved to `CHANGELOG.md`.
- `FAQ.md`: eight questions a 32 GB buyer asks, written from the
  shipped table only.
- `CHANGELOG.md`: this review's fixes with their tests, and the v1
  capacity-leak retraction.
- `check_docs.py` (stdlib): parses the SETUP table and the ceilings,
  SSD-traffic range, and 48-vs-128 GB percentage quoted in prose, and
  compares them to `results/sim_paging.json`. It passes on the current
  docs:

  ```
  SETUP.md table: 6 rows checked
  ceilings from JSON: {'24GB-M4': 14.8, '32GB-M4P': 29.9, '48GB-M4M': 59.8}
  48 GB cap 128 vs measured 128 GB: LRU 93% pinned 102%
  all documented numbers match results/sim_paging.json
  ```

- Removed the eight macOS AppleDouble files (`._*`) from the index and
  working tree and added `._*` to `.gitignore`.

## D. What to do next on a real Mac

1. Re-run `experiments/sim_paging.py` on the real traces with
   `--throttle compute`, and re-run `experiments/hetero_alloc.py` and
   `experiments/throttle_delta.py` with `--traces-dir results/traces`.
   The first line of output states the holdout row count (A5). If the
   per-layer hit curves on real traces have different shapes, the
   heterogeneous allocation deserves a second look, which the synthetic
   trace cannot settle.
2. Measure the one thing the simulator takes on faith: that a sync miss
   costs exactly its bytes at the SSD's sequential rate. Random 1.46 MiB
   reads from an APFS-backed mmap under memory pressure have queueing
   and page-fault overhead that would raise every SSD time in the table.
   A 20-line test that mmaps the expert shards, evicts them with
   `purge`, and times 1,000 random expert reads would calibrate the
   `ssd` numbers in `TIERS` the way the 0.385 factor calibrated DRAM.
3. Validate the pinned hot-set end to end: pin the build-trace top
   experts with `mlock`-style wiring on a 48 GB M4 Max, run the holdout
   prompts, and compare measured tok/s and SSD MB/s against the 58.3
   tok/s and 130 MB/token rows. That one measurement would convert the
   repo's best recommendation from simulated to measured.
