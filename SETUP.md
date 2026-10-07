# SETUP: Qwen3.8-Flash-class MoE on a 24 to 64 GB Mac

Audience: an AI agent setting this up for a human, or a competent human.
Every step has a **verify** block. Do not continue past a failing verify.
Report the failure instead. Never delete user files, never resize swap,
and never disable SIP.

## What the simulation says

Geometry: 2.9 GiB non-expert floor, 35.9 GiB routed experts (512 per
layer, 48 layers, 1.46 MiB each, top-10 routing), 17.9 GiB n-gram table.
Decode reads about 700 MB of expert weights per token from unified DRAM
whether or not they were cached, so tok/s is DRAM-bound and the ceiling
belongs to the chip: 14.8 tok/s on a base M4, 29.9 on an M4 Pro, 59.8 on
an M4 Max. Paging decides whether those bytes are already in RAM (no
stall) or arrive on demand from SSD (stall).

Simulated decode speed in tok/s, holdout routing trace, measured tier
bandwidths, capacity audited every token (`results/sim_paging.json`):

| RAM tier | experts/layer resident | LRU | pinned hot-set | routing sidecar |
|---|---|---|---|---|
| 24 GB (M4) | 32 | 14.8 | 14.1 | 13.6 |
| 24 GB (M4) | 128 | 14.8 | 14.8 | 14.2 |
| 32 GB (M4 Pro) | 64 | 26.1 | 25.8 | **26.6** |
| 32 GB (M4 Pro) | 192 | 29.9 | 29.9 (75 MB/tok SSD) | 29.2 |
| 48 GB (M4 Max) | 128 | 53.5 | **58.3** | 54.1 |
| 48 GB (M4 Max) | 192 | 59.8 | 59.8 | 58.5 |

Policies:

- **LRU**: plain least-recently-used expert cache, no prediction.
- **Pinned hot-set**: half the capacity holds the experts most used in
  the build traces, the other half is LRU. No runtime prediction, no
  prefetch bandwidth.
- **Routing sidecar**: a stored per-prefix routing trace replayed
  bit-exactly for prompts seen before (about 0.55 KB per token). The
  table shows its upper bound, where every prefix is a repeat.

What follows from the table:

1. LRU alone reaches the chip ceiling at most sizes. A 48 GB Mac holding
   128 experts per layer runs at about 93% of the 128 GB machine's
   measured 57.4 tok/s; with the pinned hot-set it reaches the ceiling.
2. The pinned hot-set beats LRU in the middle of the capacity range and
   costs no prefetch bandwidth. It is the cheapest real win.
3. The routing sidecar wins the tight-capacity rows and never hurts on
   repeated prefixes.
4. n-gram speculative prefetch loses at tight capacity: about 75% of
   prefetches are wasted and they eat the SSD time that hides other
   misses. Do not ship it for byte-exact MoE execution.
5. SSD traffic under LRU is about 141 to 341 MB per token (steady state,
   cap 128 down to cap 32). At 15 tok/s that is several terabytes per
   day. Prefer the pinned hot-set, and pick the RAM tier whose row shows
   the lowest SSD traffic you can afford.

## Step 0: measure the machine and the model

```
python3 estimate.py /path/to/model-dir
```

**verify**: a JSON verdict is printed. On `FULL`, stop and load the
model normally. On `PAGING`, note `resident_experts_per_layer` for the
later steps. On `NO-GO`, the floor exceeds usable RAM, and swap will not
fix a floor problem, so do not force the load. The estimator uses Apple's published 120 GB/s for a base
M4, so on a 24 GB M4 it prints a slightly lower ceiling than the table
above, which uses 135 GB/s.

## Step 1: warm the n-gram table (small, bit-exact)

The 17.9 GiB table is evicted under memory pressure, and every token
gathers 16 rows from random places in the table. The warmer computes those rows
exactly (hash replica verified bit-exact against the live model) and
touches their file pages before the engine needs them:

```
python3 specexp_prefetch.py --model-dir /path/to/model-dir --model <served-id> --ple-prefetch
```

Feed one JSON array of token ids per line on stdin (agent-integrated
mode). Every token in the array is warmed.

**verify**: each input line logs `warmed_rows=16` per token fed (an
N-token array logs `warmed_rows=` 16 times N). During long generations,
`iostat -d 2` shows the random-read spikes at token boundaries flatten.

## Step 2: routing sidecar for agent loops

`--sidecar <path>` persists per-prefix routing traces. Replay is
bit-exact: 8208 of 8208 router decisions were identical across greedy
re-runs of the same prompt. Repeated or growing prefixes need no
prediction and pay no stall. The default path is
`~/Library/Application Support/specexp/sidecar.jsonl`, and the tool
creates that directory.

**verify**: the second identical prompt logs sidecar hits. After
restarting the tool, the same prompt still logs hits
(`tests/test_sidecar.py` covers this).

## Step 3: what NOT to believe

- No predictor changes the DRAM-bound tok/s ceiling.
- Expert-skipping speculation (skip non-predicted experts): coverage of
  0.44 at 14 candidates makes strict per-token hits almost impossible,
  so do not implement expert skipping.
- n-gram speculative prefetch: net-negative at tight RAM in the shipped
  simulation.
- 24 GB runs at the 14.8 tok/s ceiling and the OS may still press on the
  floor. 32 GB is the comfortable minimum.

## Checking the numbers

```
python3 check_docs.py
```

**verify**: prints `all documented numbers match results/sim_paging.json`.
