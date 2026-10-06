# SETUP — running Qwen3.8-Flash-class MoE on a smaller-RAM Mac

Audience: an AI agent setting this up for a human, or a competent human.
Every step has a **verify** block. Do not continue past a failing verify;
report it instead. Never delete user files.

## The measured reality (Qwen3.8-Flash-Next oQ4e)

Simulated at measured bandwidths; DRAM model calibrated within 4% of
real-machine decode (57.4 tok/s measured on M4 Max 128 GB). Page and
capacity accounting audited — an earlier optimistic version of this
table was wrong (a cache-capacity leak) and has been replaced.

Geometry: 2.9 GiB non-expert floor + 35.9 GiB routed experts
(512/layer × 48) + 17.9 GiB PLE n-gram table. Decode consumes ~700 MB
of expert bytes per token through the same unified DRAM whether or not
they are cached, so **tokens/sec is DRAM-bound: the ceiling belongs to
the chip, not the cache** (~14.8 tok/s base M4, ~29.9 M4 Pro, ~59.8
M4 Max). Paging only decides whether those bytes arrive before the
token needs them (free) or at that instant (stall).

Best simulated configs, 8k holdout tokens:

| RAM | experts/layer resident | LRU | pinned hot-set | + routing sidecar |
|---|---|---|---|---|
| 24 GB | 32 | 14.8 (at ceiling) | 14.1 | 13.6 |
| 24 GB | 128 | 14.8 | 14.8 | 14.2 |
| 32 GB | 64 | 26.1 | 25.8 | **26.6** |
| 32 GB | 192 | 29.9 | 29.9 (75 MB/tok SSD) | 29.2 |
| 48 GB | 128 | 53.5 | **58.3** | 54.1 |
| 48 GB | 192 | 59.8 | 59.8 | 58.5 |

Honest conclusions:
1. Plain LRU expert paging already reaches the DRAM ceiling at most
   sizes. The headline result is simply: **a 48 GB Mac runs this 57 GiB
   model at ~90% of a 128 GB Mac's decode speed** once the 2.9 GiB
   floor fits.
2. A **pinned static hot-set** (learned from build traces, no runtime
   prediction) beats LRU in the mid-capacity band and costs zero
   prefetch bandwidth — the cheapest real win. The **routing sidecar**
   (bit-exact replay of cached prefixes, ~0.55 KB/token) wins the
   tight-capacity band and never does harm on repeated prefixes.
3. n-gram **speculative prefetch measured net-negative** at tight
   capacity: ~75% miss rate wastes the SSD headroom that hides other
   misses. Do not ship it for byte-exact MoE execution.
4. SSD wear: LRU steady state moves ~140–340 MB/token; at 15 tok/s
   that is terabytes/day. Prefer the pinned-hot-set (lowest SSD) and
   size RAM for the low-SSD rows.

## Step 0 — measure the machine and model
```
python3 estimate.py /path/to/model-dir
```
Verify: JSON verdict printed. `FULL` → stop, load normally. `PAGING` →
note `resident_experts_per_layer`. `NO-GO` → floor exceeds usable RAM;
do not force it (swap will not fix a floor problem).

## Step 1 — PLE table warming (small, verified bit-exact)
The 17.9 GiB n-gram table is LRU-evicted under pressure; every token
gathers 16 random rows. The warmer computes the next token's exact rows
(hash replica verified bit-exact against the live model) and touches
their file pages:
```
python3 specexp_prefetch.py --model-dir ... --model <id> --ple-prefetch
```
Feed JSON arrays of token ids per turn (agent-integrated mode).
Verify: `warmed_rows=16` lines; `iostat -d 2` shows random-read spikes
at token boundaries flatten during long generations.

## Step 2 — routing sidecar for agent loops
`--sidecar` persists per-prefix routing traces; replay is bit-exact
(verified 8208/8208 router decisions identical across greedy re-runs).
Repeated/growing prefixes need no prediction and pay no stall.
Verify: second identical prompt logs sidecar hits.

## Step 3 — what NOT to believe
- No predictor changes the DRAM-bound tok/s ceiling.
- Expert-skipping speculation (skip non-predicted experts): coverage
  ~0.43@14 makes per-token strict hits ~impossible. Do not implement.
- n-gram speculative prefetch: measured net-negative at tight RAM.
- 24 GB: runs at ~14 tok/s ceiling; the OS may still pressure the
  floor. 32 GB is the honest comfortable minimum.

## Prior art
Strata (stratallm.org), oMLX (github.com/jundot/omlx), mlx-vlm. This
repo's code is clean-room; vendored arch under `vendor_mlx_vlm/` is
Apache-2.0 with attribution.
