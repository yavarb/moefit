# SETUP: Qwen3.8-Flash-Next on a 24 to 64 GB Mac

Audience: an AI agent setting this up for a human, or a competent human.
Every step has a **verify** block. Do not continue past a failing verify.
Report the failure instead. Never delete user files, never resize swap,
and never disable SIP.

What runs what:

- **oMLX** (`omlx serve`) is the engine. It has built-in MoE expert
  offload for this model family (`moe_expert_offload_enabled`, a
  per-layer resident fraction, LRU for the rest, SSD reads overlapped
  with compute) and can keep the 29.8 GiB PLE n-gram table on SSD
  (`qwen4_ple_ssd_offload`). moefit does not patch the engine.
- **moefit** decides the numbers and verifies them: `estimate.py` (go /
  no-go and how many experts per layer fit), `scripts/configure_omlx_paging.py`
  (turns that into oMLX settings), `scripts/bench_decode.py` (wall-clock
  tok/s), and `moefit_prefetch.py` (PLE page warming and the routing
  sidecar for agent loops).

Every path below assumes the checkpoint lives at
`~/models/Qwen3.8-Flash-Next-oQ4e-mtp`. Substitute your own.

## Step 0: tools on PATH

```bash
export PATH=/opt/homebrew/bin:$PATH
brew tap jundot/omlx && brew install jundot/omlx/omlx
```

**verify**: `omlx --version` prints `0.7.0` or newer. `python3 --version`
is 3.9 or newer (the moefit scripts are stdlib only; the Xcode or
Homebrew `python3` both work).

## Step 1: download the checkpoint

```bash
hf download Jundot/Qwen3.8-Flash-Next-oQ4e-mtp \
   --local-dir ~/models/Qwen3.8-Flash-Next-oQ4e-mtp
```

`hf` ships inside the oMLX keg
(`/opt/homebrew/opt/omlx/libexec/bin/hf`) if you have no other copy. The
download is about 99 GB. Run it in the background and poll:

```bash
du -sh ~/models/Qwen3.8-Flash-Next-oQ4e-mtp
ls ~/models/Qwen3.8-Flash-Next-oQ4e-mtp/model-*-of-*.safetensors | wc -l
ls ~/models/Qwen3.8-Flash-Next-oQ4e-mtp/.cache/huggingface/download/ 2>/dev/null | grep -c incomplete
```

**verify**: 21 shards plus `model.safetensors.index.json` and
`config.json`; zero `.incomplete` files; `du` reports about 99G and
stops growing. Do not start the server on a partial download: oMLX reads
every shard header at load.

## Step 2: measure the machine and the model

```bash
python3 estimate.py ~/models/Qwen3.8-Flash-Next-oQ4e-mtp
```

**verify**: a JSON verdict is printed. On `FULL`, stop here and load the
model normally. On `PAGING`, note `resident_experts_per_layer` (153 on a
36 GB M4 Max). On `NO-GO`, the non-expert floor exceeds usable RAM; swap
will not fix a floor problem, so do not force the load. The estimator
uses Apple's published 120 GB/s for a base M4, so on a 24 GB M4 it
prints a slightly lower ceiling than the simulation table below, which
uses 135 GB/s.

## Step 3: write the oMLX paging settings

```bash
python3 scripts/configure_omlx_paging.py \
   --model-dir ~/models/Qwen3.8-Flash-Next-oQ4e-mtp \
   --save-estimate results/estimate_$(hostname -s).json
```

The script re-runs `estimate.py`, computes the resident fraction as
`resident_experts_per_layer / experts_per_layer`, clamps it so the
resident load fits under 85% of Apple's Metal working-set cap (28 GiB on
a 36 GB Mac), links the checkpoint into `~/.omlx/models/<id>`, and
writes these keys for that model id in `~/.omlx/model_settings.json`:

| key | value | why |
|---|---|---|
| `moe_expert_offload_enabled` | `true` | stream non-resident experts from the checkpoint |
| `moe_expert_offload_resident_fraction` | from estimate (0.28 on an idle 36 GB Mac) | experts per layer kept in RAM |
| `qwen4_ple_ssd_offload` | `true` | 29.8 GiB n-gram table stays on SSD, rows gathered by mmap |
| `mtp_enabled` | `false` | plain one-token-per-step decode so tok/s is comparable; pass `--mtp` to turn the native MTP head back on (oMLX allows it with offload for this family) |

Other models' entries in that file are left alone. Pass `--fraction` to
override, `--dry-run` to only print.

**verify**: the script prints `wrote ~/.omlx/model_settings.json [<id>]`
with the four keys above, and `ls -l ~/.omlx/models/` shows the symlink.
`python3 -c 'import json;print(json.load(open("$HOME/.omlx/model_settings.json"))["models"])'`
shows the entry. If it prints `checkpoint not complete`, go back to Step 1.

## Step 4: serve

```bash
scripts/serve_paging.sh
```

which runs, with the log in `~/.omlx/logs/moefit-serve.log`:

```bash
omlx serve --model-dir ~/.omlx/models --port 8000 \
   --max-concurrent-requests 1 --memory-guard aggressive
```

One request at a time keeps the KV cache small; `aggressive` lets oMLX
use all but about 1.5 GB of a 36 GB Mac. On 48 GB or more, `balanced`
is fine. The model loads on the first request, not at start-up.

**verify**: the script prints `ready` and lists the model id. Then
`curl -s localhost:8000/v1/models` shows it. After the first request,
the server log (`~/.omlx/logs/moefit-serve.log`) contains lines like
these (from the 36 GB run):

```
Qwen4-Exp PLE mode for ~/.omlx/models/Qwen3.8-Flash-Next-oQ4e-mtp: mmap
moe expert offload: wrapped 48 layers at 18.0% residency (expert tables: 67.95 GB total, 12.21 GB resident)
Loaded model: Qwen3.8-Flash-Next-oQ4e-mtp (actual: 16.40GB, local estimate: 17.01GB, full model: 103.94GB, total: 17.01GB)
```

If the first request instead returns HTTP 507 and the log says

```
Model '...' (23.77GB) does not fit under the dynamic memory ceiling (20.01GB).
Close other apps to free RAM (static cap is 34.50GB but only 19.88GB is reclaimable right now) ...
```

other apps are holding RAM. Do not kill them for the user. Either ask
the user to close them, or shrink the resident set to what is free now:

```bash
python3 scripts/configure_omlx_paging.py --model-dir ~/models/Qwen3.8-Flash-Next-oQ4e-mtp --ceiling-gb 20.0
scripts/serve_paging.sh stop && scripts/serve_paging.sh
```

(`--ceiling-gb` takes the number from the error; on the 36 GB box this
gave 0.18.) Report the 507 text and the fraction you ended up with.

## Step 5: measure decode speed

```bash
python3 scripts/bench_decode.py --model Qwen3.8-Flash-Next-oQ4e-mtp \
   --max-tokens 128 --runs 3 \
   --label "measured on $(sysctl -n machdep.cpu.brand_string) $(( $(sysctl -n hw.memsize) / 1024**3 ))GB $(hostname -s)" \
   --out results/measured_$(hostname -s)_$(( $(sysctl -n hw.memsize) / 1024**3 ))gb.json
```

The warm-up request pays the model load (several minutes cold on a 36 GB
Mac). Each timed run is a greedy streaming completion; `decode_tps` is
tokens divided by the time between the first and last streamed token.

**verify**: three runs print `decode N tok/s` with `finish=length` and
the JSON lands in `results/`. On the 36 GB M4 Max this printed 7.8 tok/s
median at 0.18 residency (`results/measured_santa_cruz_36gb.json`). Numbers from this script are **measured**;
numbers in the simulation table below are not. Keep the two labelled.
The README's measured 36 GB row came from exactly this command; the
DRAM ceiling in the table below is what the chip could do if every
expert were already in RAM, which paging never achieves.

## Step 6: warm the n-gram table and keep routing traces (optional)

The PLE table is on SSD now, and every token gathers 16 rows from random
places in it. The warmer computes those rows exactly (hash replica
verified bit-exact against the live model) and touches their file pages
before the engine needs them:

```bash
python3 moefit_prefetch.py --model-dir ~/models/Qwen3.8-Flash-Next-oQ4e-mtp \
   --model Qwen3.8-Flash-Next-oQ4e-mtp --serve-url http://127.0.0.1:8000 --ple-prefetch
```

Feed one JSON array of token ids per line on stdin (agent-integrated
mode). Every token in the array is warmed. `--sidecar <path>` persists
per-prefix routing traces (default
`~/Library/Application Support/moefit/sidecar.jsonl`); replay is
bit-exact, 8208 of 8208 router decisions identical across greedy re-runs.

**verify**: each input line logs `warmed_rows=16` per token fed. After a
restart the same prompt still logs sidecar hits (`tests/test_sidecar.py`).

## Step 7: what NOT to believe

- No predictor changes the DRAM-bound tok/s ceiling.
- 24 GB runs at the 11.8 tok/s ceiling and the OS may still press on the
  floor. 32 GB is the comfortable minimum.
- A streamed expert costs an SSD read every time it misses. The resident
  fraction, not the policy, decides how often that happens.

## What the simulation says

Geometry: 4.6 GiB non-expert floor, 64.6 GiB routed experts (512 per
layer, 48 layers, 2.69 MiB each, top-10 routing), 29.8 GiB n-gram table.
Decode reads about 1.26 GB of expert weights per token from unified DRAM
whether or not they were cached, so tok/s is DRAM-bound and the ceiling
belongs to the chip: 11.8 tok/s on a base M4, 23.6 on an M4 Pro, 45.1 on
an M4 Max at full LRU (55 with a pinned hot-set; 57.4 measured fully
resident on a 128 GB M4 Max). Paging decides whether those bytes are already in RAM (no
stall) or arrive on demand from SSD (stall).

Simulated decode speed in tok/s, holdout routing trace, measured tier
bandwidths, capacity audited every token (`results/sim_paging.json`):

| RAM tier | experts/layer resident | LRU | pinned hot-set | routing sidecar | SSD (pinned) |
|---|---|---|---|---|---|
| 24 GB (M4) | 32 | 8.1 | 7.6 | 8.6 | 670 MB/tok |
| 24 GB (M4) | 64 | 11.8 | 11.7 | 11.4 | 438 MB/tok |
| 32 GB (M4 Pro) | 64 | 14.2 | 14.0 | 14.5 | 438 MB/tok |
| 32 GB (M4 Pro) | 128 | 23.6 | **25.7** | 23.8 | 239 MB/tok |
| 48 GB (M4 Max) | 128 | 29.1 | **31.7** | 29.4 | 239 MB/tok |
| 48 GB (M4 Max) | 192 | 45.1 | **54.8** | 45.3 | 138 MB/tok |

Policies:

- **LRU**: plain least-recently-used expert cache, no prediction. This is
  what oMLX's expert offload does today.
- **Pinned hot-set**: half the capacity holds the experts most used in
  the build traces, the other half is LRU. No runtime prediction, no
  prefetch bandwidth. Not yet wired into oMLX.
- **Routing sidecar**: a stored per-prefix routing trace replayed
  bit-exactly for prompts seen before (about 0.55 KB per token). The
  table shows its upper bound, where every prefix is a repeat.

What follows from the table:

1\. Paging costs real speed at these capacities: a 48 GB Mac at
   128 experts per layer runs at about 51% of the 128 GB machine's
   measured 57.4 tok/s with LRU (55% with the pinned hot-set), and
   about 96% once 192 per layer are pinned\.
2. The pinned hot-set beats LRU in the middle of the capacity range and
   costs no prefetch bandwidth. It is the cheapest real win.
3. The routing sidecar wins the tight-capacity rows and never hurts on
   repeated prefixes.

The simulator models the DRAM-bound compute step and the SSD stall; it
does not model the Metal working-set cap, page-cache competition between
the PLE table and streamed experts, or oMLX's per-step overhead. The
measured 36 GB row in the README is the only number here that includes
all of those.

## Checking the numbers

```bash
python3 check_docs.py
```

**verify**: prints `all documented numbers match`. The check covers the
simulation table above against `results/sim_paging.json` and the measured
rows in README against `results/measured_*.json`.
