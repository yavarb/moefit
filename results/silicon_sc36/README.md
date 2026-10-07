# Santa Cruz 36 GB — measured decode decomposition (2026-10-06, lead_silicon)

All numbers are MEASURED on Santa Cruz (M4 Max 14C, 36 GB, macOS 27.0),
oMLX 0.7.0 expert offload at resident fraction 0.28 (143/512 experts/layer),
PLE on SSD, MTP off. Same already-loaded server process (pid 96103), never restarted
or reconfigured. Greedy, 256 decode tokens unless noted. Simulated numbers are
labelled SIM.

## 1. Workload regression: what decode time scales with miss traffic
`experiments/regress_prompts.py` + `measure_ssd_per_token.py`: 6 prompts x 2 passes
plus 5 other runs (17 points). Physical SSD MB/token from iostat during decode.

    ms/token = 39.2 ± 8.4  +  0.288 ± 0.053 ms/MB × disk_MB/token      (r² 0.66, n=17)

- range covered: 133–197 MB/token, 76–98 ms/token (10.2–13.1 tok/s)
- slope 0.288 ms/MB = **0.80 ms per missed expert** (2.765 MB) on the critical
  path, i.e. ~3.5 GB/s *serialized* read+install throughput
- at the 13 tok/s operating point (~140 MB/token) ≈ 40 ms/token scales with misses,
  ≈ 39 ms/token does not (compute + per-layer sync + Python/dispatch)
- prompt matters: json_tool 10.2–10.6 tok/s (187–197 MB/tok), essay/code 12.2–12.7.
  `collect_silicon_run.py`'s own prompt hit 13.9 / 15.3 tok/s (no disk counter on that run).
  **13.0 tok/s is a property of the prompt as much as of the box.**
- `results/silicon_sc36/regress{1,2,_pooled}.json`

## 2. Page cache is not a hidden expert tier here
`experiments/pagecache_expert_residency.py` (mincore over every expert slab, after
decode runs): only **2.29 GB of 71.6 GB** expert tables in the UBC; 683/24576 expert
slabs fully cached (14/layer mean). With ~16% free memory during decode the page
cache cannot hold re-missed experts, so physical SSD MB ≈ logical miss MB.
Rejects the "page-cache dedup" explanation for measured 140 MB/token < SIM LRU
207 MB/token; the lower traffic must come from real routing being more cache-friendly
than synth traces (or oMLX LRU on real routes), not from UBC hits.

## 3. GPU is idle more than half the decode
ioreg `Device Utilization %` sampled every 0.5 s during a 256-token decode:
median 42% (n=39 busy samples), idle 0% before. This *includes* oMLX's keep-alive
kernels issued while waiting on reads, so true model-compute occupancy is ≤42%.
Consistent with the serial per-layer read→install→compute chain
(`gap_santa_cruz.py`, SIM 12.1 tok/s with microbench constants).

## 4. Read path: 12 IO threads mostly idle
`sample` of the server during decode (perturbs throughput: 9.9–10.8 tok/s while
sampled, so these are proportions only): each of the 12 `omlx-moe-io` threads is
in `pread` ~8–19% of samples and waiting on a condvar the rest. Queue depth is
bounded by the per-layer miss count (~1.2 experts × 9 slabs), not by the pool size.
The engine thread splits between MLX GPU-event waits (incl. keep-alive) and Python
lock waits on read futures; `_platform_memmove` via numpy `array_subscript`
(host copy on slot install) is ~8.5% of engine samples.

## 5. What this means
- Measured miss-proportional cost (0.80 ms/expert) and microbench (0.69 ms read +
  0.30 ms install per expert at k≈1.2/layer) agree to ~20%.
- Neither SSD bandwidth (drive does 3.8–5.6 GB/s on this pattern) nor page cache
  is the limit. The ~40 ms/token miss term is serialized latency; overlapping it
  with the ~39 ms/token fixed term (cross-layer prefetch) bounds decode at
  ~1000/max(39, 40) ≈ 25 tok/s at this residency (SIM upper bound, not measured).

## Caveats
- One residency (0.28); residency not varied (no server restart).
- iostat counts the whole boot disk; idle baseline 0.03–8 MB/s subtracted.
- r² 0.66 pooled; pass 1 alone r² 0.48 (chinese_story outlier at 91.7 ms / 142 MB).
- `collect_silicon_run.py` per_token_ms: chunks carry ~3 tokens (86 chunks / 256
  tokens), reported gaps p50 ~200 ms are chunk gaps, not token gaps — flagged to T4.

## 6. PLE n-gram traffic is negligible (cycle 3)
Source read (oMLX 0.7.0 vendored mlx_vlm qwen4_exp/language.py, `_SafeTensorMMap`):
PLE rows are gathered from an mmap (MADV_RANDOM) after a 48-thread `ple-io` pool
pre-touches unseen 16 KB pages with os.pread; a seen-page bitmap skips warm pages.
Config: ngram_size 3, heads_per_ngram 8, ple_layer_ids [2] (one PLE layer).
Measured (`experiments/pagecache_families.py`, mincore by tensor family, idle box):
PLE n-gram tables 0.59 of 32.0 GB in page cache; experts 2.42 of 69.4 GB; other 0.
`experiments/pagein_attribution.py` (mincore before/after one 256-tok decode):
PLE n-gram page-ins 0.62 MB/token vs expert page-ins >=14 MB/token (lower bound;
expert pages churn in/out within the window) vs iostat ~137 MB/token.
That window OVERLAPPED another agent's requests (flagged in the json), which can only
inflate the PLE figure, so **PLE <= ~0.6 MB/token = <0.5% of decode SSD traffic**.
Rejects the "unmodeled PLE gathers ~376 MB/token" hypothesis by measurement + source.
`proc_diskio.py` (proc_pid_rusage of the server pid) is committed as a tool; its one run
is contaminated and not interpreted.
