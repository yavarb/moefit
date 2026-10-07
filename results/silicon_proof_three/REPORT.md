# Silicon proof — three invention wins

Executor: chief-silicon-exec (Grok Bot). Host: Santa Cruz. Date: 2026-10-06 evening ET.
oMLX 0.7.0; offload md5 d420b305 → 214e8823 (sidecar+LIP code merged). :8317 untouched. Lab agents left running.

## Compact table

| Win | Sim claim | Silicon measured | vs baseline | Verdict | Merge-worthy? |
|-----|-----------|------------------|-------------|---------|---------------|
| T6 coalescer | ~10.5 GB/s (1.91×) **cached** — invalid SSD | Physical SSD useful/phys: OFF **4.67** / ON **4.44** GB/s (phys/useful≈1.5) | vs 5.02 peak: 0.93×/0.88×; ON/OFF **0.95×** | **FAIL** | **No** |
| T4/T9 dbuf | sim 11.85→13.00; amend2 +0.33..+1.10 install-only | ON **15.75** / ON2 **15.64** vs OFF(workers=1) **9.2**; Δ med **+6.54** | vs 15.55: ON≈baseline; OFF −6.35 | **TRANSFERS** | **Yes** (keep default IO pool) |
| T3 LIP | WASH −0.05..+0.25 on 15.55 | **BLOCKER** (HTTP 507 after T2; partial ON 15.66/14.49 only) | — | **BLOCKER** | pending |

## Details

### (1) T6 coalescer — FAIL on real SSD
- Method: real checkpoint expert extents, F_NOCACHE+F_RDAHEAD=0, IOBlockStorageDriver APPLE SSD `Bytes (Read)`.
- Synthetic F_NOCACHE fixture gave physical≈0 (FAIL_CACHE_ONLY) — discarded for verdict.
- Real: physical tracks useful (~1.5×); coalescer does **not** beat serial.
- Artifacts: `t6_physical_real_ckpt.json`, `t6_score.json`, `t6_physical_coalescer.json`.

### (2) T4/T9 double-buffer / staged IO — TRANSFERS
- Inventory: CASE B machinery present (env `OMLX_MOE_OFFLOAD_IO_WORKERS`); default ON=12.
- A/B/A via T9 collect: ON → OFF(workers=1) → ON2; n=1024×3; stream_integrity eligible; finish=length.
- Paired Δ median +6.54 tps ≫ +0.30 rule → TRANSFERS.
- Caveat: OFF disables entire IO pool/read-ahead (not install-staging alone); Δ exceeds install-only band +0.33..+1.10 → fetch-parallelism dominates measured win.
- Text bit-exactness: collector stores token **counts** not text — not verified from blobs (greedy T=0).
- Artifacts: `t9_on_A.json`, `t9_off_B.json`, `t9_on_A2.json`, `t4_dbuf_score.json`.

### (3) T3 LIP — BLOCKER
- Code live (`OMLX_ADMISSION`, `_miss_hist`) in 214e8823.
- Partial LIP ON: 15.66, 14.49 tps then aborted (T2 lock / shutdown).
- Subsequent arms: HTTP 507 (dynamic ceiling ~18GB < 23.77GB model) until long purge; stock restored OK.
- SCORING REFUSED. Expected WASH if completed. Artifact: `t3_lip_score.json`, `t3_lip_on_partial.json`.

## Paths
`/tmp/moefit-lab/repo/results/silicon_proof_three/`
