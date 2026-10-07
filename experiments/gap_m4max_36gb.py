"""Sim<->silicon gap at the measured M4 Max 36 GB operating point.

Replays the (synthetic, calibrated) holdout router traces through an LRU
cache that mirrors oMLX 0.7.0's ExpertCache (per layer, plain LRU, no
prefetch, misses resolved synchronously layer by layer) at the capacity oMLX
used (resident fraction 0.28 -> 143 experts/layer; 0.18 -> 92).

Two time models are compared against measured tok/s:
  sim_paging (shipped): tok/s = 1000 / max(compute, miss_bytes / SSD_BW)
      -> assumes misses stream at full SSD bandwidth and overlap compute.
  serial_latency (this file): per token
      compute_ms + sum over layers with misses of (A + B*k)   # IO, serial
      + misses * install_ms + L * sync_ms
      A, B, install_ms, sync_ms are MEASURED on M4 Max 36 GB by
      experiments/microbench_expert_reads.py (F_NOCACHE preads, 12 threads,
      9 slabs/expert, 2.765 MB/expert).

compute_ms is NOT measured on the 36 GB box (it cannot hold the model); it is
the 128 GB measured 57.4 tok/s (17.4 ms) scaled by DRAM bandwidth 546->410
GB/s. That scaling is an assumption.

usage: python3 experiments/gap_m4max_36gb.py
"""
import argparse, json, sys
from collections import OrderedDict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import sim_paging as sp  # noqa: E402

EXPERT_MB = 67.95e3 / (48 * 512)   # 2.765 MB, oMLX log; microbench agrees
SSD_GBPS = 7.4
COMPUTE_MS = 1000 / 57.4 * 546 / 410   # 23.2 ms; assumption, see docstring
# microbench_expert_reads.py on M4 Max 36 GB 2026-10-06 (cold, F_NOCACHE):
# per-layer k-miss read latency: k=1 0.72, k=2 1.26, k=3 1.78, k=4 2.28 ms
IO_A_MS, IO_B_MS = 0.20, 0.52
INSTALL_MS = 0.30    # np.frombuffer -> mx.array -> slot write (0.27-0.35)
SYNC_MS = 0.122      # tiny .tolist() device readback, once per MoE layer
WARMUP = 200


def lru_misses(gold, cap):
    caches = [OrderedDict() for _ in range(sp.L)]
    M = np.zeros(gold.shape[:2], np.int32)
    for t in range(len(gold)):
        for li in range(sp.L):
            c = caches[li]
            m = 0
            for e in set(gold[t, li].tolist()):
                if e in c:
                    c.move_to_end(e)
                else:
                    m += 1
                    c[e] = 1
                    if len(c) > cap:
                        c.popitem(last=False)
            M[t, li] = m
    return M[WARMUP:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-dir", default=str(ROOT / "results/traces_synth"))
    ap.add_argument("--measured", default=str(ROOT / "results/measured_m4max_36gb_ssd.json"))
    ap.add_argument("--out", default=str(ROOT / "results/gap_m4max_36gb.json"))
    a = ap.parse_args()

    _, hl, _ = sp.load_split(Path(a.traces_dir) / "holdout.npz")
    gold = np.stack([hl[li] for li in range(sp.L)], axis=1).astype(np.int16)
    meas = json.loads(Path(a.measured).read_text())
    meas = {k: v for k, v in meas.items() if k != "samples"}
    measured_tps = {143: (meas["decode_tps"], 13.0), 92: (None, 7.8)}

    rows = []
    for cap in (92, 143):
        M = lru_misses(gold, cap)
        miss = M.sum(1).astype(float)
        mb = miss * EXPERT_MB
        stream_ms = mb / (SSD_GBPS * 1e3) * 1e3
        io_ms = np.where(M > 0, IO_A_MS + IO_B_MS * M, 0.0).sum(1)
        inst_ms = miss * INSTALL_MS
        tok_ms = COMPUTE_MS + io_ms + inst_ms + sp.L * SYNC_MS
        row = dict(
            cap=cap,
            sim_miss_experts_per_token=round(miss.mean(), 1),
            sim_hit_rate=round(1 - miss.mean() / (sp.L * sp.K), 4),
            sim_ssd_MB_per_token=round(mb.mean(), 1),
            frac_layer_steps_with_miss=round(float((M > 0).mean()), 3),
            compute_ms_assumed=round(COMPUTE_MS, 1),
            shipped_model_tps=round(float(np.mean(1000 / np.maximum(COMPUTE_MS, stream_ms))), 1),
            serial_latency_breakdown_ms=dict(
                compute=round(COMPUTE_MS, 1), io=round(io_ms.mean(), 1),
                install=round(inst_ms.mean(), 1), sync=round(sp.L * SYNC_MS, 1)),
            serial_latency_model_tps=round(float(1000 / tok_ms.mean()), 1),
            measured_tps=measured_tps[cap][1],
        )
        if cap == 143:
            row["measured_ssd_MB_per_token"] = meas["decode_disk_MB_per_token"]
            row["measured_miss_experts_per_token_lower_bound"] = round(
                meas["decode_disk_MB_per_token"] / EXPERT_MB, 1)
            row["measured_tps_this_run"] = meas["decode_tps"]
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = dict(kind="simulated_vs_measured",
               note="synthetic calibrated traces (results/traces_synth); "
                    "compute_ms is an assumption (128GB measured, BW-scaled); "
                    "IO/install/sync constants measured on M4 Max 36 GB",
               constants=dict(expert_MB=round(EXPERT_MB, 3), ssd_GBps=SSD_GBPS,
                              io_A_ms=IO_A_MS, io_B_ms=IO_B_MS,
                              install_ms=INSTALL_MS, sync_ms=SYNC_MS),
               rows=rows, measured=meas)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
