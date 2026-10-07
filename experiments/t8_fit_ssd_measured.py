"""T8 probe: fit the SSD-queue model to the NEW iostat measurement.

Measured (results/measured_santa_cruz_36gb_ssd.json, cap143, n=256, idle box):
  12.71 tok/s, 1786 MB/s physical disk reads during decode,
  140.5 MB/token (idle-subtracted), ~10.7k IOPS, avg IO 171 KB.

Model: emit[i] = max(cumsum(svc)+C, emit[i-1]+C); svc = miss_MB/ssd_gbps.
Traffic shapes: (a) shipped-LRU sim misses rescaled to measured mean,
(b) true_lru sim misses (hit-refresh fix) rescaled, (c) raw sim traffic.
Fits one knob: ssd effective BW, scored on n128+n256 from the 128-run set.
SIMULATED except where marked measured.
"""
import json
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_paging as sp

ROOT = Path("/tmp/moefit-lab/repo")
MEAS = json.load(open(ROOT / "results/measured_santa_cruz_36gb_ssd.json"))
D_MB, MBPS, KBIO = MEAS["decode_disk_MB_per_token"], MEAS["decode_disk_MBps"], MEAS["decode_avg_KB_per_io"]
C = 18.1

# miss matrices (same loader/seed as prior T8 probes — locked trace)
bf, bl, bT = sp.load_split(ROOT / "results/traces_synth/build.npz")
hf, hl, hT = sp.load_split(ROOT / "results/traces_synth/holdout.npz")
rng = np.random.default_rng(0)
hs = np.sort(rng.choice(hT, size=min(8000, hT), replace=False))
gold = np.stack([hl[li][hs] for li in range(sp.L)], axis=1).astype(np.int16)
T = len(hs)

import heapq
def miss_matrix(cap, refresh):
    caps = sp.per_layer_caps(cap)
    res = [set() for _ in range(sp.L)]; lru = [dict() for _ in range(sp.L)]
    heap = [[] for _ in range(sp.L)]; seq = [0]
    m = np.zeros(T, np.int32)
    for t in range(T):
        for li in range(sp.L):
            for e in gold[t, li]:
                e = int(e)
                if e in res[li]:
                    if refresh:  # true LRU: touch on hit
                        old = lru[li][e]
                        ent = (t, old[1]); lru[li][e] = ent
                        heapq.heappush(heap[li], (ent[0], ent[1], e))
                    continue
                res[li].add(e)
                old = lru[li].get(e); seq[0] += 1 if old is None else 0
                ent = (t, seq[0] if old is None else old[1]); lru[li][e] = ent
                heapq.heappush(heap[li], (ent[0], ent[1], e)); m[t] += 1
                while len(lru[li]) > caps[li]:
                    ts, sq, x = heapq.heappop(heap[li]); cur = lru[li].get(x)
                    if cur is None or cur != (ts, sq): continue
                    res[li].discard(x); del lru[li][x]
    return m * sp.EXPERT_MIB

def ramp(svc_ms, n):
    ssd = np.cumsum(svc_ms); emit = np.empty(len(svc_ms)); prev = 0.0
    for i in range(len(svc_ms)):
        prev = max(ssd[i] + C, prev + C); emit[i] = prev
    return (n - 1) * 1000.0 / (emit[n - 1] - emit[0])

shapes = {
    "lru_raw":      miss_matrix(143, False),                 # 206.8 MB/tok mean
    "true_lru_raw": miss_matrix(143, True),                  # 156.9 MB/tok mean
}
for k in list(shapes):
    shapes[k + "_scaled"] = shapes[k] * (D_MB / shapes[k].mean())
shapes["flat140"] = np.full(T, D_MB)   # measured mean, no burstiness

meas = {"n16": 6.43, "n128": 13.0, "n256": 12.71}
out = {"measured": MEAS and dict(decode_tps=12.71, disk_MB_per_tok=D_MB,
        disk_MBps=MBPS, iops=MEAS["decode_iops"], avg_KB_per_io=KBIO),
       "measured_ramp": meas, "fits": []}
for name, mb in shapes.items():
    best = None
    for bw in np.arange(1.2, 7.5, 0.02):
        svc = mb / bw
        r = dict(bw=round(float(bw), 2), n16=round(ramp(svc, 16), 2),
                 n128=round(ramp(svc, 128), 2), n256=round(ramp(svc, 256), 2),
                 steady=round(ramp(svc, T), 2))
        err = ((r["n128"] - 12.8) / 12.8) ** 2 + ((r["n256"] - 12.71) / 12.71) ** 2
        if best is None or err < best[-1]:
            best = (r, err)
    r, err = best
    r.update(shape=name, mean_MB=round(float(mb.mean()), 1), sse=round(err, 5))
    out["fits"].append(r)
    print(json.dumps(r), flush=True)

# IO decomposition from measured: IOPS*tok, IO size
ios_per_tok = D_MB * 1024 / KBIO
out["io_decomposition"] = dict(
    IOs_per_token=round(ios_per_tok, 0), avg_KB_per_io=KBIO,
    note="~839 physical IOs/token at ~171 KB avg — NOT 2.7 MiB expert-slab-sized "
         "reads; effective 1.79 GB/s at this IO size on an NVMe rated ~7.4 GB/s "
         "sequential (~26% of spec).",
    per_IO_overhead_feasibility={
        f"bw{bw}": f"serial overhead {((1000/12.71 - D_MB/bw)/ios_per_tok)*1000:.0f} us x {ios_per_tok:.0f} IOs + bytes {D_MB/bw:.1f} ms = 78.7 ms/tok"
        for bw in (3.0, 5.0, 7.4)})
Path(__file__).with_name("t8_ssd_fit.json").write_text(json.dumps(out, indent=1))
print("wrote t8_ssd_fit.json")
