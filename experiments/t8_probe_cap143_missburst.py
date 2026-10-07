"""T8 probe (sheryl_local_exp, LOCAL): verify the cap-143 matched sim rows and
add per-layer / per-token miss structure the 2-term sim model averages away.

SIMULATED numbers on results/traces_synth. No silicon numbers invented.
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path("/tmp/moefit-lab/repo")
sys.path.insert(0, str(Path(__file__).resolve().parent))   # scratch copy (clean sim)
import sim_paging as sp

tdir = ROOT / "results" / "traces_synth"
bf, bl, bT = sp.load_split(tdir / "build.npz")
hf, hl, hT = sp.load_split(tdir / "holdout.npz")
rng = np.random.default_rng(0)
hs = np.sort(rng.choice(hT, size=min(8000, hT), replace=False))
gold = np.stack([hl[li][hs] for li in range(sp.L)], axis=1).astype(np.int16)
T = len(hs)
prior_rank = {li: np.argsort(-np.bincount(bl[li].ravel(), minlength=sp.E))
              for li in range(sp.L)}
print(f"T={T} rows, cap=143", flush=True)

SPEC = dict(dram=546.0, ssd=7.4, usable=24.0)   # 36GB tier used by gap doc
CAP = 143

def miss_matrix(cap, mode):
    """Re-run simulate() internals cheaply: count per-layer, per-token sync
    misses by instrumenting a fresh pass (mirrors simulate lru path)."""
    caps = sp.per_layer_caps(cap)
    res = [set() for _ in range(sp.L)]
    lru = [dict() for _ in range(sp.L)]
    import heapq
    heap = [[] for _ in range(sp.L)]
    seq = [0]
    def set_ts(li, e, ts):
        old = lru[li].get(e)
        if old is None:
            seq[0] += 1; ent = (ts, seq[0])
        else:
            ent = (ts, old[1])
        lru[li][e] = ent
        heapq.heappush(heap[li], (ent[0], ent[1], e))
    def evict(li):
        while len(lru[li]) > caps[li]:
            ts, sq, e = heapq.heappop(heap[li])
            cur = lru[li].get(e)
            if cur is None or cur != (ts, sq): continue
            res[li].discard(e); del lru[li][e]
    misses = np.zeros((T, sp.L), np.int16)
    served = 0
    for t in range(T):
        for li in range(sp.L):
            for e in gold[t, li]:
                e = int(e)
                if e in res[li]:
                    served += 1
                else:
                    res[li].add(e); set_ts(li, e, t); evict(li)
                    misses[t, li] += 1
    return served / (T * sp.L * sp.K), misses

t0 = time.time()
# 1) reproduce matched rows (lru, prior) via the shipped solver
for mode in ("lru", "prior"):
    srv, sync_mb, async_mb, t, c_ms, st_ms = sp.solve_policy(
        gold, None, prior_rank, CAP, mode, 0, SPEC, throttle="legacy")
    print(f"{mode}: served={srv:.3f} sync={sync_mb:.1f}MB async={async_mb:.1f}MB "
          f"tps={t:.1f} c={c_ms:.1f} st={st_ms:.1f}", flush=True)

# 2) miss structure (LRU, cap 143)
srv, misses = miss_matrix(CAP, "lru")
tot = misses.sum(1)                      # misses/token (out of 480)
tot_mb = tot * sp.EXPERT_MIB
mean_m = tot_mb.mean()
p50, p90, p99 = np.percentile(tot_mb, [50, 90, 99])
per_layer = misses.mean(0)               # avg misses/token per layer
print(f"LRU cap143 served={srv:.3f} (verify ~0.84)")
print(f"SSD sync MB/token: mean={mean_m:.1f} p50={p50:.1f} p90={p90:.1f} p99={p99:.1f} max={tot_mb.max():.1f}")
print(f"per-layer misses/token: min={per_layer.min():.2f} (L{per_layer.argmin()}) "
      f"max={per_layer.max():.2f} (L{per_layer.argmax()}) cv={per_layer.std()/per_layer.mean():.2f}")
# SSD-bound token time at tier BW vs implied-silicon BW
for bw in (7.4, 3.0, 2.63):
    st_mean = mean_m/1024/bw*1000
    st_p90 = p90/1024/bw*1000
    print(f"ssd={bw}GB/s: stream_ms mean={st_mean:.1f} p90={st_p90:.1f} "
          f"-> mean-bound tps={1000/max(st_mean,18.1):.1f}, "
          f"p90-bound tps={1000/max(st_p90,18.1):.1f}")
print(f"elapsed {time.time()-t0:.0f}s", flush=True)

out = dict(label="T8 probe cap143 LRU miss structure on traces_synth (SIMULATED)",
           T=int(T), served=round(srv, 4),
           sync_mb_token=dict(mean=round(mean_m, 1), p50=round(p50, 1),
                              p90=round(p90, 1), p99=round(p99, 1),
                              max=round(float(tot_mb.max()), 1)),
           per_layer=dict(min=round(float(per_layer.min()), 3),
                          max=round(float(per_layer.max()), 3),
                          cv=round(float(per_layer.std()/per_layer.mean()), 3)),
           sim_tier=SPEC)
Path(__file__).with_name("t8_probe_out.json").write_text(json.dumps(out, indent=1))
print("wrote t8_probe_out.json")
