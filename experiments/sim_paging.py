"""Phase-1 simulator v2: expert paging policies for small-RAM Macs.

Regime (measured from the checkpoint): routed experts 35.9 GiB total
(1.46 MiB/expert, 512/layer x 48 layers); non-expert floor 2.9 GiB
(attn/shared/rest) stays DRAM-resident; PLE rows stream on demand.
Decode touches 10 experts/layer/token = ~701 MiB of expert weights.

Byte accounting is event-driven inside the simulation:
  sync  = expert read from SSD at hit time (blocks the token)
  async = expert read from SSD at prefetch time (hidden if SSD keeps up)
Every expert a token uses is read from DRAM whether it was resident or
just arrived (Apple Silicon reads weights from unified DRAM anyway), so
the DRAM term is the full 701 MiB/token plus prefetched bytes landing.

Coupled steady state per token:
  compute_ms = (floor + 701 MiB + async) / (DRAM_BW * DRAM_EFF)
  stream_ms  = (sync + async + PLE) / SSD_BW
  tok/s = 1000 / max(compute_ms, stream_ms)

Policies (resident capacity C experts per layer):
  lru     : pure LRU cache, no prediction
  prior   : pinned static hot set (build prior) + LRU
  probe   : LRU(C-P) + PLE-probe top-P prefetched for token t+1 (the
            token id is known when t is emitted - legal lookahead)
  sidecar : replayed routing sidecar = knows t+1's gold exactly
            (prefix-cache regime; upper bound)

Prefetch throttle (--throttle):
  legacy  : the allowance is solved from the SSD time of the whole token,
            max(compute, stream). When the SSD is already the bottleneck
            this reduces to "allow what you just issued", so prefetch is
            never cut back; it models a prefetcher that does not watch
            SSD queue depth. This produced results/sim_paging.json.
  compute : the allowance is the SSD capacity left inside the compute
            window after sync misses, so async reads stay hidden. Models
            an adaptive prefetcher. See tests/test_sim_throttle.py.

simulate() accepts either one capacity for every layer (the shipped
table) or a list of 48 per-layer capacities (used by
experiments/hetero_alloc.py). Capacity is audited every token: no layer
may hold more than its cap, or the simulator raises.

usage: .venv/bin/python experiments/sim_paging.py
       .venv/bin/python experiments/sim_paging.py --traces-dir results/traces_synth --out results/sim_paging_synth.json
"""
import argparse, heapq, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Geometry measured from the checkpoint's safetensors data_offsets
# (U32-packed 4-bit weights counted correctly): routed experts 64.6 GiB
# over 512/layer x 48, PLE n-gram table 29.8 GiB, non-expert floor
# 4.6 GiB, total 99.0 GiB on disk. An earlier revision halved packed
# tensors (dtype-table default of 2 bytes instead of U32=4).
EXPERT_MIB = 2.69
L, K, E = 48, 10, 512
GOLD_MIB = L * K * EXPERT_MIB      # expert bytes consumed per token
FLOOR_MIB = 4.6 * 1024             # resident non-expert footprint
# per-token DRAM READ from the floor side: attn 2.0 + other 1.09 +
# shared 0.23 + lm_head ~0.63 GiB (embed table resident but gather-read)
READ_FLOOR_MIB = 4.05 * 1024
PLE_STREAM_MIB = 0.3

# usable = 0.75 x RAM (macOS default GPU/VM working-set ceiling)
# minus 3 GiB headroom. dram = Apple-published peak GB/s.
TIERS = {
    "24GB-M4":  dict(dram=120.0, ssd=5.0, usable=15.0),
    "32GB-M4P": dict(dram=273.0, ssd=6.0, usable=21.0),
    "48GB-M4M": dict(dram=546.0, ssd=7.4, usable=33.0),
    "64GB-M4X": dict(dram=546.0, ssd=7.4, usable=45.0),
}


def load_split(path):
    d = np.load(path)
    tags = sorted({k.rsplit("|", 1)[0] for k in d.files if k.endswith("|ids")})
    feats = np.concatenate([d[f"{t}|PLE_ngram"] for t in tags])
    lay = {li: np.concatenate([d[f"{t}|L{li}_idx"] for t in tags])
           for li in range(L)}
    T = min(len(feats), len(lay[0]))
    return feats[:T], lay, T


def probe_picks(bf, bl, hf, hs, budget, lam=200.0):
    out = np.zeros((len(hs), L, budget), np.int16)
    mu, sd = bf.mean(0), bf.std(0) + 1e-6
    Xb = ((bf - mu) / sd).astype(np.float32)
    Xh = ((hf[hs] - mu) / sd).astype(np.float32)
    G = Xb.T @ Xb + lam * np.eye(Xb.shape[1], dtype=np.float32)
    for li in range(L):
        Yb = np.zeros((len(bf), E), np.float32)
        Yb[np.arange(len(bf))[:, None], bl[li][:len(bf)]] = 1.0
        W = np.linalg.solve(G, Xb.T @ Yb)
        s = Xh @ W
        out[:, li] = np.argpartition(-s, budget - 1, axis=1)[:, :budget]
    return out


def per_layer_caps(cap):
    """int -> uniform list; sequence -> list of 48 ints."""
    if np.isscalar(cap):
        return [int(cap)] * L
    caps = [int(c) for c in cap]
    assert len(caps) == L, "need one cap per layer"
    return caps


def simulate(gold, picks, prior_rank, cap, mode, pick, async_allow,
             audit=True):
    """One pass with hard capacity: res[li] never exceeds caps[li] experts.
    pin[li] = predicted experts pinned for token t+1 (free SSD slots up to
    async_allow). Non-pinned residents evicted LRU. sync = expert read at
    hit time (stalls); async = read at prefetch time (hidden by design,
    charged to SSD).

    LRU bookkeeping: each resident carries (last-use token, insertion
    sequence). The victim is the non-pinned resident with the smallest
    pair, which is exactly what the original list-scan
    min(dyn, key=lru.get) chose (ties broken by dict insertion order).
    A lazy heap makes that O(log n) per access instead of O(cap).
    Equivalence is checked by tests/test_sim_equivalence.py.
    """
    T = len(gold)
    caps = per_layer_caps(cap)
    res = [set() for _ in range(L)]
    lru = [dict() for _ in range(L)]      # e -> (ts, seq)
    heap = [[] for _ in range(L)]
    pin = [set() for _ in range(L)]
    seq = [0]
    allow = async_allow if async_allow < 10 ** 8 else None
    n_pf_slot = min(pick, L * K) if mode == "probe" else (K if mode == "sidecar" else 0)
    dyn_cap = [max(c - n_pf_slot, 4) for c in caps]
    if mode == "prior":
        dyn_cap = [max(c // 2, 4) for c in caps]   # pin the other half

    def set_ts(li, e, ts):
        old = lru[li].get(e)
        if old is None:
            seq[0] += 1
            ent = (ts, seq[0])
        else:
            ent = (ts, old[1])
        lru[li][e] = ent
        heapq.heappush(heap[li], (ent[0], ent[1], e))

    def n_dyn(li):
        return len(lru[li]) - sum(1 for p in pin[li] if p in lru[li])

    def prune(li):
        h = heap[li]
        kept = []
        while n_dyn(li) > dyn_cap[li]:
            ts, sq, e = heapq.heappop(h)
            cur = lru[li].get(e)
            if cur is None or cur != (ts, sq):
                continue                      # stale heap entry
            if e in pin[li]:
                kept.append((ts, sq, e))      # pinned: keep its order
                continue
            res[li].discard(e)
            del lru[li][e]
        for item in kept:
            heapq.heappush(h, item)

    def touch_dyn(li, e, t):
        if e in res[li]:
            if e not in pin[li]:
                set_ts(li, e, t)
            return False
        res[li].add(e)
        set_ts(li, e, t)
        prune(li)
        return True

    served = sync = async_ = 0
    if mode == "prior":
        for li in range(L):
            for e in prior_rank[li][:caps[li] - dyn_cap[li]]:
                e = int(e)
                res[li].add(e)
                pin[li].add(e)
                set_ts(li, e, 1e18)
    for t in range(T):
        for li in range(L):
            for e in gold[t, li]:
                e = int(e)
                if e in res[li]:
                    served += 1
                else:
                    touch_dyn(li, e, t)
                    sync += 1
        if t + 1 < T and n_pf_slot:
            newpin = [set() for _ in range(L)]
            used = 0
            for li in range(L):
                src = picks[t + 1, li][:n_pf_slot] if mode == "probe" \
                    else gold[t + 1, li]
                for e in src:
                    e = int(e)
                    newpin[li].add(e)
                    if e in res[li]:
                        continue
                    if allow is not None and used >= allow:
                        continue
                    used += 1
                    async_ += 1
                    res[li].add(e)
                    set_ts(li, e, t)
            for li in range(L):
                pin[li] = newpin[li]
                prune(li)
        if audit:
            for li in range(L):
                if len(res[li]) > caps[li]:
                    raise AssertionError(
                        f"capacity leak: layer {li} holds {len(res[li])} > "
                        f"{caps[li]} at token {t} (mode={mode})")
    n = T * L * K
    return served / n, sync * EXPERT_MIB / T, async_ * EXPERT_MIB / T


DRAM_EFF = 293.0 / 546.0
# calibration: measured 57.4 tok/s on M4 Max 128 GB fully resident, at
# 4.75 GiB/token actual DRAM reads (measured geometry) => 293 GB/s
# sustained effective bandwidth (54% of peak).


def _times(spec, sync_mb, async_mb):
    dram_mb = READ_FLOOR_MIB + GOLD_MIB + async_mb
    c_ms = dram_mb / 1024.0 / (spec["dram"] * DRAM_EFF) * 1000.0
    st_ms = (sync_mb + async_mb + PLE_STREAM_MIB) / 1024.0 \
        / spec["ssd"] * 1000.0
    return c_ms, st_ms


def solve_policy(gold, picks, prior_rank, cap, mode, pick, spec,
                 throttle="legacy"):
    """coupled fixed point over (resident set, SSD spare).

    throttle="legacy" reproduces the shipped solver exactly (6 iterations,
    window = max(compute, stream)). throttle="compute" sizes the prefetch
    allowance to the SSD time left inside the compute window.
    """
    async_allow = 10 ** 9
    if throttle == "legacy":
        for _ in range(6):
            srv_f, sync_mb, async_mb = simulate(
                gold, picks, prior_rank, cap, mode, pick, async_allow)
            c_ms, st_ms = _times(spec, sync_mb, async_mb)
            s_ms = max(c_ms, st_ms)
            spare_mb = max(spec["ssd"] * 1024.0 / 1000.0 * s_ms
                           - (sync_mb + PLE_STREAM_MIB), 0.0)
            new_allow = int(spare_mb / EXPERT_MIB)
            if abs(new_allow - async_allow) < 8:
                async_allow = new_allow
                break
            async_allow = new_allow
    elif throttle == "compute":
        for it in range(12):
            srv_f, sync_mb, async_mb = simulate(
                gold, picks, prior_rank, cap, mode, pick, async_allow)
            c_ms, st_ms = _times(spec, sync_mb, async_mb)
            spare_mb = max(spec["ssd"] * 1024.0 / 1000.0 * c_ms
                           - (sync_mb + PLE_STREAM_MIB), 0.0)
            new_allow = int(spare_mb / EXPERT_MIB)
            if async_allow < 10 ** 8 and abs(new_allow - async_allow) < 8:
                break
            if async_allow < 10 ** 8 and it > 0:
                new_allow = (new_allow + async_allow) // 2   # damping
            async_allow = new_allow
        s_ms = max(c_ms, st_ms)
    else:
        raise ValueError(throttle)
    t = 1000.0 / s_ms
    return srv_f, sync_mb, async_mb, t, c_ms, st_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-sub", type=int, default=8000)
    ap.add_argument("--probe-picks", default="6,12")
    ap.add_argument("--caps", default="32,64,128,192,384,512")
    ap.add_argument("--modes", default="lru,prior,probe,sidecar")
    ap.add_argument("--traces-dir", default=str(ROOT / "results/traces"))
    ap.add_argument("--throttle", choices=("legacy", "compute"),
                    default="legacy")
    ap.add_argument("--out", default=None,
                    help="output JSON; defaults to results/sim_paging.json "
                         "only for the shipped configuration")
    a = ap.parse_args()

    default_cfg = (a.traces_dir == str(ROOT / "results/traces")
                   and a.throttle == "legacy")
    out_path = Path(a.out) if a.out else (
        ROOT / "results/sim_paging.json" if default_cfg
        else ROOT / f"results/sim_paging_{a.throttle}.json")

    tdir = Path(a.traces_dir)
    bf, bl, bT = load_split(tdir / "build.npz")
    hf, hl, hT = load_split(tdir / "holdout.npz")
    rng = np.random.default_rng(0)
    hs = np.sort(rng.choice(hT, size=min(a.eval_sub, hT), replace=False))
    gold = np.stack([hl[li][hs] for li in range(L)], axis=1).astype(np.int16)
    print(f"build rows={bT} holdout rows={hT} evaluated={len(hs)}", flush=True)

    prior_rank = {li: np.argsort(-np.bincount(bl[li].ravel(), minlength=E))
                  for li in range(L)}
    modes = a.modes.split(",")
    picks_map = {P: probe_picks(bf.astype(np.float32), bl, hf, hs, P)
                 for P in [int(x) for x in a.probe_picks.split(",")]} \
        if "probe" in modes else {0: None}

    out = []
    for cap in [int(x) for x in a.caps.split(",")]:
        for P in sorted(picks_map):
            for mode in modes:
                p = P if mode == "probe" else 0
                row = dict(cap=cap, mode=mode, probe_picks=P,
                           throttle=a.throttle)
                for tier, spec in TIERS.items():
                    need_gib = FLOOR_MIB / 1024 + cap * L * EXPERT_MIB / 1024
                    if need_gib > spec["usable"]:
                        continue
                    srv_f, sync_mb, async_mb, t, c_ms, st_ms = \
                        solve_policy(gold, picks_map[P], prior_rank, cap,
                                     mode, p, spec, throttle=a.throttle)
                    row[f"served_{tier}"] = round(srv_f, 3)
                    row[f"tps_{tier}"] = round(t, 1)
                    row[f"ct_{tier}"] = round(c_ms, 1)
                    row[f"st_{tier}"] = round(st_ms, 1)
                    row[f"ssdMB_{tier}"] = round(sync_mb + async_mb, 0)
                out.append(row)
                print(json.dumps(row), flush=True)
    out_path.write_text(json.dumps(out, indent=1))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
