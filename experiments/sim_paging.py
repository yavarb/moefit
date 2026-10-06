"""Phase-1 simulator v2: expert paging policies for small-RAM Macs.

Regime (measured from the checkpoint): routed experts 35.9 GiB total
(1.46 MiB/expert, 512/layer x 48 layers); non-expert floor 2.9 GiB
(attn/shared/rest) stays DRAM-resident; PLE rows stream on demand.
Decode touches 10 experts/layer/token = ~701 MiB of expert weights.

Byte accounting is event-driven inside the simulation:
  sync  = expert read from SSD at hit time (blocks the token)
  async = expert read from SSD at prefetch time (hidden if SSD keeps up)
Resident experts are re-read from DRAM every token they're used (that's
the compute term — Apple Silicon reads weights from unified DRAM anyway).

Coupled steady state per token:
  compute_ms = (floor + served_frac*701MiB) / DRAM_BW
  stream_ms  = (sync + async + PLE) / SSD_BW
  tok/s = 1000 / max(compute_ms, stream_ms)

Policies (resident capacity C experts per layer):
  lru     : pure LRU cache, no prediction
  prior   : pinned static hot set (build prior) + LRU
  probe   : LRU(C-P) + PLE-probe top-P prefetched for token t+1 (the
            token id is known when t is emitted — legal lookahead)
  sidecar : replayed routing sidecar = knows t+1's gold exactly
            (prefix-cache regime; upper bound)

usage: .venv/bin/python experiments/sim_paging.py
"""
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERT_MIB = 1.46
L, K, E = 48, 10, 512
GOLD_MIB = L * K * EXPERT_MIB
FLOOR_MIB = 2900.0
PLE_STREAM_MIB = 0.3

TIERS = {
    "24GB-M4":  dict(dram=135.0, ssd=5.0, usable=15.0),
    "32GB-M4P": dict(dram=273.0, ssd=6.0, usable=23.0),
    "48GB-M4M": dict(dram=546.0, ssd=7.4, usable=39.0),
    "64GB-M4X": dict(dram=546.0, ssd=7.4, usable=55.0),
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
    for li in range(L):
        Yb = np.zeros((len(bf), E), np.float32)
        Yb[np.arange(len(bf))[:, None], bl[li]] = 1.0
        mu, sd = bf.mean(0), bf.std(0) + 1e-6
        Xb = ((bf - mu) / sd).astype(np.float32)
        Xh = ((hf[hs] - mu) / sd).astype(np.float32)
        W = np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1], dtype=np.float32),
                            Xb.T @ Yb)
        s = Xh @ W
        out[:, li] = np.argpartition(-s, budget - 1, axis=1)[:, :budget]
    return out


def simulate(gold, picks, prior_rank, cap, mode, pick, async_allow):
    """One pass with hard capacity: res[li] never exceeds cap experts.
    pin[li] = predicted experts pinned for token t+1 (free SSD slots up to
    async_allow). Non-pinned residents evicted LRU. sync = expert read at
    hit time (stalls); async = read at prefetch time (hidden by design,
    charged to SSD)."""
    T = len(gold)
    res = [set() for _ in range(L)]
    lru = [dict() for _ in range(L)]
    pin = [set() for _ in range(L)]
    allow = async_allow if async_allow < 10 ** 8 else None
    n_pf_slot = min(pick, L * K) if mode == "probe" else (K if mode == "sidecar" else 0)
    dyn_cap = max(cap - n_pf_slot, 4)
    if mode == "prior":
        dyn_cap = max(cap // 2, 4)   # pin the other half: static hot set

    def prune(li):
        dyn = [x for x in lru[li] if x not in pin[li]]
        while len(dyn) > dyn_cap:
            v = min(dyn, key=lru[li].get)
            dyn.remove(v)
            res[li].discard(v)
            del lru[li][v]

    def touch_dyn(li, e, t):
        if e in res[li]:
            if e not in pin[li]:
                lru[li][e] = t
            return False
        res[li].add(e)
        lru[li][e] = t
        prune(li)
        return True

    served = sync = async_ = 0
    if mode == "prior":
        for li in range(L):
            for e in prior_rank[li][:cap - dyn_cap]:
                res[li].add(int(e))
                pin[li].add(int(e))
                lru[li][int(e)] = 1e18
    budget_left = 0
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
                    lru[li][e] = t
            # demote old pins: recompute lru classes, then prune per layer
            for li in range(L):
                pin[li] = newpin[li]
                prune(li)
    n = T * L * K
    return served / n, sync * EXPERT_MIB / T, async_ * EXPERT_MIB / T


DRAM_EFF = 0.385   # calibrated: M4 Max measured 57 tok/s fully resident
                   # => decode sustains ~210 of 546 GB/s peak


def solve_policy(gold, picks, prior_rank, cap, mode, pick, spec):
    """coupled fixed point over (resident set, SSD spare)."""
    async_allow = 10 ** 9
    for _ in range(6):
        srv_f, sync_mb, async_mb = simulate(
            gold, picks, prior_rank, cap, mode, pick, async_allow)
        dram_mb = FLOOR_MIB + GOLD_MIB + async_mb
        c_ms = dram_mb / 1024.0 / (spec["dram"] * DRAM_EFF) * 1000.0
        st_ms = (sync_mb + async_mb + PLE_STREAM_MIB) / 1024.0 \
            / spec["ssd"] * 1000.0
        s_ms = max(c_ms, st_ms)
        spare_mb = max(spec["ssd"] * 1024.0 / 1000.0 * s_ms
                       - (sync_mb + PLE_STREAM_MIB), 0.0)
        new_allow = int(spare_mb / EXPERT_MIB)
        if abs(new_allow - async_allow) < 8:
            async_allow = new_allow
            break
        async_allow = new_allow
    t = 1000.0 / s_ms
    return srv_f, sync_mb, async_mb, t, c_ms, st_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-sub", type=int, default=8000)
    ap.add_argument("--probe-picks", default="6,12")
    ap.add_argument("--caps", default="32,64,128,192,384,512")
    a = ap.parse_args()

    bf, bl, bT = load_split(ROOT / "results/traces/build.npz")
    hf, hl, hT = load_split(ROOT / "results/traces/holdout.npz")
    rng = np.random.default_rng(0)
    hs = np.sort(rng.choice(hT, size=min(a.eval_sub, hT), replace=False))
    gold = np.stack([hl[li][hs] for li in range(L)], axis=1).astype(np.int16)

    prior_rank = {li: np.argsort(-np.bincount(bl[li].ravel(), minlength=E))
                  for li in range(L)}
    picks_map = {P: probe_picks(bf.astype(np.float32), bl, hf, hs, P)
                 for P in [int(x) for x in a.probe_picks.split(",")]}

    out = []
    for cap in [int(x) for x in a.caps.split(",")]:
        for P in sorted(picks_map):
            for mode in ("lru", "prior", "probe", "sidecar"):
                p = P if mode == "probe" else 0
                row = dict(cap=cap, mode=mode, probe_picks=P)
                for tier, spec in TIERS.items():
                    need_gib = FLOOR_MIB / 1024 + cap * L * EXPERT_MIB / 1024
                    if need_gib > spec["usable"]:
                        continue
                    srv_f, sync_mb, async_mb, t, c_ms, st_ms = \
                        solve_policy(gold, picks_map[P], prior_rank, cap,
                                     mode, p, spec)
                    row[f"served_{tier}"] = round(srv_f, 3)
                    row[f"tps_{tier}"] = round(t, 1)
                    row[f"ct_{tier}"] = round(c_ms, 1)
                    row[f"st_{tier}"] = round(st_ms, 1)
                    row[f"ssdMB_{tier}"] = round(sync_mb + async_mb, 0)
                out.append(row)
                print(json.dumps(row), flush=True)
    (ROOT / "results/sim_paging.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
