"""Reference copy of the ORIGINAL simulate()/solve_policy() from
experiments/sim_paging.py at commit 53c38bf (the code that produced
results/sim_paging.json). Do not edit. tests/test_sim_equivalence.py checks
that the current simulator reproduces these counts exactly.
"""
import numpy as np

EXPERT_MIB = 1.46
L, K, E = 48, 10, 512
GOLD_MIB = L * K * EXPERT_MIB
FLOOR_MIB = 2900.0
PLE_STREAM_MIB = 0.3

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


