"""design=admission-cache: frequency-aware admission / eviction for the oMLX
per-layer expert cache, scored under the MEASURED serial-latency cost model.

Why: lead_silicon showed (results/gap_m4max_36gb.json, microbench measured on
M4 Max 36 GB) that oMLX resolves misses serially per layer:
    token_ms = compute + sum_layers_with_miss (A + B*k) + misses*install + L*sync
so at 36 GB every avoided miss saves ~0.82 ms (0.52 IO + 0.30 install) and
every layer-step without any miss saves another 0.20 ms. The cache policy is
plain LRU (OrderedDict move_to_end). This file asks: how far from optimal is
LRU, and do new policies close that gap?

Policies (per layer, capacity C, all misses must be loaded to compute; an
expert that is *not admitted* still uses one transient scratch slot taken
out of C, so total resident never exceeds C):
  lru      : oMLX ExpertCache (true LRU, hit refresh)                 baseline
  opt      : Belady MIN (evict farthest next use; offline)            bound
  lfuda    : LFU with dynamic aging (GreedyDual-LFU): key = L + freq,
             L = key of last victim. Captures Zipf popularity and adapts
             to prompt topic shifts.
  tinylfu  : W-TinyLFU: 1-slot... no: window LRU (WIN frac of C) + main
             SLRU (protected 80%). Window victims are admitted into main
             only if their decayed frequency beats main's LRU victim.
             Frequencies are exact per-layer counters halved every
             RESET tokens (stand-in for a count-min sketch, 512 experts so
             exact is cheap).
  arc      : ARC (adaptive recency/frequency, ghost lists) per layer.

No hyperparameter sweep: each policy runs with one textbook setting
(WIN=1%->clamped >=2 slots, protected=80%, RESET=10*C/K tokens).

Traces: results/traces_synth (synthetic, calibrated; no cross-layer
structure). All tok/s here are SIMULATED with measured constants.

usage: python3 experiments/design_admission_cache.py
"""
import argparse, json, hashlib, sys
from collections import OrderedDict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import sim_paging as sp  # noqa: E402
import gap_m4max_36gb as gs  # noqa: E402  (measured constants)

L, K, E = sp.L, sp.K, sp.E


# ---------------------------------------------------------------- policies
def run_lru(seq, C):
    c = OrderedDict(); out = []
    for row in seq:
        m = 0
        for e in row:
            if e in c:
                c.move_to_end(e)
            else:
                m += 1; c[e] = 1
                if len(c) > C:
                    c.popitem(last=False)
        out.append(m)
    return out


def run_opt(seq, C):
    T = len(seq)
    INF = 1 << 30
    nxt_tok = np.full(E, INF, np.int64)
    # next use (token index) of expert e after token t, built backwards
    nxt = [None] * T
    for t in range(T - 1, -1, -1):
        nxt[t] = {e: int(nxt_tok[e]) for e in seq[t]}
        for e in seq[t]:
            nxt_tok[e] = t
    res = {}                       # e -> next use token
    out = []
    import heapq
    for t in range(T):
        cur = seq[t]
        m = 0
        for e in cur:
            if e not in res:
                m += 1
        # update next-use for current experts, then evict farthest among
        # non-current residents until within cap
        for e in cur:
            res[e] = nxt[t][e]
        if len(res) > C:
            cs = set(cur)
            cand = sorted(((nu, e) for e, nu in res.items() if e not in cs),
                          reverse=True)
            for nu, e in cand[:len(res) - C]:
                del res[e]
        out.append(m)
    return out


def run_lfuda(seq, C):
    freq = {}; key = {}; Lage = 0.0; out = []
    for row in seq:
        cs = set(row); m = 0
        for e in row:
            if e in key:
                freq[e] += 1
            else:
                m += 1; freq[e] = 1
            key[e] = Lage + freq[e]
        while len(key) > C:
            v = min((x for x in key if x not in cs), key=lambda x: key[x])
            Lage = key[v]; del key[v]; del freq[v]
        out.append(m)
    return out


def run_tinylfu(seq, C, win_frac=0.01, prot_frac=0.8):
    W = max(2, int(round(C * win_frac)))
    Mcap = C - W
    Pcap = int(Mcap * prot_frac)
    win = OrderedDict(); prob = OrderedDict(); prot = OrderedDict()
    cnt = np.zeros(E, np.float32)
    reset = max(1, 10 * C // K); out = []
    for t, row in enumerate(seq):
        if t and t % reset == 0:
            cnt *= 0.5
        cs = set(row); m = 0
        for e in row:
            cnt[e] += 1
            if e in prot:
                prot.move_to_end(e)
            elif e in prob:
                del prob[e]; prot[e] = 1
                while len(prot) > Pcap:
                    d, _ = prot.popitem(last=False); prob[d] = 1
                    prob.move_to_end(d, last=False)
            elif e in win:
                win.move_to_end(e)
            else:
                m += 1; win[e] = 1
        # settle: window overflow -> candidate vs main victim
        while len(win) > W:
            cand = next((x for x in win if x not in cs), None)
            if cand is None:
                break
            del win[cand]
            if len(prob) + len(prot) < Mcap:
                prob[cand] = 1; continue
            victim = next((x for x in prob if x not in cs), None)
            if victim is None:
                victim = next((x for x in prot if x not in cs), None)
                if victim is None:
                    continue
                del prot[victim]
            elif cnt[cand] > cnt[victim]:
                del prob[victim]
            else:
                continue                  # reject candidate (evict it)
            prob[cand] = 1
        # window may exceed W only if all its members are current; then
        # borrow from main to keep the hard cap
        while len(win) + len(prob) + len(prot) > C:
            pool = prob if any(x not in cs for x in prob) else prot
            v = next(x for x in pool if x not in cs); del pool[v]
        out.append(m)
    return out


def run_arc(seq, C):
    T1, T2, B1, B2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
    p = 0.0; out = []

    def replace(e_in_b2, cs):
        nonlocal p
        if T1 and (len(T1) > p or (e_in_b2 and len(T1) == int(p))):
            src, ghost = T1, B1
        else:
            src, ghost = T2, B1 if not T2 else B2
            if not T2:
                src = T1
        v = next((x for x in src if x not in cs), None)
        if v is None:
            other = T2 if src is T1 else T1
            v = next(x for x in other if x not in cs)
            src = other; ghost = B2 if other is T2 else B1
        del src[v]; ghost[v] = 1

    for row in seq:
        cs = set(row); m = 0
        for e in row:
            if e in T1:
                del T1[e]; T2[e] = 1
            elif e in T2:
                T2.move_to_end(e)
            else:
                m += 1
                if e in B1:
                    p = min(C, p + max(len(B2) / max(len(B1), 1), 1))
                    del B1[e]
                    if len(T1) + len(T2) >= C:
                        replace(False, cs)
                    T2[e] = 1
                elif e in B2:
                    p = max(0.0, p - max(len(B1) / max(len(B2), 1), 1))
                    del B2[e]
                    if len(T1) + len(T2) >= C:
                        replace(True, cs)
                    T2[e] = 1
                else:
                    if len(T1) + len(B1) >= C:
                        if len(T1) < C and B1:
                            B1.popitem(last=False)
                            if len(T1) + len(T2) >= C:
                                replace(False, cs)
                        elif len(T1) >= C:
                            v = next(x for x in T1 if x not in cs)
                            del T1[v]
                    elif len(T1) + len(T2) + len(B1) + len(B2) >= C:
                        if len(T1) + len(T2) + len(B1) + len(B2) >= 2 * C and B2:
                            B2.popitem(last=False)
                        if len(T1) + len(T2) >= C:
                            replace(False, cs)
                    T1[e] = 1
        assert len(T1) + len(T2) <= C, "ARC capacity leak"
        out.append(m)
    return out


POLICIES = dict(lru=run_lru, opt=run_opt, lfuda=run_lfuda,
                tinylfu=run_tinylfu, arc=run_arc)


# ------------------------------------------------------------ cost model
def serial_cost(M):
    """M [T, L] misses per (token, layer) -> per-token ms (measured consts)."""
    io = np.where(M > 0, gs.IO_A_MS + gs.IO_B_MS * M, 0.0).sum(1)
    inst = M.sum(1) * gs.INSTALL_MS
    return gs.COMPUTE_MS + io + inst + L * gs.SYNC_MS, io.mean(), inst.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", default="92,143,192")
    ap.add_argument("--policies", default="lru,opt,lfuda,tinylfu,arc")
    ap.add_argument("--traces-dir", default=str(ROOT / "results/traces_synth"))
    ap.add_argument("--out", default=str(ROOT / "results/design_admission_cache.json"))
    a = ap.parse_args()
    tdir = Path(a.traces_dir)
    _, hl, hT = sp.load_split(tdir / "holdout.npz")
    gold = np.stack([hl[li] for li in range(L)], axis=1).astype(np.int16)
    sha = hashlib.sha256((tdir / "holdout.npz").read_bytes()).hexdigest()[:12]
    seqs = [[list(dict.fromkeys(gold[t, li].tolist())) for t in range(len(gold))]
            for li in range(L)]
    rows = []
    for C in [int(x) for x in a.caps.split(",")]:
        base = None
        for pol in a.policies.split(","):
            M = np.array([POLICIES[pol](seqs[li], C) for li in range(L)]).T
            M = M[gs.WARMUP:]
            tok_ms, io, inst = serial_cost(M)
            tps = 1000.0 / tok_ms.mean()
            miss = M.sum(1).mean()
            row = dict(cap=C, policy=pol, miss_per_tok=round(float(miss), 2),
                       hit_rate=round(1 - miss / (L * K), 4),
                       layer_steps_with_miss=round(float((M > 0).mean()), 3),
                       io_ms=round(float(io), 1), install_ms=round(float(inst), 1),
                       serial_tps_SIM=round(float(tps), 2))
            if pol == "lru":
                base = row
            if base:
                row["tps_vs_lru"] = round(tps / base["serial_tps_SIM"], 3)
                row["miss_vs_lru"] = round(miss / base["miss_per_tok"], 3)
            rows.append(row); print(json.dumps(row), flush=True)
    out = dict(kind="simulated", traces=str(tdir), holdout_sha256_12=sha,
               cost_model="serial-latency, measured M4 Max 36 GB constants "
                          "(gap_m4max_36gb.py); compute_ms assumed",
               rows=rows)
    Path(a.out).write_text(json.dumps(out, indent=1)); print("wrote", a.out)


if __name__ == "__main__":
    main()
