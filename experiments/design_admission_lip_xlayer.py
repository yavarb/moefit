#!/usr/bin/env python3
"""T3: LIP admission on REAL decode routes (composition with T1's guarded x-layer fetch).

Reuses load/preview/constants from T1's experiments/design_xlayer_guarded.py
(commit ba024ae) WITHOUT editing it; the simulate body is replicated here with
one change: demand-miss installs honor LIP (Qureshi ISCA'07) — a
FIRST-LIFETIME miss installs byte-identically but is inserted at the LRU
position (next eviction victim) instead of MRU, when the cache is full.
Warm/free-slot phase inserts stock. Hits and recurrent misses insert MRU.

Question this answers: does LIP compose with T1's guarded cross-layer spec
installs (which already cold-end insert), or does it double-penalize?
SIM timing; REAL routes (8 prompts x 161 tok, MBP capture). No silicon claim.
"""
import json, sys
from collections import OrderedDict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import design_xlayer_guarded as G  # T1's module: load, preview, constants

A_MS, B_MS, INSTALL_MS, SYNC_MS = G.A_MS, G.B_MS, G.INSTALL_MS, G.SYNC_MS
BG_MS, CONTEND_MS, TAU, EXPERT_MB = G.BG_MS, G.CONTEND_MS, G.TAU, G.EXPERT_MB


def simulate_lip(P, I, cap, compute_ms, max_pf=10, mode="baseline", warm=160,
                 d=1, issue="after_demand", oracle=False, tau=None,
                 staged=False, lip=False):
    if oracle:
        mode = "oracle"
    tau = TAU if tau is None else tau
    nL = I[0].shape[1]
    slot = compute_ms / nL
    caches = [OrderedDict() for _ in range(nL)]
    prev_need = [set() for _ in range(nL)]
    miss_hist = [dict() for _ in range(nL)]  # T3: lifetime demand-miss counts
    st = dict(n=0, miss=0, io=0.0, inst=0.0, contend=0.0, spec=0, spec_hit=0,
              spec_dropped_window=0, lip_inserts=0)
    g = 0
    for P, I in zip(P, I):
        for t in range(I.shape[0]):
            score = g >= warm
            g += 1
            spec_set = [set() for _ in range(nL)]
            clock = lane = 0.0
            ilane = 0.0
            tk = dict(miss=0, io=0.0, inst=0.0, contend=0.0)
            queued = {}
            for L in range(nL):
                c = caches[L]
                for e, done in queued.pop(L, []):
                    if done <= clock and e not in c:
                        if len(c) >= cap:
                            v = next((v for v in c if v not in prev_need[L]
                                      and v not in spec_set[L]), None)
                            if v is None:
                                continue
                            del c[v]
                        c[e] = 1
                        c.move_to_end(e, last=False)
                        spec_set[L].add(e)
                    elif score:
                        st["spec_dropped_window"] += 1
                issue_clock = clock
                need = [int(e) for e in I[t, L]]
                miss = []
                for e in need:
                    if e in c:
                        c.move_to_end(e)
                        if e in spec_set[L] and score:
                            st["spec_hit"] += 1
                    else:
                        miss.append(e)
                k = len(miss)
                io = (A_MS + B_MS * k) if k else 0.0
                inst = INSTALL_MS * k
                # ---- T3 LIP: demand-miss install position ----
                for e in miss:
                    mh = miss_hist[L]
                    mh[e] = mh.get(e, 0) + 1
                    was_full = len(c) >= cap
                    c[e] = 1
                    while len(c) > cap:
                        c.popitem(last=False)
                    if (lip and was_full and mh[e] == 1 and score):
                        # first-lifetime miss in steady state: LRU position
                        c.move_to_end(e, last=False)
                        st["lip_inserts"] += 1
                clock += io + inst + SYNC_MS
                tk["miss"] += k; tk["io"] += io; tk["inst"] += inst
                tgt = L + d
                if issue == "at_xL" and mode not in ("baseline", "unguarded") and tgt < nL:
                    ct = caches[tgt]
                    if mode == "oracle":
                        cand = [int(e) for e in I[t, tgt]]
                    else:
                        pv = P[t, tgt]
                        order = np.argsort(-pv)[:40]
                        cand = [int(e) for e in order if pv[e] >= tau]
                    cand = [e for e in cand if e not in ct and
                            all(e != q for q, _ in queued.get(tgt, []))][:max_pf]
                    lane = max(lane, issue_clock)
                    for e in cand:
                        if staged:
                            lane += BG_MS
                            ilane = max(ilane, lane) + INSTALL_MS
                            done = ilane
                        else:
                            lane += BG_MS + INSTALL_MS
                            done = lane
                        queued.setdefault(tgt, []).append((e, done))
                        tk["contend"] += CONTEND_MS
                        if score:
                            st["spec"] += 1
                clock += slot
                prev_need[L] = set(need)
            if score:
                st["n"] += 1
                for k2 in ("miss", "io", "inst", "contend"):
                    st[k2] += tk[k2]
    n = st["n"]
    tok_ms = compute_ms + nL * SYNC_MS + (st["io"] + st["inst"] + st["contend"]) / n
    return dict(mode=mode, lip=lip, tokens_scored=n,
                miss_per_tok=round(st["miss"] / n, 2),
                io_ms=round(st["io"] / n, 2), install_ms=round(st["inst"] / n, 2),
                contend_ms=round(st["contend"] / n, 2),
                tok_ms=round(tok_ms, 2), tps=round(1000 / tok_ms, 2),
                lip_inserts_per_tok=round(st["lip_inserts"] / max(n, 1), 2),
                spec_per_tok=round(st["spec"] / n, 2),
                spec_hits_per_tok=round(st["spec_hit"] / n, 2),
                spec_precision=round(st["spec_hit"] / max(st["spec"], 1), 3))


def main():
    npz = sys.argv[1]
    out = sys.argv[2]
    W, tags, X, I = G.load(npz)
    P1 = [G.preview(W, x, 1) for x in X]
    P2 = [G.preview(W, x, 2) for x in X]
    res = dict(kind="T3 LIP admission on REAL decode routes (SIM timing; xlayer capture)",
               source="experiments/design_xlayer_guarded.py (ba024ae) loader+constants",
               prompts=tags, decode_tokens=int(sum(x.shape[0] for x in X)), cap=143,
               constants=dict(A=A_MS, B=B_MS, install=INSTALL_MS, sync=SYNC_MS,
                              bg_ms=round(BG_MS, 3), contend_ms=CONTEND_MS, tau=TAU))
    rows = []
    designs = [
        ("baseline true-LRU", P1, dict(mode="baseline")),
        ("baseline + LIP", P1, dict(mode="baseline", lip=True)),
        ("guarded d=2 tau.01 at_xL staged (T1 best)", P2,
         dict(mode="guarded", issue="at_xL", d=2, tau=0.01, staged=True)),
        ("guarded d=2 tau.01 at_xL staged + LIP", P2,
         dict(mode="guarded", issue="at_xL", d=2, tau=0.01, staged=True, lip=True)),
        ("ORACLE d=2 staged (bound)", P2,
         dict(issue="at_xL", d=2, oracle=True, staged=True)),
        ("ORACLE d=2 staged + LIP (bound)", P2,
         dict(issue="at_xL", d=2, oracle=True, staged=True, lip=True)),
    ]
    for comp in (24.1, 18.1):
        for name, P, kw in designs:
            r = simulate_lip(P, I, 143, comp, **kw)
            r["design"] = name
            r["compute_ms"] = comp
            rows.append(r)
            print(json.dumps(r), flush=True)
    res["rows"] = rows
    Path(out).write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
