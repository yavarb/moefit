"""Guarded cross-layer expert fetch pipeline - evaluated on REAL decode routes.

Input: capture_xlayer.py npz: real per-layer MoE inputs x_L, real top-10
routes, router weights W_L, from greedy decode of the real checkpoint (MBP).

MECHANISM (what an oMLX ExpertCache hook would do, per decode token)
  When layer L's MoE input x_L is available (just before L's own demand reads),
  compute a router PREVIEW for layer L+1:  p = softmax(W_{L+1} @ x_L)  (one
  2560x512 GEMV on weights that are already resident). Candidates = experts
  not resident at L+1, in descending p. Guards:
    G1 confidence: only p_e >= TAU is eligible (TAU fixed a priori = 0.02,
       i.e. ~10x the uniform 1/512; not swept)
    G2 never-fight-demand: speculative reads are issued only AFTER layer L's
       own demand reads complete, on a background lane, and only those that
       finish (read + install) before layer L+1's demand would start (end of
       L's compute slot). Demand never waits on a speculative read.
    G3 cold insertion + hot protection: a speculative install is placed at the
       eviction end of L+1's LRU and may only evict an expert not routed at
       L+1 in the previous token; a wrong guess is the next victim.
  Installs of speculative experts run on the background lane (staged install,
  T4-style), so they are charged to the lane, not to the critical path.

COST MODEL (measured Santa Cruz constants)
  demand: per layer with k misses  A + B*k  read + INSTALL*k  (critical path)
  sync 0.122 ms/layer; compute C ms/token spread evenly over 48 layers.
  background lane: BG_MS read + INSTALL per speculative expert (k=4 random
  chunk 4.85 GB/s) and, per background expert, a CONTEND_MS critical-path
  penalty (design_inventor measured 0.08-0.2 ms/bg expert on Santa Cruz).
Cache: per-layer true LRU with hit refresh = Santa Cruz oMLX 0.7.0 ExpertCache
(source-verified by design_inventor 23:55). Cache is retained across prompts
(concatenated, server-like); first WARM tokens of the whole stream unscored.

SIM on REAL routes; no silicon tok/s is claimed.
"""
import argparse, json
from collections import OrderedDict
from pathlib import Path
import numpy as np

A_MS, B_MS, INSTALL_MS, SYNC_MS = 0.20, 0.52, 0.30, 0.122
EXPERT_MB = 2.765
BG_MS = EXPERT_MB / 4.85
CONTEND_MS = 0.15
TAU = 0.02


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def load(path):
    d = np.load(path)
    W = d["W"].astype(np.float32)
    tags = sorted({k.split("|")[0] for k in d.files if "|" in k})
    X = [d[f"{t}|x"].astype(np.float32) for t in tags]
    I = [d[f"{t}|idx"].astype(np.int32) for t in tags]
    return W, tags, X, I


def preview(W, x, d):
    """P[t, L] = softmax(W_L @ x_{L-d}); rows L<d are zero (no preview)."""
    T, nL, _ = x.shape
    P = np.zeros((T, nL, W.shape[1]), np.float32)
    for L in range(d, nL):
        P[:, L] = softmax(x[:, L - d] @ W[L].T)
    return P


def prev_token_route(I):
    """Baseline predictor: layer L at token t predicted by layer L's routes at t-1."""
    P = np.zeros(I.shape[:2] + (512,), np.float32)
    for t in range(1, I.shape[0]):
        for L in range(I.shape[1]):
            P[t, L, I[t - 1, L]] = 0.1
    return P


def quality(Ps, Is, d):
    out = {}
    for m in (10, 20):
        hit = tot = 0
        for P, I in zip(Ps, Is):
            top = np.argpartition(-P[:, d:], m - 1, axis=-1)[..., :m]
            real = I[:, d:]
            for t in range(I.shape[0]):
                for l in range(I.shape[1] - d):
                    hit += len(set(top[t, l].tolist()) & set(real[t, l].tolist()))
            tot += I.shape[0] * (I.shape[1] - d) * I.shape[2]
        out[f"recall@{m}"] = round(hit / tot, 4)
    sel = cor = n = 0
    for P, I in zip(Ps, Is):
        for t in range(I.shape[0]):
            for l in range(d, I.shape[1]):
                s = np.nonzero(P[t, l] >= TAU)[0]
                sel += len(s); n += 1
                cor += len(set(s.tolist()) & set(I[t, l].tolist()))
    out[f"tau{TAU}_picks_per_layer"] = round(sel / n, 2)
    out[f"tau{TAU}_precision"] = round(cor / max(sel, 1), 4)
    out[f"tau{TAU}_recall"] = round(cor / (n * 10), 4)
    return out


def simulate(Ps, Is, cap, compute_ms, max_pf=10, mode="guarded", warm=160, d=1,
             issue="after_demand", oracle=False, tau=None, staged=False):
    """mode: baseline | guarded | unguarded | oracle (preview = real next routes).
    issue='after_demand': spec reads start after L's demand IO+install (strict G2)
    issue='at_xL': spec reads start when x_L exists, concurrently with L's demand
        reads (lane = separate queue; each spec read pays measured CONTEND_MS on
        the critical path); still never delays demand directly.
    d: preview target layer L+d from x_L (deadline = start of L+d's demand)."""
    if oracle:
        mode = "oracle"
    tau = TAU if tau is None else tau
    ilane = 0.0
    nL = Is[0].shape[1]
    slot = compute_ms / nL
    caches = [OrderedDict() for _ in range(nL)]
    prev_need = [set() for _ in range(nL)]
    st = dict(n=0, miss=0, io=0.0, inst=0.0, contend=0.0, spec=0, spec_hit=0,
              spec_dropped_window=0)
    g = 0
    for P, I in zip(Ps, Is):
        for t in range(I.shape[0]):
            score = g >= warm
            g += 1
            spec_set = [set() for _ in range(nL)]
            clock = lane = 0.0
            ilane = 0.0
            tk = dict(miss=0, io=0.0, inst=0.0, contend=0.0)
            queued = {}                                     # tgt -> list of (expert, done_time)
            for L in range(nL):
                c = caches[L]
                for e, done in queued.pop(L, []):           # land arrivals that made it
                    if done <= clock and e not in c:
                        if len(c) >= cap:
                            v = next((v for v in c if v not in prev_need[L] and v not in spec_set[L]), None)
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
                for e in miss:
                    c[e] = 1
                    while len(c) > cap:
                        c.popitem(last=False)
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
                        if staged:                 # read lane + separate install stream
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
                elif mode != "baseline" and tgt < nL and d == 1:
                    ct = caches[tgt]
                    if mode == "oracle":
                        cand = [int(e) for e in I[t, tgt]]
                    else:
                        pv = P[t, tgt]
                        order = np.argsort(-pv)[:40]
                        cand = [int(e) for e in order
                                if mode == "unguarded" or pv[e] >= TAU]
                    cand = [e for e in cand if e not in ct][:max_pf]
                    lane = max(lane, clock)                # G2: after own demand
                    deadline = clock + slot                # L+1 demand start
                    for e in cand:
                        done = lane + BG_MS + INSTALL_MS
                        if mode != "unguarded" and done > deadline:
                            if score:
                                st["spec_dropped_window"] += len(cand) - cand.index(e)
                            break
                        if len(ct) >= cap:
                            if mode == "unguarded":
                                ct.popitem(last=False)
                            else:                          # G3
                                v = next((v for v in ct if v not in prev_need[tgt] and v not in spec_set[tgt]), None)
                                if v is None:
                                    break
                                del ct[v]
                        ct[e] = 1
                        if mode != "unguarded":
                            ct.move_to_end(e, last=False)
                        lane = done
                        spec_set[tgt].add(e)
                        tk["contend"] += CONTEND_MS
                        if score:
                            st["spec"] += 1
                    if mode == "unguarded":               # unguarded: lane overrun stalls L+1
                        clock = max(clock, lane - slot)
                clock += slot
                prev_need[L] = set(need)
            if score:
                st["n"] += 1
                for k2 in ("miss", "io", "inst", "contend"):
                    st[k2] += tk[k2]
                if mode == "unguarded":
                    st.setdefault("stall", 0.0)
                    st["stall"] += clock - (compute_ms + nL * SYNC_MS + tk["io"] + tk["inst"])
    n = st["n"]
    stall = st.get("stall", 0.0) / n
    tok_ms = compute_ms + nL * SYNC_MS + (st["io"] + st["inst"] + st["contend"]) / n + stall
    return dict(mode=mode, cap=cap, compute_ms=compute_ms, tokens_scored=n,
                miss_per_tok=round(st["miss"] / n, 2),
                io_ms=round(st["io"] / n, 2), install_ms=round(st["inst"] / n, 2),
                contend_ms=round(st["contend"] / n, 2), stall_ms=round(stall, 2),
                tok_ms=round(tok_ms, 2), tps=round(1000 / tok_ms, 2),
                spec_per_tok=round(st["spec"] / n, 2),
                spec_hits_per_tok=round(st["spec_hit"] / n, 2),
                spec_precision=round(st["spec_hit"] / max(st["spec"], 1), 3),
                spec_MB_per_tok=round(st["spec"] / n * EXPERT_MB, 1),
                wasted_MB_per_tok=round((st["spec"] - st["spec_hit"]) / n * EXPERT_MB, 1),
                spec_dropped_by_window_per_tok=round(st["spec_dropped_window"] / n, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--cap", type=int, default=143)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    W, tags, X, I = load(a.npz)
    P1 = [preview(W, x, 1) for x in X]
    P0 = [preview(W, x, 0) for x in X]
    res = dict(kind="SIM on REAL decode routes (capture_xlayer.py, MBP 128GB, greedy)",
               prompts=tags, decode_tokens=int(sum(x.shape[0] for x in X)), cap=a.cap,
               constants=dict(A=A_MS, B=B_MS, install=INSTALL_MS, sync=SYNC_MS,
                              bg_ms=round(BG_MS, 3), contend_ms=CONTEND_MS, tau=TAU))
    res["selfcheck_same_layer_preview"] = quality(P0, I, 0)
    res["xlayer_preview_d1"] = quality(P1, I, 1)
    res["prev_token_same_layer"] = quality([prev_token_route(i) for i in I], I, 0)
    print(json.dumps({k: res[k] for k in res if "preview" in k or "prev_token" in k}), flush=True)
    P2 = [preview(W, x, 2) for x in X]
    P3 = [preview(W, x, 3) for x in X]
    res["xlayer_preview_d2"] = quality(P2, I, 2)
    res["xlayer_preview_d3"] = quality(P3, I, 3)
    print(json.dumps({k: res[k] for k in ("xlayer_preview_d2", "xlayer_preview_d3")}), flush=True)
    rows = []
    # tau 0.02: fixed a priori (~10x uniform). tau 0.01: derived from cost break-even
    # (hit saves B+install=0.82 ms, spec costs CONTEND 0.15 ms -> need P(routed)>=0.18;
    # measured P(routed|p in [0.01,0.02))=0.56, [0.005,0.01)=0.11 -> cut at 0.01).
    designs = [
        ("baseline (true LRU, serial demand)", P1, dict(mode="baseline")),
        ("guarded d=1, issue after own demand (strict G2)", P1, dict(mode="guarded")),
        ("unguarded d=1 top10 (ablation)", P1, dict(mode="unguarded")),
        ("guarded d=1 tau.02 at x_L", P1, dict(issue="at_xL", d=1)),
        ("guarded d=2 tau.02 at x_L", P2, dict(issue="at_xL", d=2)),
        ("guarded d=1 tau.01(break-even) at x_L", P1, dict(issue="at_xL", d=1, tau=0.01)),
        ("guarded d=2 tau.01(break-even) at x_L", P2, dict(issue="at_xL", d=2, tau=0.01)),
        ("guarded d=2 tau.01 at x_L + staged install", P2, dict(issue="at_xL", d=2, tau=0.01, staged=True)),
        ("ORACLE routes d=2 at x_L (bound)", P2, dict(issue="at_xL", d=2, oracle=True)),
        ("ORACLE routes d=2 at x_L + staged (bound)", P2, dict(issue="at_xL", d=2, oracle=True, staged=True)),
    ]
    for comp in (24.1, 18.1):
        for name, P, kw in designs:
            r = simulate(P, I, a.cap, comp, **kw)
            r["design"] = name
            rows.append(r)
            print(json.dumps(r), flush=True)
    res["rows"] = rows
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
