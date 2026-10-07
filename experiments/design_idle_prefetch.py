"""design=idle-window prefetch (+ sidecar-Belady), scored under the MEASURED
serial-latency model of oMLX 0.7.0 on Santa Cruz (gap_santa_cruz.py).

Observation (lead_silicon, measured): oMLX resolves expert misses serially per
layer: per token  compute + sum_layers_with_miss(A + B*k) + misses*install +
L*sync. During compute+sync (~29 ms/token, compute ASSUMED 23.2) the SSD
does nothing. Fill that idle window with background reads for token t+1.

Mechanism
  * After layer li of token t routes, a predictor proposes candidates for
    (t+1, li). Candidates are served round-robin over layers (rank 0 of every
    layer, then rank 1, ...) until the per-token background budget
        B = idle_ms * BW_bg / expert_MB      (BW_bg = 4.85 GB/s, measured k=4)
    is spent. Demand reads keep priority (assumed: background IO only runs
    inside the idle window; it never delays a demand miss).
  * Prefetched experts are inserted as MRU in the per-layer LRU cache (same
    hard capacity C; they evict the LRU victim -> pollution is simulated).
  * Precision gate (no tuning): each predictor keeps a running precision
    (hits / issued, EMA over ~1000 issues). Prefetch only while precision >=
    break-even, derived from the cost model:
      install on critical path (pessimistic):  install / (B + install) = 0.37
      install hidden on a helper thread (optimistic): 0.05 (pollution only)
Predictors
  none        plain oMLX LRU                                       baseline
  decay_freq  per-layer decayed use counts (half-life C/K tokens), top
              non-resident: online, legal
  coact       build-split co-activation graph M[e_t -> f_{t+1}], top
              non-resident given token t's experts: online, legal
  sidecar     exact routing of t+1 from a replayed sidecar (prefix-replay
              regime only): upper bound for prefetch
  sidecar_opt sidecar + Belady eviction (next use known from the replay):
              the full replay-regime mechanism
All tok/s are SIMULATED (synthetic calibrated traces, measured IO constants,
assumed compute). Two install assumptions are always reported.

usage: python3 experiments/design_idle_prefetch.py
"""
import argparse, hashlib, json, sys
from collections import OrderedDict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import sim_paging as sp  # noqa: E402
import gap_santa_cruz as gs  # noqa: E402

L, K, E = sp.L, sp.K, sp.E
BW_BG_GBPS = 4.85            # measured k=4 step rate (microbench)
IDLE_MS = gs.COMPUTE_MS + L * gs.SYNC_MS
EXPERT_MB = gs.EXPERT_MB
BUDGET = int(IDLE_MS * BW_BG_GBPS / EXPERT_MB)     # experts/token
BREAKEVEN = dict(pess=gs.INSTALL_MS / (gs.IO_B_MS + gs.INSTALL_MS), opt=0.05)


class Gate:
    def __init__(self, thr, alpha=1e-3):
        self.p, self.thr, self.a = 1.0, thr, alpha   # optimistic start

    def ok(self):
        return self.p >= self.thr

    def update(self, hit):
        self.p += self.a * ((1.0 if hit else 0.0) - self.p)


def next_use_table(gold):
    """nu[t][li] = dict e -> next token index > t where e is used (or INF)."""
    T = len(gold); INF = 1 << 30
    last = np.full((L, E), INF, np.int64)
    nu = [None] * T
    for t in range(T - 1, -1, -1):
        nu[t] = [{int(e): int(last[li, e]) for e in gold[t, li]} for li in range(L)]
        for li in range(L):
            last[li, gold[t, li]] = t
    return nu


def run(gold, C, predictor, install, graphs=None, nu=None):
    T = len(gold)
    INF = 1 << 30
    caches = [OrderedDict() for _ in range(L)]
    nxt = [dict() for _ in range(L)]          # Belady: resident -> next use
    belady = predictor == "sidecar_opt"
    sidecar = predictor in ("sidecar", "sidecar_opt")
    decay = 0.5 ** (1.0 / max(1.0, C / K))
    cnt = np.zeros((L, E), np.float32)
    gate = Gate(BREAKEVEN[install])
    pending = [set() for _ in range(L)]       # prefetched for t, not yet checked
    M = np.zeros((T, L), np.int32)
    issued = useful = gated_off = 0

    def insert(li, e, protect):
        c = caches[li]
        c[e] = 1; c.move_to_end(e)
        while len(c) > C:
            if belady:
                v = max((x for x in c if x not in protect),
                        key=lambda x: nxt[li].get(x, INF))
                nxt[li].pop(v, None)
            else:
                v = next(x for x in c if x not in protect)
            del c[v]

    for t in range(T):
        for li in range(L):
            cur = set(int(x) for x in gold[t, li])
            m = 0
            for e in cur:
                if e in caches[li]:
                    caches[li].move_to_end(e)
                else:
                    m += 1
                    insert(li, e, cur)
                if belady:
                    nxt[li][e] = nu[t][li][e]
            M[t, li] = m
            for e in pending[li]:
                h = e in cur
                useful += h
                gate.update(h)
            pending[li] = set()
            cnt[li] *= decay
            cnt[li, list(cur)] += 1.0
        if predictor == "none" or t + 1 >= T:
            continue
        cands = []
        for li in range(L):
            c = caches[li]
            if sidecar:
                lst = [int(e) for e in gold[t + 1, li] if int(e) not in c]
            else:
                s = cnt[li] if predictor == "decay_freq" else \
                    graphs[li][gold[t, li].astype(np.int64)].sum(0)
                order = np.argsort(-s)
                lst = [int(e) for e in order[:C + 3 * K] if int(e) not in c][:3]
            cands.append(lst)
        budget = BUDGET
        rank = 0
        while budget > 0 and any(rank < len(x) for x in cands):
            for li in range(L):
                if budget <= 0:
                    break
                if rank >= len(cands[li]):
                    continue
                if not sidecar and not gate.ok():
                    gated_off += 1
                    continue
                e = cands[li][rank]
                protect = set(int(x) for x in gold[t + 1, li]) if sidecar else pending[li]
                if belady:
                    nxt[li][e] = t + 1
                insert(li, e, protect | {e})
                pending[li].add(e)
                issued += 1; budget -= 1
            rank += 1
        for li in range(L):
            assert len(caches[li]) <= C
    M = M[gs.WARMUP:]
    io = np.where(M > 0, gs.IO_A_MS + gs.IO_B_MS * M, 0.0).sum(1)
    inst = M.sum(1) * gs.INSTALL_MS
    pf_inst = issued / T * gs.INSTALL_MS if install == "pess" else 0.0
    tok_ms = gs.COMPUTE_MS + io + inst + L * gs.SYNC_MS + pf_inst
    return dict(miss_per_tok=round(float(M.sum(1).mean()), 2),
                layer_steps_with_miss=round(float((M > 0).mean()), 3),
                prefetch_per_tok=round(issued / T, 2),
                prefetch_precision=round(useful / max(issued, 1), 3),
                gated_off_per_tok=round(gated_off / T, 1),
                io_ms=round(float(io.mean()), 1),
                install_ms=round(float(inst.mean() + pf_inst), 1),
                serial_tps_SIM=round(float(1000.0 / tok_ms.mean()), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", default="92,143,192")
    ap.add_argument("--predictors", default="none,decay_freq,coact,sidecar,sidecar_opt")
    ap.add_argument("--traces-dir", default=str(ROOT / "results/traces_synth"))
    ap.add_argument("--out", default=str(ROOT / "results/design_idle_prefetch.json"))
    a = ap.parse_args()
    tdir = Path(a.traces_dir)
    _, bl, bT = sp.load_split(tdir / "build.npz")
    _, hl, _ = sp.load_split(tdir / "holdout.npz")
    gold = np.stack([hl[li] for li in range(L)], axis=1).astype(np.int16)
    preds = a.predictors.split(",")
    graphs = None
    if "coact" in preds:
        graphs = []
        for li in range(L):
            G = np.zeros((E, E), np.float32)
            x = bl[li][:bT - 1].astype(np.int64); y = bl[li][1:bT].astype(np.int64)
            np.add.at(G, (np.repeat(x, K, axis=1).ravel(), np.tile(y, (1, K)).ravel()), 1.0)
            graphs.append(G)
    nu = next_use_table(gold) if "sidecar_opt" in preds else None
    sha = hashlib.sha256((tdir / "holdout.npz").read_bytes()).hexdigest()[:12]
    rows = []
    for C in [int(x) for x in a.caps.split(",")]:
        base = {}
        for pr in preds:
            for inst in ("pess", "opt"):
                if pr == "none" and inst == "opt":
                    continue
                r = dict(cap=C, predictor=pr, install=inst,
                         **run(gold, C, pr, inst, graphs, nu))
                if pr == "none":
                    base = r
                r["tps_vs_lru"] = round(r["serial_tps_SIM"] / base["serial_tps_SIM"], 3)
                rows.append(r); print(json.dumps(r), flush=True)
    out = dict(kind="simulated", traces=str(tdir), holdout_sha256_12=sha,
               budget_experts_per_token=BUDGET, idle_ms=round(IDLE_MS, 1),
               breakeven=BREAKEVEN,
               assumptions=["compute_ms 23.2 assumed (BW-scaled 128GB)",
                            "background IO confined to compute+sync window, never delays demand reads",
                            "GPU compute not slowed by concurrent SSD DMA",
                            "synthetic traces: no cross-layer / real token structure"],
               rows=rows)
    Path(a.out).write_text(json.dumps(out, indent=1)); print("wrote", a.out)


if __name__ == "__main__":
    main()
