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

Coupled steady state per token (default --time-model serial):
  compute_ms = (floor + 701 MiB + async) / (DRAM_BW * DRAM_EFF)
  stream_ms  = sum_layers(A + B*k | k>0) + install*misses + L*sync
               (A=0.20, B=0.52, install=0.30, sync=0.122 ms; Santa Cruz
               microbench). tok/s = 1000 / (compute_ms + stream_ms).
  Legacy --time-model bandwidth keeps max(compute, bytes/SSD_BW).

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
# module-level default so simulate(mode="prior") does not NameError when
# no per-layer prior/persist mix is requested (set by callers that want one)
pin_counts = None
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
    # float64 throughout: float32 matmul hit a BLAS edge case on this
    # box (spurious "divide by zero/overflow in matmul" RuntimeWarnings
    # on well-scaled data, Xb absmax ~5; T8 saw actual NaNs on their
    # env). G is feat_dim^2 and Xb.T@Yb is feat_dim x 512 - trivial in
    # double precision.
    out = np.zeros((len(hs), L, budget), np.int16)
    mu, sd = bf.mean(0), bf.std(0) + 1e-6
    Xb = ((bf - mu) / sd).astype(np.float64)
    Xh = ((hf[hs] - mu) / sd).astype(np.float64)
    G = Xb.T @ Xb + lam * np.eye(Xb.shape[1], dtype=np.float64)
    for li in range(L):
        Yb = np.zeros((len(bf), E), np.float64)
        Yb[np.arange(len(bf))[:, None], bl[li][:len(bf)]] = 1.0
        W = np.linalg.solve(G, Xb.T @ Yb)
        s = Xh @ W
        if not np.isfinite(s).all():
            raise FloatingPointError(
                f"probe_picks: non-finite scores on layer {li}; "
                f"|Xb|max={np.abs(Xb).max()}, |G|max={np.abs(G).max()}")
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
             audit=True, avail=None, pin_counts=None,
             hit_refresh=False, want_misses=False):
    """One pass with hard capacity: res[li] never exceeds caps[li] experts.
    pin[li] = predicted experts pinned for token t+1 (free SSD slots up to
    async_allow). Non-pinned residents evicted LRU. sync = expert read at
    hit time (stalls); async = read at prefetch time (hidden by design,
    charged to SSD).

    hit_refresh=False (default, SHIPPED semantics): a hit does not refresh
    the resident's LRU timestamp, so eviction order is insertion order.
    hit_refresh=True: true LRU. (Attribution note 2026-10-07: oMLX 0.7.0
    ExpertCache is NOT LRU — source-read by design_inventor
    (omlx-ref @79f4488) shows decayed-routing-count eviction (+1 per
    route, x0.7 every 4 calls, current call protected; exact replay in
    experiments/design_omlx_exact.py). On synth traces decay-count is
    numerically ~true-LRU (misses 56.9 vs 58.3/tok @cap143; serial tps
    12.15 vs 12.08), so serial-model anchors computed under true-LRU
    stand; use design_omlx_exact.py for policy-exact numbers.

    want_misses=True appends M to the return: (T, L) int16 per-token
    per-layer SYNC miss counts (needed by the serial-latency time model).

    mode="sidecar_probe" (hybrid): avail is a per-token bool array; when
    avail[t+1] is True the routing sidecar has the answer and gold is
    pinned exactly (like sidecar); otherwise the PLE probe picks fill the
    gap (like probe with `pick` slots). Models a prefix-cache sidecar that
    only covers part of the traffic.

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
        if pin_counts is not None:                 # per-layer prior/persist mix
            dyn_cap = [max(c - int(p), 4) for c, p in zip(caps, pin_counts)]

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
    M = np.zeros((T, L), dtype=np.int16) if want_misses else None
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
                    if hit_refresh and e not in pin[li]:
                        set_ts(li, e, t)
                else:
                    touch_dyn(li, e, t)
                    sync += 1
                    if M is not None:
                        M[t, li] += 1
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
    out = (served / n, sync * EXPERT_MIB / T, async_ * EXPERT_MIB / T)
    return out + (M,) if want_misses else out


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


# ---- serial-latency time model (T3, 2026-10-07) -------------------------
# MEASURED on Santa Cruz (M4 Max 36 GB, oMLX 0.7.0) by lead_silicon,
# results/microbench_expert_reads_santa_cruz.json (commit 463c856):
#   per-layer-step k-miss cold reads (9 preads/expert, 12 threads):
#     k=1..4 -> 0.72/1.26/1.78/2.28 ms  ~= IO_A + IO_B * k
#   host->slot install 0.27-0.35 ms/expert; tiny per-layer device sync 0.12 ms
# The bandwidth model above (_times) assumes misses stream at spec SSD
# bandwidth and overlap compute; the measured runtime resolves misses
# SERIALLY per layer (sync -> miss reads -> install -> compute, nothing
# overlaps across layers). Validated out of sample at 3 measured points
# (results/fidelity_serial_validate.json): steady cap143 12.1 vs measured
# 12.7/13.0 tok/s, cap92 8.9 vs 7.8 (crowded), cold 16-token transient
# 5.0 vs 6.43. Constants are Santa-Cruz-specific: applying this model to
# other tiers is EXTRAPOLATION until their constants are measured.
SERIAL_IO_A_MS = 0.20
SERIAL_IO_B_MS = 0.52
SERIAL_INSTALL_MS = 0.30
SERIAL_SYNC_MS = 0.122
# 36 GB M4 Max is the 410 GB/s DRAM bin (Santa Cruz). usable=0.75*36-3.
TIERS["36GB-M4M36"] = dict(dram=410.0, ssd=7.4, usable=24.0)


def serial_ms(M, compute_ms):
    """Per-token serial-latency ms given (T, L) per-token per-layer
    SYNC miss counts M (from simulate(..., want_misses=True)).
    tok_ms = compute + sum_layers(A + B*k | k>0) + install*misses + L*sync
    """
    k = np.asarray(M, dtype=np.float64)
    io = np.where(k > 0, SERIAL_IO_A_MS + SERIAL_IO_B_MS * k, 0.0).sum(axis=1)
    return (compute_ms + io + SERIAL_INSTALL_MS * k.sum(axis=1)
            + L * SERIAL_SYNC_MS)


def solve_policy_serial(gold, prior_rank, cap, mode, spec,
                        hit_refresh=True, warmup=200,
                        picks=None, pick=0, async_allow=0):
    """Serial-latency solve (default shipped time model).

    hit_refresh=True by default: true LRU. NOTE: oMLX 0.7.0 ExpertCache
    is actually decayed-count eviction (see simulate() attribution note
    and experiments/design_omlx_exact.py for the exact replay); on synth
    traces the two are within 0.6% in serial tps. The old bandwidth-model
    rows were FIFO-ish (no refresh).

    Prefetch modes (probe/sidecar): remaining SYNC misses after prefetch
    are charged with the same A+B*k + install + sync serial formula;
    successful async prefetch simply reduces M. Async bytes are not
    double-charged (they replaced a sync miss).

    compute_ms uses the bandwidth model's DRAM term at the tier's dram
    spec (preserves the 128 GB DRAM_EFF / 57.4 calibration when M=0 and
    only PLE/sync residual remains — full-fit still ~compute-bound).
    Returns dict(tps, compute_ms, stream_ms, io_ms, install_ms, sync_ms,
    misses_per_tok, served, sync_mb, async_mb).
    """
    T = len(gold)
    if picks is None:
        picks = np.zeros((T, L, max(pick, 1)), np.int16)
    if mode in ("lru", "prior"):
        async_allow = 0
        pick = 0
    srv, sync_mb, async_mb, M = simulate(
        gold, picks, prior_rank, cap, mode, pick, async_allow,
        want_misses=True, hit_refresh=hit_refresh)
    assert M is not None
    compute_ms = (READ_FLOOR_MIB + GOLD_MIB + async_mb) / 1024.0 \
        / (spec["dram"] * DRAM_EFF) * 1000.0
    Mw = M[warmup:]
    ms = serial_ms(Mw, compute_ms)
    n = max(T - warmup, 1)
    io = float(np.where(Mw > 0,
                        SERIAL_IO_A_MS + SERIAL_IO_B_MS * Mw,
                        0.0).sum() / n)
    stream_ms = float(ms.mean() - compute_ms)
    return dict(tps=round(1000.0 / float(ms.mean()), 1),
                compute_ms=round(compute_ms, 1),
                stream_ms=round(stream_ms, 1),
                io_ms=round(io, 1),
                install_ms=round(SERIAL_INSTALL_MS
                                 * float(Mw.sum()) / n, 1),
                sync_ms=round(L * SERIAL_SYNC_MS, 1),
                misses_per_tok=round(float(Mw.sum()) / n, 1),
                served=round(srv, 3),
                sync_mb=round(sync_mb, 1),
                async_mb=round(async_mb, 1))


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
    ap.add_argument("--time-model", choices=("serial", "bandwidth"),
                    default="serial",
                    help="serial (default): Santa Cruz microbench A+B*k + "
                         "install + sync, no compute/stream overlap. "
                         "bandwidth: legacy max(compute, bytes/ssd) model.")
    ap.add_argument("--hit-refresh", action="store_true", default=True,
                    help="true-LRU hit refresh (default on for serial)")
    ap.add_argument("--no-hit-refresh", action="store_false",
                    dest="hit_refresh")
    ap.add_argument("--out", default=None,
                    help="output JSON; defaults to results/sim_paging.json "
                         "only for the shipped configuration")
    a = ap.parse_args()

    default_cfg = (a.traces_dir == str(ROOT / "results/traces")
                   and a.throttle == "legacy"
                   and a.time_model == "serial")
    out_path = Path(a.out) if a.out else (
        ROOT / "results/sim_paging.json" if default_cfg
        else ROOT / f"results/sim_paging_{a.time_model}_{a.throttle}.json")

    tdir = Path(a.traces_dir)
    bf, bl, bT = load_split(tdir / "build.npz")
    hf, hl, hT = load_split(tdir / "holdout.npz")
    rng = np.random.default_rng(0)
    hs = np.sort(rng.choice(hT, size=min(a.eval_sub, hT), replace=False))
    gold = np.stack([hl[li][hs] for li in range(L)], axis=1).astype(np.int16)
    print(f"build rows={bT} holdout rows={hT} evaluated={len(hs)} "
          f"time_model={a.time_model}", flush=True)

    prior_rank = {li: np.argsort(-np.bincount(bl[li].ravel(), minlength=E))
                  for li in range(L)}
    modes = a.modes.split(",")
    picks_map = {P: probe_picks(bf.astype(np.float64), bl, hf, hs, P)
                 for P in [int(x) for x in a.probe_picks.split(",")]} \
        if "probe" in modes else {0: None}

    out = []
    for cap in [int(x) for x in a.caps.split(",")]:
        for P in sorted(picks_map):
            for mode in modes:
                p = P if mode == "probe" else 0
                row = dict(cap=cap, mode=mode, probe_picks=P,
                           throttle=a.throttle, time_model=a.time_model)
                for tier, spec in TIERS.items():
                    need_gib = FLOOR_MIB / 1024 + cap * L * EXPERT_MIB / 1024
                    if need_gib > spec["usable"]:
                        continue
                    if a.time_model == "serial":
                        r = solve_policy_serial(
                            gold, prior_rank, cap, mode, spec,
                            hit_refresh=a.hit_refresh,
                            picks=picks_map[P], pick=p,
                            async_allow=10 ** 9)
                        row[f"served_{tier}"] = r["served"]
                        row[f"tps_{tier}"] = r["tps"]
                        row[f"ct_{tier}"] = r["compute_ms"]
                        row[f"st_{tier}"] = r["stream_ms"]
                        row[f"ssdMB_{tier}"] = round(
                            r["sync_mb"] + r["async_mb"], 0)
                    else:
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
