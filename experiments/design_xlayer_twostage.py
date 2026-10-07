"""T1 v2: TWO-STAGE (refine-at-L+1) guarded cross-layer fetch, judged by FULL
LAYER COVERAGE on REAL decode routes (capture_xlayer.py npz).

Finding that motivates it (MBP, real ExpertCache, 290f06a): with oMLX's
DB-ON windowed pool, a layer's demand reads run in parallel, so a layer's
wait is ~one read latency if ANY miss is uncovered. A prefetch only pays when
it covers ALL of that layer's misses. v1 (d=2 preview only) fully covers 14%
of miss-bearing layer-steps.

v2: stage 1 at layer L issues gate_{L+2}(x_L) (lead ~2 layers); stage 2 at
layer L+1 issues gate_{L+2}(x_{L+1}) (higher recall, lead ~1 layer) for any
confident candidates not already in flight. Same guards (tau, max per stage,
separate pool, held-not-installed, cancel on wrong route).

Cost models (both SIM, constants from measured M4 Max 36 GB numbers):
  PAR (DB-ON, the default silicon path): layer stall if any uncovered miss =
      R1 + R_extra*(k_uncov-1); covered experts must have LANDED (lane model:
      4 bg threads, BG_MS per expert, layer time T_LAYER).
  per-token gain = sum of removed stalls - contention per spec read.
Metric that does not depend on the cost model: fraction of miss-bearing
layer-steps fully covered by landed prefetches.
"""
import argparse, json, sys
from collections import OrderedDict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import design_xlayer_guarded as g  # noqa: E402

BG_MS = g.BG_MS            # 0.570 ms per 2.765 MB expert at 4.85 GB/s
LANES = 4
T_LAYER = 63.6 / 48        # measured DB-ON ms/tok (15.72 tok/s) / 48 layers
R1, R_EXTRA = 0.72, 0.14   # SIM: parallel batch read; first + bandwidth-share extra
CONTEND = 0.15             # measured bg contention per expert (design_inventor probe)


def run(Ps1, Ps2, Is, cap, tau, max_pf, stages, warm=160, cover_guard=False, resid_gate=None):
    """stages: subset of {'d2','d1'}; Ps1/Ps2 previews (d=1 / d=2)."""
    caches = [OrderedDict() for _ in range(48)]
    st = dict(steps=0, full=0, part=0, none=0, misses=0, cov=0, spec=0, used=0,
              stall_base=0.0, stall=0.0, n=0)
    gtok = 0
    for P1, P2, I in zip(Ps1, Ps2, Is):
        for t in range(I.shape[0]):
            score = gtok >= warm
            gtok += 1
            inflight = {}            # (layer, e) -> land time (ms from token start)
            lane = [0.0] * LANES
            nspec = 0

            def issue(tgt, pv, now):
                nonlocal nspec
                c = caches[tgt]
                k = 0
                if cover_guard:
                    # G4 coverage guard (derived, not tuned): a layer's DB-ON stall is
                    # removed only if EVERY miss is covered. Expected uncovered misses
                    # E = sum over non-resident, not-picked experts of min(1, 10 p_e)
                    # (top-10 routing); P(full) ~ exp(-E). Issue the batch only if
                    # P(full) * R1 > (#new picks) * CONTEND.
                    order = np.argsort(-pv)
                    picks, E = [], 0.0
                    for e in order.tolist():
                        if e in c:
                            continue
                        if pv[e] >= tau and len(picks) < max_pf:
                            if (tgt, e) not in inflight:
                                picks.append(e)
                            continue
                        E += min(1.0, 10 * float(pv[e]))
                        if pv[e] < 1e-4:
                            break
                    if not picks or np.exp(-E) * R1 <= len(picks) * CONTEND:
                        return
                if resid_gate is not None:
                    # G5 residual-mass gate: preview mass on non-resident experts that
                    # would NOT be fetched. Low residual => batch likely completes the
                    # layer's miss set (the only case that removes a DB-ON stall).
                    nonres = np.ones(pv.shape[0], bool)
                    nonres[list(c.keys())] = False
                    picked = [e for e in np.argsort(-pv)[:4 * max_pf].tolist()
                              if pv[e] >= tau and nonres[e]][:max_pf]
                    resid = float(pv[nonres].sum() - pv[picked].sum())
                    if resid > resid_gate:
                        return
                for e in np.argsort(-pv)[:4 * max_pf].tolist():
                    if pv[e] < tau or k >= max_pf:
                        break
                    if e in c or (tgt, e) in inflight:
                        continue
                    i = int(np.argmin(lane))
                    lane[i] = max(lane[i], now) + BG_MS
                    inflight[(tgt, e)] = lane[i]
                    k += 1
                    nspec += 1
            clock = 0.0
            for L in range(48):
                # stage issues happen at layer L's sync (start of L's slot)
                if "d2" in stages and L + 2 < 48:
                    issue(L + 2, P2[t, L + 2], clock)
                if "d1" in stages and L + 1 < 48:
                    issue(L + 1, P1[t, L + 1], clock)
                c = caches[L]
                need = [int(e) for e in I[t, L]]
                miss = [e for e in need if e not in c]
                landed = [e for e in miss if inflight.get((L, e), 1e9) <= clock]
                unc = len(miss) - len(landed)
                base = (R1 + R_EXTRA * (len(miss) - 1)) if miss else 0.0
                stall = (R1 + R_EXTRA * (unc - 1)) if unc else 0.0
                if score and miss:
                    st["steps"] += 1
                    st["misses"] += len(miss)
                    st["cov"] += len(landed)
                    st["full"] += unc == 0
                    st["part"] += 0 < unc < len(miss)
                    st["none"] += unc == len(miss)
                if score:
                    st["stall_base"] += base
                    st["stall"] += stall
                    st["used"] += sum((L, e) in inflight for e in miss)
                for e in need:              # true LRU, held-not-installed spec
                    if e in c:
                        c.move_to_end(e)
                    else:
                        c[e] = 1
                        if len(c) > cap:
                            c.popitem(last=False)
                clock += T_LAYER + stall
            if score:
                st["n"] += 1
                st["spec"] += nspec
    n, s = st["n"], st["steps"]
    contend = st["spec"] / n * CONTEND
    saved = (st["stall_base"] - st["stall"]) / n
    base_ms = 63.6
    return dict(stages="+".join(sorted(stages)) or "none", cover_guard=cover_guard, resid_gate=resid_gate, tau=tau, max_per_stage=max_pf,
                tokens=n, miss_steps_per_tok=round(s / n, 2), misses_per_tok=round(st["misses"] / n, 2),
                full_cover=round(st["full"] / s, 4), partial=round(st["part"] / s, 4),
                none=round(st["none"] / s, 4), miss_coverage=round(st["cov"] / st["misses"], 4),
                spec_per_tok=round(st["spec"] / n, 2), spec_precision=round(st["used"] / max(st["spec"], 1), 3),
                stall_removed_ms=round(saved, 2), contention_ms=round(contend, 2),
                net_ms=round(saved - contend, 2),
                tps_on_measured_15_72=round(1000 / (base_ms - (saved - contend)), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    W, tags, X, I = g.load(a.npz)
    P1 = [g.preview(W, x, 1) for x in X]
    P2 = [g.preview(W, x, 2) for x in X]
    rows = []
    for stages, cg in (((), False), (("d2",), False), (("d1",), False), (("d2", "d1"), False),
                       (("d2",), True), (("d1",), True), (("d2", "d1"), True)):
        r = run(P1, P2, I, 143, 0.01, 10, set(stages), cover_guard=cg)
        rows.append(r)
        print(json.dumps(r), flush=True)
    res = dict(kind="SIM on REAL decode routes; parallel (DB-ON) per-layer stall model; "
                    "tps column is an offset from the MEASURED 15.72 DB-ON anchor, not a measurement",
               constants=dict(BG_MS=round(BG_MS, 3), lanes=LANES, T_LAYER=round(T_LAYER, 3), R1=R1,
                              R_EXTRA=R_EXTRA, contend=CONTEND, tau=0.01, cap=143),
               rows=rows)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
