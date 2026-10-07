"""design_spec_pagein: speculative page-in from the router's SOFT TOP-K
(decayed routing-history scores, the same signal oMLX's ExpertCache evicts
by), with cancel-on-wrong-route. Scored under the MEASURED serial-latency
model.

Finding that forced the design (kept for honesty): speculating from the
CURRENT token's own top-10 finds nothing to fetch — demand resolution
makes every current pick resident, so next-token misses always come from
experts outside t's top-10. The legal online "soft top-k" is the router's
DECAYED ROUTING HISTORY: S[li,e] = EMA of past router scores for expert e
(decay 0.7 per 4 tokens — oMLX's own ExpertCache decay constant), which
ranks recently-routed-but-evicted experts: the recurrent-miss population
(78.95% of LRU misses recur, T6).

Mechanism:
  * After token t's demand misses resolve, each layer ranks NON-RESIDENT
    experts by soft score S (decayed routing history). Speculative
    page-ins are issued round-robin over layers within the measured
    idle-window budget (~52 experts/token at 5.02 GB/s).
  * CANCEL-ON-WRONG-ROUTE (deferred install): the read runs in the idle
    window (bytes spent either way — honest accounting), the expert is
    staged, NOT installed. When token t+1's route arrives:
      - expert routed at t+1 -> CONFIRM: install (0.30 ms), insert MRU;
        the demand miss at t+1 disappears (~0.82 ms saved).
      - not routed at t+1    -> CANCEL: no install, no cache pollution.
        Bytes already read are WASTED.
    (A no-cancel variant that installs immediately is reported for
    contrast; its pollution shows what cancel buys.)
  * Variant "hist1" issues only rank-1 soft candidates per layer;
    "hist3" the top-3 (one textbook setting each; no sweep).

Measured quantities: spec issued/confirmed/cancelled per token, SSD
MB/tok wasted vs useful, precision, net serial-model tok/s under both
install assumptions. All SIMULATED on locked synth traces with measured
IO constants (compute ASSUMED). Traces carry per-position router scores,
which seed the decayed soft-score state.

usage: python3 experiments/design_spec_pagein.py
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
EXPERT_MB = gs.EXPERT_MB
BW_BG_GBPS = 5.02            # chief_executor re-measured k=4 step rate
IDLE_MS = gs.COMPUTE_MS + L * gs.SYNC_MS
BUDGET = int(IDLE_MS * BW_BG_GBPS / EXPERT_MB)   # experts/token idle budget
IO_PER_MISS = gs.IO_B_MS + gs.INSTALL_MS         # 0.82 ms serial per miss

DECAY = 0.7 ** 0.25                 # oMLX ExpertCache: x0.7 per 4 tokens


def load_gold_scores(tdir):
    d = np.load(Path(tdir) / "holdout.npz")
    tags = sorted({k.rsplit("|", 1)[0] for k in d.files if k.endswith("|ids")})
    lay = {li: np.concatenate([d[f"{t}|L{li}_idx"] for t in tags])
           for li in range(L)}
    sc = {li: np.concatenate([d[f"{t}|L{li}_score"] for t in tags])
          for li in range(L)}
    T = min(len(lay[0]), len(sc[0]))
    gold = np.stack([lay[li][:T] for li in range(L)], 1).astype(np.int16)
    scores = np.stack([sc[li][:T] for li in range(L)], 1).astype(np.float32)
    return gold[:T], scores[:T]


def run(gold, scores, C, depth, cancel, warm=200):
    """true-LRU (hit-refresh) replay + speculative decayed-history page-in.

    depth=0     : baseline (no speculation)
    depth>0     : per layer, issue the top-`depth` non-resident experts by
                  decayed routing score S (spec phase), within budget
    cancel=True : deferred install, CANCEL on wrong route (the invention)
    cancel=False: install MRU immediately on issue (no-cancel contrast)
    """
    T = len(gold)
    lru = [OrderedDict() for _ in range(L)]
    S = np.zeros((L, E), np.float32)          # decayed routing history
    demand_miss = spec_issued = spec_conf = spec_cancel = 0
    staged = [dict() for _ in range(L)]       # staged (not installed)

    def touch(li, e, t):
        if e in lru[li]:
            lru[li].move_to_end(e)
            lru[li][e] = t
            return False
        lru[li][e] = t          # install on demand (MRU)
        evict_to(li, C)
        return True

    def evict_to(li, cap):
        while len(lru[li]) > cap:
            lru[li].popitem(last=False)

    def update_history(t):
        np.multiply(S, DECAY, out=S)
        for li in range(L):
            S[li, gold[t, li]] += 1.0

    # warm start: fill caches + history from the first `warm` tokens
    for t in range(warm):
        for li in range(L):
            for e in gold[t, li]:
                if e not in lru[li]:
                    lru[li][e] = t
                else:
                    lru[li].move_to_end(e)
                    lru[li][e] = t
            evict_to(li, C)
        update_history(t)

    Tw = T - warm
    for t in range(warm - 1, T - 1):
        # ---- demand phase for token t ----
        for li in range(L):
            for e in gold[t, li]:
                if touch(li, int(e), t):
                    demand_miss += 1
        update_history(t)

        if depth <= 0:
            continue

        # ---- speculative phase: soft top-k of decayed routing history ----
        budget = BUDGET
        rr = 0
        while budget > 0:
            progressed = False
            for li in list(range(L))[rr:] + list(range(L))[:rr]:
                if budget <= 0:
                    break
                order = np.argsort(-S[li])
                picked = 0
                for e in order:
                    if picked >= depth or budget <= 0:
                        break
                    e = int(e)
                    if S[li, e] <= 0.0 or e in lru[li] or e in staged[li]:
                        continue
                    picked += 1
                    budget -= 1
                    spec_issued += 1
                    progressed = True
                    if cancel:
                        staged[li][e] = t     # bytes read, install deferred
                    else:
                        # no-cancel: install immediately (pollution now)
                        lru[li][e] = t
                        evict_to(li, C)
            rr = (rr + 1) % L
            if not progressed:
                break

        # ---- route arrival for t+1: confirm or cancel staged experts ----
        if cancel:
            for li in range(L):
                nxt = set(int(e) for e in gold[t + 1, li])
                for e in list(staged[li].keys()):
                    if e in nxt:
                        spec_conf += 1       # CONFIRM: install, insert MRU
                        if e not in lru[li]:
                            lru[li][e] = t + 1
                            evict_to(li, C)
                        del staged[li][e]
                    else:
                        spec_cancel += 1     # CANCEL: bytes wasted
                        del staged[li][e]

    for li in range(L):                        # leftover staged -> cancelled
        spec_cancel += len(staged[li])

    miss_per_tok = demand_miss / Tw
    issued = spec_issued / Tw
    conf = spec_conf / Tw
    canc = spec_cancel / Tw
    wasted_mb = spec_cancel * EXPERT_MB / Tw
    useful_mb = spec_conf * EXPERT_MB / Tw
    # misses avoided: a confirmed expert is resident when demanded at t+1...
    # only counts if it would otherwise have been a miss; approximated by
    # conf (all confirms are non-resident at issue time and MRU-inserted).
    miss_saved = conf
    pess_ms = (gs.COMPUTE_MS
               + miss_per_tok * (gs.IO_B_MS + gs.INSTALL_MS)
               + conf * gs.INSTALL_MS
               + L * gs.SYNC_MS)
    opt_ms = (gs.COMPUTE_MS
              + max(0.0, miss_per_tok - miss_saved) * IO_PER_MISS
              + L * gs.SYNC_MS)
    return dict(
        gate="",
        cancel=cancel, tokens_scored=Tw, budget=BUDGET,
        demand_miss_per_tok=round(miss_per_tok, 2),
        spec_issued_per_tok=round(issued, 2),
        spec_confirmed_per_tok=round(conf, 3),
        spec_cancelled_per_tok=round(canc, 2),
        precision=round(spec_conf / max(1, spec_issued), 3),
        ssd_mb_wasted_per_tok=round(wasted_mb, 1),
        ssd_mb_useful_per_tok=round(useful_mb, 1),
        waste_per_hit_mb=round(wasted_mb / max(1e-9, conf), 1),
        serial_tps_pess=round(1000.0 / pess_ms, 2),
        serial_tps_opt=round(1000.0 / opt_ms, 2),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=143)
    ap.add_argument("--traces-dir", default=str(ROOT / "results/traces_synth"))
    ap.add_argument("--out", default=str(ROOT / "results/design_spec_pagein.json"))
    a = ap.parse_args()

    gold, scores = load_gold_scores(a.traces_dir)
    rows = []
    for depth, name in ((0, "none (baseline LRU, no speculation)"),
                        (1, "hist1"), (3, "hist3")):
        for cancel in (True, False):
            r = run(gold, scores, a.cap, depth, cancel)
            r["gate"] = name
            rows.append(r)

    out = dict(
        kind="simulated",
        mechanism=("speculative page-in from router soft top-k "
                   "(self-prediction from t-1's own picks), "
                   "cancel-on-wrong-route = deferred install"),
        traces=a.traces_dir,
        holdout_sha256_12=hashlib.sha256(
            Path(a.traces_dir, "holdout.npz").read_bytes()).hexdigest()[:12],
        constants=("serial model measured (A .20/B .52/install .30/sync .122), "
                   "bg BW 5.02 GB/s (measured k4), compute ASSUMED 23.2"),
        budget_experts_per_token=BUDGET,
        rows=rows,
    )
    Path(a.out).write_text(json.dumps(out, indent=1))
    for r in rows:
        print(json.dumps(r))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
