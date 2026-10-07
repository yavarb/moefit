"""T3 invention: IDLE-DECAYED COUNT eviction vs true-LRU and oMLX decayed-count.

Assignment (Chief voice order, invention-only lab): improved hot-set/eviction,
measure miss rate + simulated tok/s vs baseline. Code, not docs.

Baselines:
  lru        true-LRU (hit refresh) — the retro's hit-refresh fix, already in sim.
  omlx_decay oMLX 0.7.0 ExpertCache exact replay: per-layer counts, +1 per
             route, x0.7 every 4 calls (global clock), current call protected
             (source read by design_inventor; replay validated by T6:
             57.407/56.911 miss/tok @cap143 global-drop200).

INVENTION — idle_decay: decay applied on the RESIDENT'S OWN idle time instead
of a global clock. On use at token t:  score = score_prev * g^(t - t_last) + 1,
g = 0.7^(1/4) per token (the EXACT aggregate decay rate of oMLX's x0.7-every-4
— anchored, not tuned). Evict argmin score, current call protected (same rule
as oMLX for fairness).

Mechanism-level motivation: oMLX's global-clock decay discounts ALL counts
equally per call, so an expert that was burst-hot 20 tokens ago keeps a high
count while a steady low-rate core expert's count decays at the same rate —
the burst leftover wins the argmin eviction race against the genuinely
recurring core. Idle decay makes stale bursts decay by their own staleness.
The traffic facts that motivate it (T6 exact stack distances): a tiny
reused core (D p50=24) plus 78.95% of misses recurring later — residency
given to a stale burst is residency taken from the recurring tail.

Measurement protocol (per T6's window audit): report the cache-reset policy
and BOTH windows — global drop-first-200 (comparable to the published
anchors) AND per-prompt suffixes (drop first 128 of EACH prompt; the
window that cannot masquerade a policy effect). Serial tok/s via the
measured constants (A=0.20/B=0.52/install=0.30/sync=0.122, compute 24.1
410-bin) on the per-token miss matrix; SIM, not silicon.

usage: python3 experiments/design_idle_decay.py
"""
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import sim_paging as sp  # noqa: E402

G_IDLE = 0.7 ** 0.25          # per-token decay, anchored to oMLX's aggregate rate
WARMUP_GLOBAL = 200
SUFFIX_SKIP = 128            # per-prompt suffix window (T6 protocol)


def load_gold():
    d = np.load(ROOT / "results/traces_synth/holdout.npz")
    tags = sorted({k.rsplit("|", 1)[0] for k in d.files if k.endswith("|ids")})
    # per-tag row counts give the true prompt boundaries (load_split
    # concatenates tags in this order); boundaries recovered without gaps
    counts = [len(d[f"{t}|PLE_ngram"]) for t in tags]
    hf, hl, hT = sp.load_split(ROOT / "results/traces_synth/holdout.npz")
    assert hT == sum(counts)
    rng = np.random.default_rng(0)
    hs = np.sort(rng.choice(hT, size=min(8000, hT), replace=False))
    gold = np.stack([hl[li][hs] for li in range(sp.L)], axis=1).astype(np.int16)
    return gold, np.array(hs), counts


def prompt_boundaries(hs, counts):
    """Map the global holdout row index -> position within the selected hs
    array, then cut at tag boundaries: a prompt ends where hs crosses a
    cumulative tag count. Rows selected with seed 0 are position-preserving
    in order; boundaries = first selected position after each cut."""
    cuts = np.cumsum([0] + counts)
    b = [0]
    for c in cuts[1:-1]:
        pos = np.searchsorted(hs, c, side="left")
        b.append(int(pos))
    b.append(len(hs))
    # strictly increasing, dedupe
    return sorted(set(b))


def replay(gold, cap, policy):
    """Per-layer cache replay. Returns (T, L) per-token per-layer misses."""
    T = len(gold)
    M = np.zeros((T, sp.L), np.int16)
    if policy == "lru":
        caches = [OrderedDict() for _ in range(sp.L)]
        for t in range(T):
            for li in range(sp.L):
                c = caches[li]
                for e in set(int(x) for x in gold[t, li]):
                    if e in c:
                        c.move_to_end(e)
                    else:
                        M[t, li] += 1
                        c[e] = 1
                        if len(c) > cap:
                            c.popitem(last=False)
    elif policy == "omlx_decay":
        res = [dict() for _ in range(sp.L)]
        calls = [0] * sp.L
        for t in range(T):
            for li in range(sp.L):
                r = res[li]
                cur = set(int(x) for x in gold[t, li])
                for e in cur:
                    if e not in r:
                        M[t, li] += 1
                        while len(r) >= cap:
                            cand = [x for x in r if x not in cur]
                            if not cand:
                                break
                            del r[min(cand, key=lambda x: r[x])]
                    r[e] = r.get(e, 0.0) + 1.0
                calls[li] += 1
                if calls[li] % 4 == 0:
                    for e in r:
                        r[e] *= 0.7
    elif policy == "idle_decay":
        # v1 (count reset on use) THRASHED the core: steady-state score for
        # a gap-24 reuse converges to 1/(1-g^24) ~= 1.13 ~= a fresh single's
        # 1.0, so min-score eviction churns the core (112 miss/tok vs 57.4
        # LRU; recorded as the negative result it is).
        # v2: keep oMLX's PROVEN count accumulation (+1 on use, x0.7 every 4
        # calls, global) and apply the idle discount multiplicatively at
        # eviction time: score = count * g^(t - t_last). A single touch 200
        # tokens stale scores ~0 (evicted first); the recurring core keeps
        # its accumulated count; a stale burst loses only its staleness.
        res = [dict() for _ in range(sp.L)]      # e -> [count, t_last]
        calls = [0] * sp.L
        t_now = 0

        def score(entry):
            return entry[0] * (G_IDLE ** (t_now - entry[1]))

        for t_now in range(T):
            for li in range(sp.L):
                r = res[li]
                cur = set(int(x) for x in gold[t_now, li])
                for e in cur:
                    if e not in r:
                        M[t_now, li] += 1
                        while len(r) >= cap:
                            cand = [x for x in r if x not in cur]
                            if not cand:
                                break
                            del r[min(cand, key=lambda x: score(r[x]))]
                        r[e] = [1.0, t_now]
                    else:
                        r[e][0] += 1.0
                        r[e][1] = t_now
                calls[li] += 1
                if calls[li] % 4 == 0:
                    for e in r:
                        r[e][0] *= 0.7
    elif policy == "admit_second_chance":
        # T3 invention v2: ADMISSION CONTROL on the omlx-decay machinery.
        # Motivation (T6 exact replay): 78.95% of LRU misses recur later —
        # i.e. ~21% are one-shot. A one-shot miss that gets cached evicts a
        # useful resident for zero future benefit. Policy: first miss of an
        # expert is fetched but NOT installed (no eviction triggered);
        # second and later misses install under oMLX decay rules. No knobs.
        res = [dict() for _ in range(sp.L)]      # e -> count (omlx decay)
        miss_hist = [dict() for _ in range(sp.L)]  # e -> lifetime miss count
        calls = [0] * sp.L
        for t in range(T):
            for li in range(sp.L):
                r, hist = res[li], miss_hist[li]
                cur = set(int(x) for x in gold[t, li])
                for e in cur:
                    if e in r:
                        r[e] = r.get(e, 0.0) + 1.0
                        continue
                    M[t, li] += 1
                    hist[e] = hist.get(e, 0) + 1
                    if hist[e] >= 2:
                        # recurring miss: install under omlx eviction rules
                        while len(r) >= cap:
                            cand = [x for x in r if x not in cur]
                            if not cand:
                                break
                            del r[min(cand, key=lambda x: r[x])]
                        r[e] = 1.0
                calls[li] += 1
                if calls[li] % 4 == 0:
                    for e in r:
                        r[e] *= 0.7
    elif policy == "admit_warmup":
        # v3: second-chance admission + WARMUP BYPASS. The v2 global-window
        # loss came from cold-start compulsory misses: a first miss is gated,
        # then its second touch (which for warm-up experts comes SOON) pays a
        # second fetch. Fix: until a layer's cache is FULL, install on first
        # miss (classic fill); once eviction begins (cache full), gate first
        # misses and install only recurrent ones. The bypass is tied to
        # cache-fullness, not a token count — no knob.
        res = [dict() for _ in range(sp.L)]
        miss_hist = [dict() for _ in range(sp.L)]
        calls = [0] * sp.L
        for t in range(T):
            for li in range(sp.L):
                r, hist = res[li], miss_hist[li]
                warm = len(r) < cap          # bypass while cache not full
                cur = set(int(x) for x in gold[t, li])
                for e in cur:
                    if e in r:
                        r[e] = r.get(e, 0.0) + 1.0
                        continue
                    M[t, li] += 1
                    hist[e] = hist.get(e, 0) + 1
                    if hist[e] >= 2 or warm:
                        while len(r) >= cap:
                            cand = [x for x in r if x not in cur]
                            if not cand:
                                break
                            del r[min(cand, key=lambda x: r[x])]
                        r[e] = 1.0
                calls[li] += 1
                if calls[li] % 4 == 0:
                    for e in r:
                        r[e] *= 0.7
    else:
        raise ValueError(policy)
    return M


def serial_tps(M, window, compute_ms=24.1):
    k = M[window].astype(float)
    io = np.where(k > 0, sp.SERIAL_IO_A_MS + sp.SERIAL_IO_B_MS * k,
                  0.0).sum(axis=1)
    total = compute_ms + io + sp.SERIAL_INSTALL_MS * k.sum(axis=1) \
        + sp.L * sp.SERIAL_SYNC_MS
    return float(1000.0 / total.mean())


def main():
    gold, hs, counts = load_gold()
    bounds = prompt_boundaries(hs, counts)
    cap = 143
    out = {"kind": "simulated (replay + serial model)",
           "traces": "locked synth holdout (seed 0 selection)",
           "cap": cap, "policies": {},
           "protocol": {"cache_reset": "caches empty at t=0 (retained across "
                         "the whole concatenated holdout)",
                         "windows": ["global drop-first-200",
                                     "per-prompt suffix drop-128 (T6 protocol)"]},
           "validation_targets": {"lru_global_drop200": 57.407,
                                  "omlx_global_drop200": 56.911}}

    # suffix window: first SUFFIX_SKIP tokens of each prompt dropped
    suffix_mask = np.ones(len(gold), bool)
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        suffix_mask[lo:min(lo + SUFFIX_SKIP, hi)] = False
    w_global = slice(WARMUP_GLOBAL, None)

    for policy in ("lru", "omlx_decay", "idle_decay"):
        M = replay(gold, cap, policy)
        g_miss = float(M[w_global].sum() / len(gold[w_global]))
        s_miss = float(M[suffix_mask].sum() / int(suffix_mask.sum()))
        row = {
            "miss_per_tok_global_drop200": round(g_miss, 3),
            "miss_per_tok_prompt_suffix": round(s_miss, 3),
            "serial_tps_global": round(serial_tps(M, w_global), 2),
            "serial_tps_suffix": round(serial_tps(M, suffix_mask), 2),
        }
        # per-prompt suffix misses (T6's window that cannot masquerade)
        per_prompt = []
        for i in range(len(bounds) - 1):
            lo, hi = bounds[i], bounds[i + 1]
            if hi - lo <= SUFFIX_SKIP:
                continue
            w = slice(lo + SUFFIX_SKIP, hi)
            per_prompt.append(float(M[w].sum() / (hi - lo - SUFFIX_SKIP)))
        row["per_prompt_suffix_miss"] = [round(x, 3) for x in per_prompt]
        out["policies"][policy] = row
        print(policy, json.dumps(row), flush=True)

    # deltas vs baselines
    p = out["policies"]
    for base in ("lru", "omlx_decay"):
        if "idle_decay" in p and base in p:
            dg = p["idle_decay"]["miss_per_tok_global_drop200"] - p[base]["miss_per_tok_global_drop200"]
            ds = p["idle_decay"]["miss_per_tok_prompt_suffix"] - p[base]["miss_per_tok_prompt_suffix"]
            out[f"idle_decay_vs_{base}"] = {
                "miss_delta_global": round(dg, 3),
                "miss_delta_suffix": round(ds, 3),
                "miss_delta_pct_suffix": round(100 * ds / p[base]["miss_per_tok_prompt_suffix"], 2),
                "serial_tps_delta_suffix": round(
                    p["idle_decay"]["serial_tps_suffix"] - p[base]["serial_tps_suffix"], 2),
            }
    path = ROOT / "results/design_idle_decay.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", path)


if __name__ == "__main__":
    main()
