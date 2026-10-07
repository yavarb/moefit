"""T3 fidelity, part 5: FIRST LOOK at real router traces (one prompt).

results/traces/holdout.npz appeared 2026-10-07 (design_inventor's
collection, paused by the Chief voice order after ONE prompt):
  p0_code_py_review2, 144 tokens, per-layer top-10 expert idx + scores.
No build split, no PLE features, cold-only. This is NOT a steady-state
anchor: 143 usable tokens, cache starts empty, the transient dominates.

What it CAN honestly answer (and nothing more):
  1. Does REAL routing's cold-window miss pattern resemble the synth
     trace's same window, or does it differ materially?
  2. Same comparison under the oMLX-exact policy (decayed routing count,
     +1 per route, x0.7 every 4 calls, current call protected — source
     read by design_inventor).
  3. Cold-transient serial-model curve on real vs synth routing vs the
     measured warmup point (6.43 tok/s over the first 16 tokens).

usage: python3 experiments/fidelity_real_traces_firstlook.py
"""
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import sim_paging as sp  # noqa: E402

CAP = 143
IO_A_MS, IO_B_MS = sp.SERIAL_IO_A_MS, sp.SERIAL_IO_B_MS
INSTALL_MS, SYNC_MS = sp.SERIAL_INSTALL_MS, sp.SERIAL_SYNC_MS


def load_real(path):
    d = np.load(path)
    tags = sorted({k.rsplit("|", 1)[0] for k in d.files if k.endswith("|ids")})
    assert len(tags) == 1, f"expected one prompt, got {tags}"
    t = tags[0]
    idx = [d[f"{t}|L{li}_idx"] for li in range(sp.L)]
    T = min(len(x) for x in idx)
    gold = np.stack([idx[li][:T] for li in range(sp.L)], axis=1).astype(np.int16)
    scores = np.stack([d[f"{t}|L{li}_score"][:T] for li in range(sp.L)],
                      axis=1).astype(np.float32)
    return gold, scores


def lru_misses(gold, cap):
    """true-LRU (hit refresh) per-token misses, from EMPTY caches."""
    caches = [OrderedDict() for _ in range(sp.L)]
    miss = np.zeros(len(gold), int)
    for t in range(len(gold)):
        for li in range(sp.L):
            c = caches[li]
            for e in set(int(x) for x in gold[t, li]):
                if e in c:
                    c.move_to_end(e)
                else:
                    miss[t] += 1
                    c[e] = 1
                    if len(c) > cap:
                        c.popitem(last=False)
    return miss


def omlx_misses(gold, cap, decay_every=4, decay=0.7):
    """oMLX 0.7.0 ExpertCache: per layer, counts[e] += 1 on route; every
    `decay_every` calls counts *= decay; evict argmin(counts) among
    residents NOT in the current call's ids (current protected)."""
    res = [dict() for _ in range(sp.L)]     # e -> decayed count
    calls = [0] * sp.L
    miss = np.zeros(len(gold), int)
    for t in range(len(gold)):
        for li in range(sp.L):
            r = res[li]
            cur = set(int(x) for x in gold[t, li])
            for e in cur:
                if e not in r:
                    miss[t] += 1
                    while len(r) >= cap:
                        # evict argmin count among non-protected residents
                        cand = [x for x in r if x not in cur]
                        if not cand:
                            break
                        del r[min(cand, key=lambda x: r[x])]
                r[e] = r.get(e, 0.0) + 1.0
            calls[li] += 1
            if calls[li] % decay_every == 0:
                for e in r:
                    r[e] *= decay
    return miss


def serial_cold_ms(miss, compute_ms):
    """Per-token serial ms given per-TOKEN miss counts (layer detail
    approximated: k spread evenly over layers with misses)."""
    ms = []
    for m in miss:
        if m == 0:
            ms.append(compute_ms + sp.L * SYNC_MS)
            continue
        # distribute m misses over layers (uniform over all 48 at cold
        # start most layers have misses; exact split needs per-layer M)
        per = m / sp.L
        io = sp.L * (IO_A_MS + IO_B_MS * per)
        ms.append(compute_ms + io + INSTALL_MS * m + sp.L * SYNC_MS)
    return np.array(ms)


def main():
    real_gold, real_scores = load_real(ROOT / "results/traces/holdout.npz")
    bf, bl, bT = sp.load_split(ROOT / "results/traces_synth/build.npz")
    hf, hl, hT = sp.load_split(ROOT / "results/traces_synth/holdout.npz")
    synth_gold = np.stack([hl[li] for li in range(sp.L)],
                          axis=1).astype(np.int16)[:len(real_gold)]

    out = {"kind": "simulated (replay)",
           "real_trace": "results/traces/holdout.npz (p0_code_py_review2, "
                         "144 tok, cold-only, single prompt — NOT a steady anchor)",
           "cap": CAP}

    # routing-shape comparison (policy-free)
    def shape(g):
        # all-layer cross-position repeat fraction (the metric the synth
        # calibration targets); L0-only is misleading (lowest-persist layer)
        fracs = []
        for li in range(sp.L):
            fr = np.mean([len(set(g[t, li]) & set(g[t + 1, li]))
                          for t in range(len(g) - 1)])
            fracs.append(fr / sp.K)
        return dict(cross_position_repeat_all_layers=float(np.mean(fracs)),
                    repeat_layer_min=float(np.min(fracs)),
                    repeat_layer_max=float(np.max(fracs)),
                    cross_token_repeat_frac_L0=float(fracs[0]))

    out["routing_shape"] = {"real": shape(real_gold), "synth": shape(synth_gold)}

    res = {}
    for label, gold in (("real", real_gold), ("synth_first143", synth_gold)):
        m_lru = lru_misses(gold, CAP)
        m_omlx = omlx_misses(gold, CAP)
        s16_lru = serial_cold_ms(m_lru[:16], 18.1)
        s16_omlx = serial_cold_ms(m_omlx[:16], 18.1)
        res[label] = dict(
            tokens=int(len(gold)),
            lru_miss_per_tok_cold=float(m_lru.mean()),
            lru_miss_first16=float(m_lru[:16].mean()),
            omlx_miss_per_tok_cold=float(m_omlx.mean()),
            omlx_miss_first16=float(m_omlx[:16].mean()),
            serial_cold16_tps_lru=float(1000 / s16_lru.mean()),
            serial_cold16_tps_omlx=float(1000 / s16_omlx.mean()),
        )
    out["cold_window_cap143"] = res
    out["measured_reference"] = {"warmup_16tok_tps": 6.43,
                                 "steady_cap143_tps": "12.71-13.07 (NOT "
                                 "comparable: this trace is cold+single-prompt)"}

    p = ROOT / "results/fidelity_real_traces_firstlook.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
