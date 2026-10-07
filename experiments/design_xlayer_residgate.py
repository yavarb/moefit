"""T1 v3: residual-mass completion gate (G5), FIT on train prompts, scored on held-out.

Train = first 4 captured prompts, held-out = last 4 (same split as T8). One scalar
(theta) fit by maximizing train net ms/tok over a fixed log-spaced candidate set;
held-out evaluated once at the fitted theta. SIM timing, real routes."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import design_xlayer_guarded as g
import design_xlayer_twostage as ts

W, tags, X, I = g.load(sys.argv[1])
P1 = [g.preview(W, x, 1) for x in X]
P2 = [g.preview(W, x, 2) for x in X]
tr, te = slice(0, 4), slice(4, 8)
cands = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
fit = []
for th in cands:
    r = ts.run(P1[tr], P2[tr], I[tr], 143, 0.01, 10, {"d1"}, warm=80, resid_gate=th)
    fit.append((r["net_ms"], th, r)); print("train", th, r["net_ms"], r["spec_per_tok"], r["full_cover"], flush=True)
best = max(fit, key=lambda t: t[0]); th = best[1]
out = dict(kind="SIM timing on REAL routes; theta fit on train prompts 0-3, scored on held-out 4-7",
           theta=th, train=[f[2] for f in fit])
for nm, kw in (("ungated_d1", {}), ("G5_d1", {"resid_gate": th})):
    r = ts.run(P1[te], P2[te], I[te], 143, 0.01, 10, {"d1"}, warm=80, **kw)
    out[nm] = r; print("heldout", nm, json.dumps(r), flush=True)
for c in (0.073, 0.198):
    for nm in ("ungated_d1", "G5_d1"):
        r = out[nm]; net = r["stall_removed_ms"] - r["spec_per_tok"] * c
        out.setdefault("contention_bounds", {})[f"{nm}@{c}"] = dict(net_ms=round(net, 2), tps=round(1000 / (63.6 - net), 2))
print(json.dumps(out["contention_bounds"]))
Path("results/design_xlayer_residgate.json").write_text(json.dumps(out, indent=1))
