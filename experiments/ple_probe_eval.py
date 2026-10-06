"""Phase-0b: can the model's OWN n-gram pathway predict routing?

For each MoE layer, fit a ridge multi-label classifier
    PLE feature [d] -> 512-dim oracle top-k membership
on build traces, then score candidate-budget coverage on holdout,
comparing against the hand-rolled hashed 3-gram table and baselines.

Two feature sets:
  ngramemb  : raw hashed n-gram table embedding (2560-d, layer-1,
              pure function of token ids — legitimate pre-router trigger)
  pleinj    : full PLE injection value (10240-d, includes the query gate)

usage: .venv/bin/python experiments/ple_probe_eval.py [--budget 14]
"""
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_split(path):
    d = np.load(path)
    tags = sorted({k.rsplit("|", 1)[0] for k in d.files if k.endswith("|ids")})
    ids = np.concatenate([d[f"{t}|ids"] for t in tags])
    feats = {name: np.concatenate([d[f"{t}|PLE_{name}"] for t in tags])
             for name in ("ple", "ngram")}
    lay, sco = {}, {}
    for li in range(48):
        idx = np.concatenate([d[f"{t}|L{li}_idx"] for t in tags])
        lay[li] = idx
        sco[li] = np.concatenate([d[f"{t}|L{li}_score"] for t in tags]).astype(np.float32)
    # rows align with ids[:-1]? PLE rows lag ids by 1 (observed: rows=155 ids=156)
    T = min(len(ids), len(feats["ple"]), len(lay[0]))
    return ids[:T], feats, lay, T, sco


def ridge_multilabel(X, Y, lam):
    """closed-form multi-label ridge; X [N,d] f32, Y [N,E] f32 -> W [d,E]."""
    d = X.shape[1]
    XtX = X.T @ X + lam * np.eye(d, dtype=np.float32)
    XtY = X.T @ Y
    return np.linalg.solve(XtX, XtY)


def mass_coverage(cand_top, gold, score):
    """fraction of renormalized oracle weight landing inside the cand set."""
    tot = 0.0
    for i in range(cand_top.shape[0]):
        cs = set(cand_top[i].tolist())
        tot += float(score[i][[e in cs for e in gold[i].tolist()]].sum())
    return tot / cand_top.shape[0]


def coverage(cand_top, gold, k):
    """mean |cand ∩ gold| / k over rows; cand_top [N,b] int, gold [N,k] int."""
    hits = 0.0
    for i in range(cand_top.shape[0]):
        hits += len(set(cand_top[i].tolist()) & set(gold[i].tolist()))
    return hits / (cand_top.shape[0] * k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=14)
    ap.add_argument("--lam", type=float, default=200.0)
    ap.add_argument("--train-sub", type=int, default=80000)
    ap.add_argument("--eval-sub", type=int, default=80000)
    a = ap.parse_args()

    b_tr, b_f, b_l, bT, b_s = load_split(ROOT / "results/traces/build.npz")
    h_tr, h_f, h_l, hT, h_s = load_split(ROOT / "results/traces/holdout.npz")
    E, K = 512, 10
    rng = np.random.default_rng(0)

    # rows with PLE features correspond to ids positions 1..T (1-token lag
    # verified in smoke: rows = ids-1); router rows share the same order.
    bt = np.arange(min(bT, len(b_f["ngram"])))
    ht = np.arange(min(hT, len(h_f["ngram"])))
    bs = rng.choice(bt, size=min(a.train_sub, len(bt)), replace=False)
    hs = rng.choice(ht, size=min(a.eval_sub, len(ht)), replace=False)

    out = {"config": vars(a), "build_rows": int(len(bt)),
           "holdout_rows": int(len(ht)), "per_layer": {}}
    cov = {}
    for li in range(48):
        Yb = np.zeros((len(bs), E), np.float32)
        Yb[np.arange(len(bs))[:, None], b_l[li][bs]] = 1.0
        res = {}
        for name in ("ngram", "ple"):
            Xb = b_f[name][bs].astype(np.float32)
            Xh = h_f[name][hs].astype(np.float32)
            mu, sd = Xb.mean(0), Xb.std(0) + 1e-6
            Xb = (Xb - mu) / sd
            Xh = (Xh - mu) / sd
            W = ridge_multilabel(Xb, Yb, a.lam)
            s = Xh @ W
            top = np.argpartition(-s, a.budget - 1, axis=1)[:, :a.budget]
            res[name] = coverage(top, h_l[li][hs], K)
            res[name + "_mass"] = mass_coverage(top, h_l[li][hs], h_s[li][hs])
        # prior baseline (context-free hot set from build)
        prior = np.bincount(b_l[li][bs].ravel(), minlength=E)
        hot = np.argsort(-prior)[:a.budget]
        idxs = np.arange(0, len(hs), 4)
        res["prior"] = np.mean([len(set(hot.tolist()) &
                                    set(h_l[li][hs][i].tolist()))
                                for i in idxs]) / K
        res["prior_mass"] = mass_coverage(np.tile(hot, (len(idxs), 1)),
                                          h_l[li][hs][idxs], h_s[li][hs][idxs])
        out["per_layer"][li] = {k: round(v, 4) for k, v in res.items()}
        for k, v in res.items():
            cov.setdefault(k, []).append(v)
        print(f"L{li:02d} " + " ".join(f"{k}={v:.3f}" for k, v in res.items()),
              flush=True)
    out["overall"] = {k: float(np.mean(v)) for k, v in cov.items()}
    (ROOT / "results/ple_probe.json").write_text(json.dumps(out, indent=1))
    print("overall:", json.dumps({k: round(v, 4) for k, v in out["overall"].items()}))


if __name__ == "__main__":
    main()
