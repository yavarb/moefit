"""Phase-0 go/no-go: n-gram -> routing prediction study.

Builds n-gram conditioned expert-candidate tables from the build split,
scores coverage on the holdout split, per layer and per traffic class,
plus two baselines for calibration:
  - prior: global top-budget experts per layer (context-free)
  - prev : the SAME positions top-k experts (temporal persistence)
  - union: ngram table intersected/merged with prev-token set

usage: .venv/bin/python experiments/trigger_eval.py --budget 14 --ngram 3
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moefit.trigger.ngram_table import NgramExpertTable, ngram_key

TRACES = Path(__file__).resolve().parents[1] / "results" / "traces"


def load_split(split):
    """-> list of dicts {id, cat, ids[T], layers: {li: (idx[T,k], sc[T,k])}}"""
    z = np.load(TRACES / f"{split}.npz")
    # keys: "pN_pid|ids", "pN_pid|L12_idx", ...
    seqs = defaultdict(dict)
    for k in z.files:
        seq, field = k.split("|", 1)
        seqs[seq][field] = z[k]
    out = []
    for seq, d in sorted(seqs.items()):
        pid = seq.split("_", 1)[1]
        rec = {"pid": pid, "ids": d["ids"].tolist(), "layers": {}}
        for f in d:
            if f.endswith("_idx"):
                li = int(f[1:-4])
                rec["layers"][li] = (d[f].astype(np.int64),
                                     d[f"L{li}_score"].astype(np.float32))
        out.append(rec)
    return out


def cat_of(pid, corpus):
    for c in corpus:
        if c["id"] == pid:
            return c["cat"]
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=14)
    ap.add_argument("--ngram", type=int, default=3)
    ap.add_argument("--corpus", default=str(Path(__file__).resolve().parents[1] / "data/prompts.jsonl"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    corpus = [json.loads(l) for l in Path(a.corpus).read_text().splitlines() if l.strip()]
    build = load_split("build")
    hold = load_split("holdout")
    layers = sorted(build[0]["layers"].keys())
    k = build[0]["layers"][layers[0]][0].shape[1]
    budget = max(a.budget, k)

    tab = NgramExpertTable(n=a.ngram, cand_budget=budget)
    # per-layer global prior counts
    prior_cnt = {li: np.zeros(512) for li in layers}
    for rec in build:
        for li, (idx, sc) in rec["layers"].items():
            tab.observe(li, rec["ids"], idx, sc)
            for row in idx:
                prior_cnt[li][row] += 1
    tab.finalize()
    prior_top = {li: set(np.argsort(-prior_cnt[li])[:budget].tolist())
                 for li in layers}

    percat = defaultdict(lambda: [0.0, 0.0, 0.0, 0])  # cov, prev, union, n
    perlayer = defaultdict(lambda: [0.0, 0.0, 0.0, 0])  # cov, prev, union, n
    total = {"ngram": 0.0, "prior": 0.0, "prev": 0.0, "union": 0.0,
             "exact_ngram": 0.0, "exact_prev": 0.0, "seen": 0.0, "n": 0}
    cold_by_len = defaultdict(lambda: [0.0, 0])

    for rec in hold:
        cat = cat_of(rec["pid"], corpus)
        for li, (idx, sc) in rec["layers"].items():
            for t in range(idx.shape[0]):
                key = ngram_key(rec["ids"][: t + 1], a.ngram)
                gs = set(idx[t].tolist())
                c = tab.candidates(key, li)
                cov = len(set(c[0].tolist()) & gs) / k if c is not None else 0.0
                cov_prior = len(prior_top[li] & gs) / k
                if t > 0:
                    prev = set(idx[t - 1].tolist())
                    cov_prev = len(prev & gs) / k
                    cov_union = len((prev | (set(c[0].tolist()) if c is not None else set())) & gs) / k
                else:
                    cov_prev = cov_union = 0.0
                exact = int(c is not None and gs == set(c[0][:k].tolist()))
                exact_prev = int(t > 0 and gs == set(idx[t - 1].tolist()))
                seen = int(key in tab.cand and li in tab.cand.get(key, {}))
                pc = percat[cat]; pc[0] += cov; pc[1] += cov_prev; pc[2] += cov_union; pc[3] += 1
                pl = perlayer[li]; pl[0] += cov; pl[1] += cov_prev; pl[2] += cov_union; pl[3] += 1
                total["ngram"] += cov; total["prior"] += cov_prior
                total["prev"] += cov_prev; total["union"] += cov_union
                total["exact_ngram"] += exact; total["exact_prev"] += exact_prev
                total["seen"] += seen; total["n"] += 1
                cl = cold_by_len[min(t // 32, 8)]; cl[0] += seen; cl[1] += 1

    n = total["n"]
    report = {
        "config": {"ngram": a.ngram, "budget": budget, "top_k": k,
                   "build_seqs": len(build), "holdout_seqs": len(hold),
                   "eval_positions": n},
        "overall": {key: total[key] / n for key in total if key != "n"},
        "by_class": {c: {"coverage_ngram": v[0]/v[3], "coverage_prev": v[1]/v[3],
                         "coverage_union": v[2]/v[3], "positions": v[3]}
                     for c, v in sorted(percat.items())},
        "by_layer": {str(li): {"ngram": v[0]/v[3], "prev": v[1]/v[3],
                               "union": v[2]/v[3]}
                     for li, v in sorted(perlayer.items())},
        "key_seen_by_decile": {f"pos_bin{b}": v[0]/v[1] for b, v in sorted(cold_by_len.items())},
        "table": {"n_keys": len(tab.cand), "build_tokens": tab.n_seen},
    }
    print(json.dumps(report, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
