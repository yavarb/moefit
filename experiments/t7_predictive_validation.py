"""T7 held-prompt prediction and conditional overlap uncertainty.
Existing silicon aggregates only. Resampled bounds are NOT measured gains.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def ols(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta


def main():
    folder = ROOT/'results/silicon_sc36'
    paths = [folder/'regress1.json', folder/'regress2.json']
    rows = [r for p in paths for r in json.loads(p.read_text())['points']]
    assert len(rows) == 12 and all(r['tokens'] == 256 for r in rows)
    names = sorted(set(r['prompt'] for r in rows))
    assert len(names) == 6
    x = np.array([r['disk_MB_per_token'] for r in rows])
    y = np.array([r['ms_per_token'] for r in rows])
    groups = np.array([r['prompt'] for r in rows])
    predicted = np.empty(len(y))
    baseline = np.empty(len(y))
    folds = []
    for name in names:
        test = groups == name
        train = ~test
        beta = ols(x[train], y[train])
        predicted[test] = beta[0] + beta[1]*x[test]
        baseline[test] = y[train].mean()
        folds.append(dict(prompt=name, actual_ms=y[test].tolist(),
                          predicted_ms=predicted[test].tolist(),
                          baseline_ms=baseline[test].tolist(),
                          train_intercept=float(beta[0]), train_slope=float(beta[1])))
    rmse = lambda a: float(np.sqrt(np.mean((a-y)**2)))
    # Cluster bootstrap: each draw resamples 6 prompt groups, preserving
    # both observed runs. Exclude rank-deficient draws (single unique group).
    rng = np.random.default_rng(744)
    betas = []
    rejected = 0
    for _ in range(10000):
        selected = rng.choice(names, size=len(names), replace=True)
        if len(set(selected)) < 2:
            rejected += 1
            continue
        idx = np.concatenate([np.flatnonzero(groups == name) for name in selected])
        betas.append(ols(x[idx], y[idx]))
    b = np.array(betas)
    anchor_MB = 140.5
    fixed = b[:,0]
    variable = b[:,1]*anchor_MB
    admissible = (fixed>0)&(variable>0)
    serial = fixed[admissible]+variable[admissible]
    overlap = np.maximum(fixed[admissible],variable[admissible])
    speed = 1000/overlap
    gain = serial/overlap
    assert np.all((gain>=1)&(gain<=2))
    beta = ols(x,y)
    point = 1000/max(beta[0], beta[1]*anchor_MB)
    result = dict(kind='held-prompt reanalysis of existing measured data + conditional model bootstrap; no new silicon',
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        actual_completion_tokens=256, n_rows=12,n_prompt_groups=6,
        held_prompt_out=dict(regression_rmse_ms=rmse(predicted),mean_baseline_rmse_ms=rmse(baseline),
                             regression_mae_ms=float(np.mean(abs(predicted-y))),
                             mean_baseline_mae_ms=float(np.mean(abs(baseline-y))),folds=folds),
        full_12_row_fit=dict(intercept_ms=float(beta[0]),slope_ms_per_MB=float(beta[1])),
        conditional_overlap=dict(assumptions='Two separable positive stages; complete overlap; no contention, extra IO, or changed installation costs. Descriptive coefficients assumed to represent stages, not established by regression.',
            anchor_physical_MB_per_token=anchor_MB, point_tps=float(point),
            seed=744,bootstrap_draws=10000,excluded_single_group_draws=rejected,
            positive_stage_draws=int(admissible.sum()),nonpositive_stage_draws=int((~admissible).sum()),
            tps_p025_p50_p975=np.quantile(speed,[.025,.5,.975]).tolist(),
            gain_p025_p50_p975=np.quantile(gain,[.025,.5,.975]).tolist(),
            caveat='Percentile sensitivity interval conditional on positive stages; only six clusters. NOT a measured speedup, calibrated confidence interval, or universal upper bound.'),
        normalization_clarification='All counts are256 (255 timed token intervals in collector). Variable-token-count denominator bias is absent. Previous rate permutations demonstrate possible duration exposure, not actual correlated measurement error. A matched independently integrated byte count would avoid constructing total bytes from rate times full duration.',
    )
    target=ROOT/'results/t7_predictive_validation.json'
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
