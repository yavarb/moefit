"""T7 offline statistical audit of measured 17-point regression.
Permutation constructions are null diagnostics, NOT silicon measurements.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def fit(x, y) -> dict:
    X = np.column_stack((np.ones(len(x)), x))
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residual = y - X @ beta
    cov = np.linalg.inv(X.T @ X) * (residual @ residual) / (len(y)-2)
    se = np.sqrt(np.diag(cov))
    return dict(n=len(y), intercept=float(beta[0]), slope=float(beta[1]),
                intercept_se=float(se[0]), slope_se=float(se[1]),
                r2=float(1-residual @ residual / np.sum((y-y.mean())**2)),
                intercept_slope_corr=float(cov[0,1]/np.prod(se)),
                intercept_normal95=[float(beta[0]-1.96*se[0]),float(beta[0]+1.96*se[0])])


def main():
    path = ROOT/'results/silicon_sc36/regress_pooled.json'
    src = json.loads(path.read_text())
    x = np.array([r['MB'] for r in src['points']])
    y = np.array([r['ms'] for r in src['points']])
    observed = fit(x,y)
    assert abs(observed['slope']-src['slope_ms_per_MB']) < .0001
    assert abs(observed['intercept']-src['intercept_ms']) < .1
    # Reconstructed rate in MB/s; original fields are rounded, so this
    # audit deliberately tests the exact algebra of the pooled table.
    rate = 1000*x/y
    assert np.allclose(rate*y/1000,x)
    rng = np.random.default_rng(731)
    count = 20000
    perm_rate = np.array([rng.permutation(rate) for _ in range(count)])
    xp = perm_rate*y[None,:]/1000
    xc = xp-xp.mean(axis=1,keepdims=True)
    yc = y-y.mean()
    cross = np.sum(xc * yc[None, :], axis=1)
    slopes = cross/np.sum(xc*xc,axis=1)
    intercepts = y.mean()-slopes*xp.mean(axis=1)
    rsq = cross**2/(np.sum(xc*xc,axis=1)*np.sum(yc*yc))
    assert np.isfinite(slopes).all() and np.isfinite(rsq).all()
    # Noncausal construction with rate fixed: time can vary for reasons
    # unrelated to IO, and both ratios share the same token-rate divisor.
    fixed_rate_x = rate.mean()*y/1000
    mechanical = fit(fixed_rate_x,y)
    assert abs(mechanical['r2']-1)<1e-12
    assert abs(mechanical['intercept'])<1e-9
    omitted = []
    for prompt in sorted(set(r['src'].split(':')[-1] for r in src['points'])):
        mask = np.array([r['src'].split(':')[-1] != prompt for r in src['points']])
        omitted.append(dict(omitted=prompt, **fit(x[mask],y[mask])))
    out = dict(kind='offline reanalysis of measured aggregates + synthetic null diagnostics; no new silicon',
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        observed_ols=observed, observed_x_range_MB=[float(x.min()),float(x.max())],
        fixed_term_candidates={str(v):dict(z_from_intercept=(v-observed['intercept'])/observed['intercept_se'],
            in_normal95=observed['intercept_normal95'][0]<=v<=observed['intercept_normal95'][1]) for v in (30.,36.)},
        permutation=dict(seed=731,n=count,
            null='Permute measured rate across measured token durations; reconstruct x=rate*y/1000. Removes rate-duration pairing but preserves shared duration factor.',
            slope_p025_p50_p975=np.quantile(slopes,[.025,.5,.975]).tolist(),
            intercept_p025_p50_p975=np.quantile(intercepts,[.025,.5,.975]).tolist(),
            r2_p025_p50_p975=np.quantile(rsq,[.025,.5,.975]).tolist(),
            fraction_r2_ge_observed=float(np.mean(rsq>=observed['r2'])),
            caution='Diagnostic null, NOT a causal p-value; prompt clusters, rounded aggregates and measurement endogeneity remain.'),
        constant_rate_counterexample=mechanical,
        leave_source_prompt_out=omitted,
        verdict='Descriptive association reproduced; shared denominator permits association without causal miss latency. Both candidate fixed terms remain inside even the approximate normal95 interval. Compute not statistically resolved by this intercept.',
        caveats=['OLS normal95 is optimistic for n=17 and clustered prompts; not a definitive coverage guarantee.',
                 'Intercept extrapolates to 0 MB/token outside observed support.',
                 'Slope per physical MB converts to cost per logical expert only with independently validated byte/miss mapping.',
                 'Microbench evidence of serialization is independent and not refuted.'])
    target=ROOT/'results/t7_regression_identifiability.json'
    target.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='leave_source_prompt_out'},indent=2))


if __name__=='__main__':
    main()
