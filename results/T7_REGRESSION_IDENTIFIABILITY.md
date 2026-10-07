# T7 regression identifiability — astra_analysis_b | Astra gpt-6

Offline reanalysis of lead_silicon's measured 17-point pooled table. No new silicon acquisition. The null constructions below are synthetic diagnostics, not measured runs.

## Reproduced descriptive fit

`python3 experiments/t7_regression_identifiability.py` reproduces:

    ms/token = 39.2489 + 0.288272 * physical MB/token
    intercept SE = 8.3702 ms; slope SE = 0.053405
    R2 = 0.660146; intercept/slope covariance correlation = -0.994034

The source table and SHA256 are recorded in `t7_regression_identifiability.json`. Leave-source-prompt-out fits are included as diagnostics; no policy/hardware grid is run.

## Compute is not resolved by this intercept

The approximate normal 95% intercept interval is 22.84–55.65 ms. Both proposed total fixed terms, 30ms and 36ms (corresponding to the disputed compute assumptions plus other fixed costs), lie inside it. They sit 1.105 and 0.388 standard errors below the estimate. Being closer is not statistical resolution. Small-sample and prompt clustering caveats make this simple interval optimistic, not stronger evidence for separation.

Observed traffic spans 132.9–196.6 MB/token. The intercept is an extrapolation to zero traffic outside that range and includes all modeled/unmodeled fixed costs; it is not a direct compute measurement. Hardware bandwidth-bin knowledge may motivate a compute assumption, but does not turn that assumption into an observed compute duration.

## Shared-denominator coupling

The collector constructs:

    y = 1000 / tok_per_second
    x = disk_MB_per_second / tok_per_second = rate * y / 1000

Thus x and y share a measured duration factor. This does NOT automatically invalidate regression or prove the observed association is spurious. It does invalidate using association strength alone as independent causal confirmation of miss latency.

Two executable diagnostics:

1. Hold the reconstructed physical transfer rate constant while retaining observed durations. The constructed x and y have R2=1 and zero intercept by algebra, even if all duration variation were driven by a non-IO cause. This is a counterexample, not a model fitted to explain silicon.
2. Permute reconstructed rates across the measured durations, preserving the shared duration factor (seed731, 20000 draws). R2 has median0.57563 and 2.5–97.5% range0.31235–0.77930; 23.23% reach or exceed observed0.66015. Null slopes center0.30645. The observed slope0.28827 lies below the null's central interval: the diagnostic does NOT reproduce every detail of the measured fit. Its role is to show that a substantial R2 survives destruction of the original rate-duration pairing.

The permutation exceedance is NOT a causal p-value: exchangeability is not established across prompts, and aggregate rounding, clustering, traffic attribution, and endogeneity remain. The script uses the exact rounded pooled table and labels all derived/null values separately.

## Consequences and handoff

Keep: measured throughput variation; descriptive workload association; independent per-layer IO/install microbench evidence; source-read serial execution evidence.

Downgrade: 'compute term resolved', 'every constant independently measured-confirmed', and treating the fitted physical-MB slope as a directly measured logical-miss cost. The last conversion still needs independent physical/logical byte attribution (previous T7 provenance audit).

A useful next measurement within T1/T2's existing work is direct timing of resolve/install/compute plus logical misses and physical bytes on matched decode windows. Controlled perturbations with the same replayed workload can test miss-cost causality; changing prompt alone changes multiple costs simultaneously. No request to start another concurrent silicon benchmark.

The mincore snapshot's separate stock-versus-hit-flow interpretation is not analyzed here. This report does not endorse or refute that absorption claim.

Verification: original slope/intercept reproduction assertions; physical-rate reconstruction identity; finite null arrays; constant-rate algebra assertions; successful warning-free execution after replacing the large BLAS dot product with explicit elementwise reduction.
