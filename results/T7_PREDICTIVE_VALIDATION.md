# T7 predictive validation — astra_analysis_b | Astra gpt-6

Existing measured-data reanalysis only; hypothetical overlap estimates and resampling are not measured design results.

## Descriptive regression: KEEP for conditional prediction

The paired `regress1.json` and `regress2.json` contain six named prompts with two runs each. Every completion count is256. Holding out both runs of each prompt avoids training on another replicate of the test prompt.

Executed `python3 experiments/t7_predictive_validation.py`:

| Held-prompt score | Traffic regression | Training-mean baseline |
|---|---:|---:|
| Latency RMSE, ms/token | 4.9901 | 6.8240 |
| Latency MAE, ms/token | 3.7200 | 5.9338 |

This is evidence of useful conditional prediction in this small dataset, not mechanism identification. Test-time physical MB/token must already be measured: this is not an ex-ante prompt-only throughput predictor. All folds and input hashes are in `t7_predictive_validation.json`.

The five supplemental pooled runs are omitted deliberately, not because of their values: using just paired prompt runs yields balanced explicit prompt groups. This is a robustness analysis, not a replacement for the published17-point estimate. The held-out Chinese-story first run is poorly predicted (79.66ms vs actual91.66ms), so errors remain workload-dependent.

## Hypothetical overlap: not a firm25tok/s ceiling

The balanced12-row fit is:

    latency =45.5256 +0.254616 * physical MB/token

At140.5MB/token, interpreting these coefficients as two separable stages and perfectly overlapping them yields21.97tok/s, rather than the roughly25tok/s obtained with the17-row fit. The interpretation additionally assumes unchanged traffic, no contention, and no extra installation/synchronization cost; regression alone does not establish any of those conditions.

Prompt-cluster bootstrap (seed744,10000 draws, resampling six groups with replacement and keeping both runs together):

    positive-stage draws9344; nonpositive-stage draws656
    conditional overlap tps percentiles2.5/50/97.5:13.29/21.39/24.37
    corresponding gain percentiles:1.105/1.715/1.948

The discarded nonpositive-stage draws are reported explicitly: the proposed two-positive-stage physical interpretation fails in those fits. These are sensitivity percentiles conditional on admissible stages, NOT a calibrated confidence interval, not hardware speedup, and not a universal upper bound. Six prompt clusters are too few for strong bootstrap coverage claims. The algebraic two-stage gain is asserted to stay within1–2 for every admissible draw.

## Normalization clarification / self-audit

T3 correctly notes that constant completion counts remove the usual variable-token-count shared-denominator bias. The collector times255 intervals for256 completion tokens; the normalization scale is constant across these runs.

My previous rate-permutation diagnostics show that duration exposure can generate association, not that this dataset actually has correlated measurement error. Do not cite them as proof of such error or as a causal p-value. If bytes were independently integrated over the exact same timed window, constant-N normalization is simply a harmless rescaling of total time versus total bytes. The current measurement instead estimates bytes/token by multiplying an interior-window average disk rate by full-window seconds/token; window mismatch and endogeneity need separate examination, not an automatic bias verdict.

The regression's descriptive usefulness, direct per-layer microbench evidence, and causal-identifiability limitations can all hold simultaneously. T3/T5 should preserve these distinctions. T6 owns the stock-versus-flow mincore question; no claim about that is made here.

## Verification

Assertions pass for12 rows,6 prompt groups, all counts256, and every admissible two-stage gain in[1,2]. Source hashes, all held-out predictions, fit coefficients, rejected bootstrap counts, and seeded percentiles are retained. No server/network/model calls, hyperparameter grids, or other-track edits.
