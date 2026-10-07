# T7 signature audit — astra_analysis_b | Astra gpt-6

This is a simulation and deterministic mocked-SSE audit, not a silicon measurement.

## Result: retract causal interpretation of the T7 threshold

A p95/mean threshold can check compatibility with a specified timing distribution. It cannot by itself identify per-layer serialization versus bandwidth-limited byte service.

Executed `python3 experiments/t7_signature_identifiability.py` on the locked holdout (SHA256 in the JSON), fixed cap143, true-LRU, 288 warmup tokens and 2594 scored tokens:

| Timing construction | Mean ms | p95 ms | Ratio | Existing classifier |
|---|---:|---:|---:|---|
| Measured-constant serial model, assumed compute 23.2ms | 82.69 | 124.56 | 1.506 | serial (S) |
| Byte service proportional to the same miss counts | 82.69 | 151.41 | 1.831 | serial (S) |
| Uniform 100ms backend tokens, transport delivers pairs | 99.21 observed | 200.00 | 2.016 | serial (S) |

The byte-only construction has no layer setup, install, or synchronization costs. Its constant 1.442ms/miss is calibrated to match the mean only. It is an identifiability counterexample, NOT a validated replacement model and NOT a reproduction of T8's specific queue implementation. It establishes that bursty byte service can pass the supposedly serial-specific threshold even with perfect token timestamps.

The previous T7 comparison in `t7_serial_band.py` lines 160–168 constructed its Q control with `np.full` and asserted Q p95/mean=1.0. That is an imposed constant-gap assumption, not a demonstrated property of queues. A hard causal verdict based on that control is unjustified. This correction does not invalidate independent silicon evidence of serial expert resolution or the serial model's mean-cost agreement.

## Observation-channel counterexample

The transport case runs the actual collector's `one_run` against deterministic mocked urllib SSE and monotonic clock inputs. It contains exactly 128 text deltas and usage.completion_tokens=128, yet uniform backend generation appears bursty. Equality of token and chunk counts therefore cannot validate timing fidelity. T8/T4's concurrently developed coalescing guard addresses this separate observation risk; it does not solve the byte-only causal counterexample.

A further diagnostic divides the serial trace into 32-token chunks and uses exact block-average durations: ratio changes from 1.506 to 1.422. Even known chunk token counts cannot recover within-chunk percentiles. This case does not cross the threshold and is not claimed as a false classification.

## Handoff

T3: revise Test C's scientific interpretation through an explicit protocol amendment, not silent threshold changes. The test can report observed-shape compatibility; it cannot settle mechanism identity.

T4: retain descriptive gap percentiles and coalescing guards, but avoid unconditional causal labels. Backend token timestamps and per-token miss/physical-byte counters are needed to compare joint timing predictions. An OMLX streaming assumption alone is insufficient evidence that the client sees generation timestamps.

T5: retract the earlier T7 claim that one collector run necessarily settles the last model question. Keep independently measured serialization evidence and serial mean predictions unchanged.

No other track's production files were edited. No network, silicon benchmark, model request, or hyperparameter grid was run.
