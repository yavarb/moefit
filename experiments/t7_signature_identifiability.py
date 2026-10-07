"""T7: falsification audit of p95/mean as a causal S-vs-Q discriminator.

SIMULATION and deterministic mocked SSE only; never a silicon measurement.
Fixed cap143 locked trace. No policy or hardware parameter sweep.
Run from repo: python3 experiments/t7_signature_identifiability.py
"""
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import collect_silicon_run as collector
import t7_serial_band as serial
from moefit.metrics import latency_stats_from_deltas, serial_signature_check


def summarize(gaps):
    stats = latency_stats_from_deltas([float(x) for x in gaps])
    return dict(stats=stats, classifier=serial_signature_check(stats))


def mocked_buffered_stream():
    """Uniform 100ms backend tokens, delivered in pairs by transport.

    Exactly 128 one-token text deltas and authoritative usage=128: even
    a chunk/token cardinality check cannot detect this timing distortion.
    The real collector parser and clock calls execute against these events.
    """
    count = 128
    payloads, clocks = [], [0.0]
    for i in range(count):
        payloads.append(b"data: " + json.dumps({"choices": [{"delta": {
            "content": "x"}}]}).encode() + b"\n")
        clocks.append(0.2 * (i // 2 + 1))
    payloads.append(b"data: " + json.dumps({"usage": {
        "completion_tokens": count, "prompt_tokens": 1}}).encode() + b"\n")
    payloads.append(b"data: [DONE]\n")
    clocks.append(12.8)

    class Response:
        def __enter__(self):
            return iter(payloads)

        def __exit__(self, *args):
            return False

    with patch.object(collector.urllib.request, "urlopen", return_value=Response()), \
            patch.object(collector.time, "monotonic", side_effect=clocks):
        run = collector.one_run("http://fixture.invalid/v1/chat/completions",
                                "fixture", "fixture", count, 900)
    assert run["tokens"] == run["streamed_deltas"] == count
    return dict(kind="mocked SSE, NOT silicon", true_backend_gap_ms=100.0,
                tokens=run["tokens"], streamed_deltas=run["streamed_deltas"],
                decode_tps=run["decode_tps"],
                observed=summarize(run["per_token_ms"]))


def main():
    td = ROOT / "results/traces_synth"
    gold = serial.load_gold(td, full=True)
    M = serial.lru_miss_matrix(gold, 143)
    warm = len(M) // 10
    M = M[warm:]
    ms = serial.tok_ms_series(M, 23.2)
    miss = M.sum(axis=1).astype(float)

    # Byte-service-only null: no per-layer setup, install, or sync.
    # Its fitted constant matches the serial mean, not its variability.
    # Q=constant gaps is an additional smoothing assumption, not a property
    # of bandwidth-limited service. This is a counterexample, not a fitted
    # replacement for the measured serial mechanism.
    byte_service = miss * (float(ms.mean()) / float(miss.mean()))
    blocks = len(ms) // 32
    coalesced = ms[:blocks * 32].reshape(blocks, 32).mean(axis=1)
    output: dict = dict(
        kind="simulated identifiability audit + mocked SSE; no silicon run",
        trace_sha256=hashlib.sha256((td / "holdout.npz").read_bytes()).hexdigest(),
        cap=143, warmup_tokens=warm, evaluated_tokens=len(M),
        serial=summarize(ms),
        byte_service_only_same_mean=summarize(byte_service),
        byte_service_ms_per_miss=float(ms.mean() / miss.mean()),
        byte_service_note="Mean calibrated null; no serial layer/install costs. Not T8's specific queue implementation.",
        serial_with_known_32_token_chunk_averaging=summarize(coalesced),
        chunk_averaging_note="Even exact chunk counts recover block averages, not within-block token gaps.",
        uniform_backend_buffered_transport=mocked_buffered_stream(),
        verdict="Threshold distinguishes shapes under assumptions, not causal S vs Q mechanisms.",
        recommendations=[
            "Keep serial mean-cost validation separate; this audit does not refute measured per-layer serialization.",
            "Retract hard Test C causal arbitration; describe S-like/Q-like observed shapes only.",
            "Require backend per-token timestamps plus miss/byte counters to test mechanism predictions.",
            "Exclude non-text/control SSE events; usage token counts do not validate chunk timing.",
            "Do not infer within-chunk token percentiles by dividing chunk gaps by token counts.",
        ],
    )
    # Assertions are the acceptance criteria for the two counterexamples.
    assert abs(np.mean(ms) - np.mean(byte_service)) < 1e-10
    assert output["byte_service_only_same_mean"]["classifier"]["p95_over_mean"] >= 1.3
    assert output["uniform_backend_buffered_transport"]["observed"]["classifier"]["p95_over_mean"] >= 1.3
    target = ROOT / "results/t7_signature_identifiability.json"
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
