"""Offline transport counterexamples for Test C; NOT silicon measurements.

Run: python3 experiments/t8_sse_timing_probe.py
Exercises the real collector with deterministic SSE fixtures + a virtual clock,
then the real metrics normalizer/signature classifier. No network/model loads.
Only this probe's two files are owned by T8; production fixes belong to T4.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moefit.metrics import latency_stats_from_deltas, serial_signature_check, silicon_record_from_measured

spec = importlib.util.spec_from_file_location("t8_collector", ROOT / "experiments/collect_silicon_run.py")
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load collector module")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def sse(content=None, finish=None, usage=None):
    if usage is not None:
        obj = {"choices": [], "usage": {"completion_tokens": usage, "prompt_tokens": 1}}
    else:
        obj = {"choices": [{"delta": {"content": content} if content else {}, "finish_reason": finish}]}
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def run_case(name, gaps_ms, grouping, buffered_lines=False, trailer_delay=0.0):
    # Each synthetic token has known generation time; transport may group tokens
    # into a delta or release several single-token SSE events simultaneously.
    generated = [1.0]
    for gap in gaps_ms:
        generated.append(generated[-1] + gap / 1000.0)
    n = len(generated)
    events = [(generated[0], sse("x"))]
    pos = 1
    batch_index = 0
    while pos < n:
        size = min(grouping[batch_index % len(grouping)], n - pos)
        stamp = generated[pos + size - 1]
        if buffered_lines:
            events.extend((stamp, sse("x")) for _ in range(size))
        else:
            events.append((stamp, sse("x" * size)))
        pos += size
        batch_index += 1
    events.extend([(generated[-1], sse(finish="stop")),
                   (generated[-1] + trailer_delay, sse(usage=n)),
                   (generated[-1] + trailer_delay, b"data: [DONE]\n\n")])
    clock = SimpleNamespace(now=0.0)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            for timestamp, payload in events:
                clock.now = timestamp
                yield payload

    with patch.object(collector, "time", SimpleNamespace(monotonic=lambda: clock.now)), \
         patch.object(collector.urllib.request, "urlopen", return_value=Response()):
        run = collector.one_run("http://fixture.invalid", "SYNTHETIC-FIXTURE", "unused", n, 1)
    record = silicon_record_from_measured({"runs": [run], "max_tokens": n}, "SYNTHETIC-NOT-SILICON")
    true_stats = latency_stats_from_deltas(gaps_ms)
    true_signature = serial_signature_check(true_stats)
    observed_signature = serial_signature_check(record["tok_gap_ms"])
    return dict(name=name, kind="synthetic_transport_fixture_NOT_silicon",
                generation_gap_ms_stats=true_stats,
                true_signature=true_signature,
                collector_gap_ms_stats=record["tok_gap_ms"],
                collector_signature=observed_signature,
                true_decode_tps=(n - 1) / (generated[-1] - generated[0]),
                collector_decode_tps=run["decode_tps"],
                tokens=n, streamed_deltas=run["streamed_deltas"],
                collector_gap_samples=len(run["per_token_ms"]),
                zero_token_chunks=run["chunk_tokens"].count(0),
                trailer_delay_s=trailer_delay,
                verdict_changed=true_signature["verdict"] != observed_signature["verdict"])


def main():
    rows = [
        run_case("constant_80ms_single_token_control", [80] * 256, [1]),
        run_case("constant_80ms_variable_1_3_token_deltas", [80] * 256, [1, 3]),
        run_case("constant_80ms_four_SSE_lines_buffered", [80] * 256, [4], True),
        run_case("alternating_40_120ms_single_token_control", [40, 120] * 128, [1]),
        run_case("alternating_40_120ms_pair_coalesced", [40, 120] * 128, [2]),
        run_case("constant_80ms_delayed_usage_trailer", [80] * 256, [1], trailer_delay=5),
    ]
    assert not rows[0]["verdict_changed"]
    assert rows[1]["verdict_changed"]
    assert rows[2]["verdict_changed"]
    assert not rows[3]["verdict_changed"]
    assert rows[4]["verdict_changed"]
    assert rows[5]["collector_decode_tps"] < rows[5]["true_decode_tps"]
    assert all(r["tokens"] == 257 for r in rows)
    out = dict(kind="offline_synthetic_counterexamples_NOT_silicon",
               acceptance_assertions="7 assertions passed",
               limitations="Demonstrates non-identifiability, NOT that current oMLX actually batches this way. Thresholds unchanged. No silicon inference.",
               source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                              ["experiments/collect_silicon_run.py", "moefit/metrics.py", "experiments/t8_sse_timing_probe.py"]},
               cases=rows)
    target = ROOT / "results/t8_sse_timing_probe.json"
    target.write_text(json.dumps(out, indent=2) + "\n")
    for r in rows:
        print(f"{r['name']}: true={r['true_signature']} observed={r['collector_signature']} tps={r['collector_decode_tps']} true_tps={r['true_decode_tps']:.4f}")
    print("7 assertions passed; wrote", target)


if __name__ == "__main__":
    main()
