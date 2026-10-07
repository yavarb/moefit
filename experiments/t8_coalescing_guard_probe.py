"""Offline regression probe of T4's coalescing guard; no silicon acquisition."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moefit.metrics import detect_coalescing, silicon_record_from_measured, signature_from_silicon_record
from experiments.t8_sse_timing_probe import run_case


def main():
    paths = [ROOT / 'moefit/metrics.py', ROOT / 'experiments/collect_silicon_run.py',
             ROOT / 'results/silicon_sc36/collect_silicon_run_256x2.json']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    def run(gaps, chunk_tokens, tokens, deltas):
        return dict(decode_tps=12.5, per_token_ms=gaps, chunk_tokens=chunk_tokens,
                    tokens=tokens, streamed_deltas=deltas, count_source='usage.completion_tokens')
    cases = [
        ('known_two_tokens_each', [run([160]*128, [2]*129, 258, 129)], True),
        ('usage_count_disagrees', [run([160]*128, [1]*129, 258, 129)], True),
        ('buffered_pairs_equal_counts', [run([160, 0]*128, [1]*257, 257, 257)], True),
        ('mixed_flagged_and_clean_runs', [run([0.01]*90+[500]*10, [1]*101, 101, 101),
                                          run([80]*1000, [1]*1001, 1001, 1001)], True),
        ('clean_control', [run([80]*256, [1]*257, 257, 257)], False),
    ]
    rows = []
    for name, runs, known_coalesced in cases:
        rec = silicon_record_from_measured(dict(runs=runs), 'synthetic_NOT_silicon')
        sig = signature_from_silicon_record(rec)
        rows.append(dict(name=name, kind='synthetic_NOT_silicon',
                         known_coalesced=known_coalesced,
                         run_heuristics=[detect_coalescing(r['per_token_ms']) for r in runs],
                         normalized_coalescing=rec.get('tok_gap_coalesced'), signature=sig,
                         refused=sig['verdict'].startswith('inadmissible')))
    measured = json.loads(paths[-1].read_text())
    rec = silicon_record_from_measured(measured, str(paths[-1]))
    real = dict(kind='reanalysis_existing_measured_artifact_NO_new_run',
                counts=[dict(tokens=r.get('tokens'), deltas=r.get('streamed_deltas'),
                             declared_chunk_token_sum=sum(r.get('chunk_tokens', []))) for r in measured['runs']],
                coalescing=rec.get('tok_gap_coalesced'), signature=signature_from_silicon_record(rec))
    # Exercise the actual collector again. Its newly fixed endpoints should
    # exclude both a finish control chunk and a delayed usage-only trailer.
    control = run_case('fixed_single_token', [80]*256, [1])
    trailer = run_case('fixed_usage_trailer', [80]*256, [1], trailer_delay=5)
    assert control['collector_decode_tps'] == trailer['collector_decode_tps'] == 12.5
    assert control['collector_gap_samples'] == trailer['collector_gap_samples'] == 256
    assert control['collector_gap_ms_stats']['mean'] == 80.0
    assert len(rows) == 5
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(ROOT))] for p in paths), 'Sources changed during probe'
    out = dict(source_sha256=hashes, synthetic_cases=rows, existing_silicon=real,
               collector_fix_checks=dict(control=control, trailer=trailer, assertions_passed=3),
               escaped_known_coalescing=sum(r['known_coalesced'] and not r['refused'] for r in rows),
               caveat='A negative heuristic is not verified timing provenance. Current verdicts are transport-provisional, not definitive causal claims.')
    target = ROOT / 'results/t8_coalescing_guard_probe.json'
    target.write_text(json.dumps(out, indent=2)+'\n')
    for r in rows:
        print(r['name'], 'refused=', r['refused'], 'normalized=', r['normalized_coalescing'])
    print('existing measured:', json.dumps(real))
    print('collector endpoint/control fixes: 3 assertions PASS; known escapes:', out['escaped_known_coalescing'])


if __name__ == '__main__':
    main()
