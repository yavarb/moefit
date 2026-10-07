"""Offline eligibility counterexamples; fabricated fixtures, NOT silicon data.

Runs the actual A/B/D scorer CLI without touching production files. Assertions
check execution/schema; findings are reported rather than asserting bugs must
remain forever. Scratch is kept under TMPDIR, never system /tmp.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / 'experiments/score_silicon_run.py'


def blob(tps, counts=(1024, 1024, 1024), max_tokens=1024) -> Dict[str, Any]:
    runs = [dict(tokens=n, decode_tps=tps, ttft_s=0.7,
                 count_source='usage.completion_tokens') for n in counts]
    return dict(kind='synthetic_fixture_NOT_silicon', host='fixture-host',
                model='fixture-model', max_tokens=max_tokens, runs=runs,
                decode_tps_median=tps, ram_gib=48,
                memory_during_run=dict(resident_experts_per_layer_approx=192,
                                       free_percent=20))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = [SCORER, ROOT / 'moefit/metrics.py']
    before = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    cases = []
    cases.append(('A_valid_length_control', 'A', blob(15.9), blob(14.5), None))
    cases.append(('A_requested1024_actual128', 'A', blob(15.9, (128,)*3), blob(14.5, (128,)*3), 'actual tokens below 1024'))
    a, b = blob(15.9, (128,)*3, 128), blob(14.5, (128,)*3, 128)
    a['tokens'] = b['tokens'] = 128
    cases.append(('A_explicit128', 'A', a, b, 'actual tokens below 1024'))
    a, b = blob(15.9), blob(14.5)
    for d in (a, b):
        d.pop('max_tokens')
        for r in d['runs']:
            r.pop('tokens')
    cases.append(('A_missing_all_counts', 'A', a, b, 'actual count unavailable'))
    cases.append(('A_mixed_lengths', 'A', blob(15.9, (1024, 128, 1024)), blob(14.5), 'one constituent run below 1024'))
    cases.append(('A_single_run', 'A', blob(15.9, (1024,)), blob(14.5, (1024,)), 'protocol asks 3 runs'))
    a, b = blob(15.9), blob(14.5)
    b['host'] = 'different-host'
    b['model'] = 'different-model'
    b['ram_gib'] = 36
    b['memory_during_run']['resident_experts_per_layer_approx'] = 143
    cases.append(('A_mismatched_pair', 'A', a, b, 'different host/model/RAM/cap'))
    a, b = blob(15.9), blob(14.5)
    for d in (a, b):
        d['memory_during_run']['free_percent'] = 5
    cases.append(('A_crowded', 'A', a, b, 'during-run free percent below 10'))
    for name, test, tps, cap in [('B_requested1024_actual128', 'B', 14.5, 180), ('D_requested1024_actual128', 'D', 15.8, 224)]:
        a = blob(tps, (128,)*3)
        a['memory_during_run']['resident_experts_per_layer_approx'] = cap
        cases.append((name, test, a, None, 'actual tokens below 1024'))
    rows = []
    scratch = os.environ.get('TMPDIR', str(Path.home() / '.hermes/cache/scratch'))
    with tempfile.TemporaryDirectory(prefix='t8_eligibility_', dir=scratch) as td:
        for name, test, left, right, invalid_reason in cases:
            p = Path(td) / 'left.json'
            p.write_text(json.dumps(left))
            args = [sys.executable, str(SCORER), '--test', test]
            if right is not None:
                q = Path(td) / 'right.json'
                q.write_text(json.dumps(right))
                args += ['--lru', str(p), '--prior', str(q)]
            else:
                args += ['--measured', str(p)]
            run = subprocess.run(args, capture_output=True, text=True, timeout=30)
            assert run.returncode == 0, (name, run.stderr)
            out = json.loads(run.stdout)
            assert out['test'] == test
            substantive = any(any(k in v for k in ['TRANSFERS', 'IN the pre-registered', 'compute term follows'])
                              for v in out.get('verdicts', []))
            rows.append(dict(name=name, input_left=left, input_right=right,
                             protocol_problem=invalid_reason, output=out,
                             substantive_conclusion=substantive,
                             invalid_fixture_gets_conclusion=bool(invalid_reason and substantive)))
    after = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    assert before == after, 'Production files changed mid-probe: rerun for a coherent snapshot'
    report = dict(kind='offline_synthetic_fixtures_NOT_silicon',
                  source_sha256=before, cases=rows,
                  total_cases=len(rows),
                  invalid_cases=sum(bool(r['protocol_problem']) for r in rows),
                  invalid_cases_with_conclusions=sum(r['invalid_fixture_gets_conclusion'] for r in rows),
                  note='No production edits. Numeric scoring rules untouched. Unknown eligibility should not be accepted as verified eligibility.')
    target = ROOT / 'results/t8_scorer_eligibility_probe.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    for r in rows:
        print(r['name'], 'conclusion=', r['substantive_conclusion'], 'warnings=', r['output'].get('warnings'))
    print(json.dumps({k: report[k] for k in ('total_cases', 'invalid_cases', 'invalid_cases_with_conclusions')}))
    print('wrote', target)


if __name__ == '__main__':
    main()
