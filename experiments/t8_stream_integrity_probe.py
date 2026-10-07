"""Offline collector completion-integrity fixtures; NOT silicon measurements.
Prototype validator is probe-only; T4 owns production integration.
"""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.t8_sse_timing_probe import collector


def event(text=None, finish=None, usage=None):
    d: dict = {'choices': [{'delta': {'content': text} if text else {}, 'finish_reason': finish}]}
    if usage is not None:
        d['usage'] = {'completion_tokens': usage, 'prompt_tokens': 1}
    return b'data: ' + json.dumps(d).encode() + b'\n\n'


def integrity(events):
    """Explicit finish + terminal marker + authoritative count, no parse damage.
    Strict benchmark contract, not a universal requirement for every SSE API.
    """
    done = False
    finish = None
    usage = None
    errors = 0
    texts = 0
    after_done = 0
    for _, raw in events:
        if not raw.startswith(b'data:'):
            continue
        payload = raw[5:].strip()
        if done:
            after_done += 1
            continue
        if payload == b'[DONE]':
            done = True
            continue
        try:
            obj = json.loads(payload)
        except (ValueError, TypeError):
            errors += 1
            continue
        if obj.get('usage'):
            usage = obj['usage'].get('completion_tokens')
        for c in obj.get('choices', []):
            delta = c.get('delta') or {}
            texts += bool(delta.get('content') or delta.get('reasoning_content'))
            finish = c.get('finish_reason') or finish
    reasons = []
    if not done:
        reasons.append('missing_done')
    if finish not in ('stop', 'length'):
        reasons.append('missing_or_unsupported_finish')
    if type(usage) is not int or usage <= 0:
        reasons.append('missing_authoritative_positive_count')
    if errors:
        reasons.append('malformed_data')
    if after_done:
        reasons.append('data_after_done')
    return dict(eligible=not reasons, reasons=reasons, text_events=texts,
                usage_tokens=usage, parse_errors=errors, done=done, finish=finish)


def collect(events):
    clock = SimpleNamespace(now=0.0)
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def __iter__(self):
            for t, data in events:
                clock.now = t
                yield data
    with patch.object(collector, 'time', SimpleNamespace(monotonic=lambda: clock.now)), \
         patch.object(collector.urllib.request, 'urlopen', return_value=Response()):
        return collector.one_run('http://fixture.invalid', 'SYNTHETIC', 'unused', 1024, 1)


def main():
    source = ROOT / 'experiments/collect_silicon_run.py'
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    text = [(1+i*.08, event('x')) for i in range(16)]
    end = text[-1][0]
    trailer = [(end, event(finish='length')), (end+.01, event(usage=16)),
               (end+.02, b'data: [DONE]\n\n')]
    cases = [
        ('complete_control', text+trailer, True),
        ('abrupt_eof_after_text', text, False),
        ('usage_then_eof_no_finish', text+[(end+.01, event(usage=16))], False),
        ('finish_but_missing_usage', text+[trailer[0], trailer[-1]], False),
        ('malformed_event_before_finish', text+[(end+.001,b'data: {broken\n\n')]+trailer, False),
        ('usage_and_final_text_together', text[:-1]+[(end,event('x',finish='length',usage=16)),trailer[-1]], True),
        ('trailing_text_after_done', text+trailer+[(end+1,event('late'))], False),
    ]
    rows = []
    for name, events, expected in cases:
        guard = integrity(events)
        assert guard['eligible'] == expected, name
        run = collect(events)
        rows.append(dict(name=name, kind='synthetic_NOT_silicon',
                         prototype_integrity=guard, collector=run,
                         reports_positive_tps=bool(run.get('decode_tps') and run['decode_tps'] > 0)))
        print(name, 'eligible=',guard['eligible'], 'tps=',run['decode_tps'],
              'deltas=',run['streamed_deltas'], 'finish=',run['finish_reason'])
    assert before == hashlib.sha256(source.read_bytes()).hexdigest(), 'Collector changed during probe'
    out = dict(kind='offline_synthetic_NOT_silicon', source_sha256=before, cases=rows,
               invalid_cases_with_positive_tps=sum(not r['prototype_integrity']['eligible'] and r['reports_positive_tps'] for r in rows),
               validator_assertions_passed=len(cases),
               caveat='Integrity is necessary, not sufficient: does not validate token timing, run length, hardware or protocol eligibility.')
    (ROOT/'results/t8_stream_integrity_probe.json').write_text(json.dumps(out,indent=2)+'\n')
    print('invalid positive reports:',out['invalid_cases_with_positive_tps'])


if __name__ == '__main__':
    main()
