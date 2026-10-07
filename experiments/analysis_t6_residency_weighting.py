"""T6: resident footprint is not request-weighted byte-hit probability.
Offline counterexamples and fixed-cap143 synthetic miss-traffic weighting.
Resident sets here are analytical static sets, NOT a simulated macOS UBC.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from collections import OrderedDict

ROOT = Path(__file__).resolve().parents[1]
L, E, CAP = 48, 512, 143


def missed_ids(gold):
    counts = np.zeros((L, E), np.int64)
    caches = [OrderedDict() for _ in range(L)]
    total = 0
    for row in gold:
        for li, ids in enumerate(row):
            c = caches[li]
            for x in ids:
                e = int(x)
                if e not in c:
                    counts[li, e] += 1
                    total += 1
                c[e] = None
                c.move_to_end(e)
                if len(c) > CAP:
                    c.popitem(last=False)
    assert int(counts.sum()) == total
    return counts


def load(path):
    d = np.load(path)
    tags = sorted({k.rsplit('|', 1)[0] for k in d.files if k.endswith('|ids')})
    return np.stack([np.concatenate([d[f'{t}|L{li}_idx'] for t in tags]) for li in range(L)], axis=1)


def main():
    paths = [ROOT/'results/traces_synth/build.npz', ROOT/'results/traces_synth/holdout.npz',
             ROOT/'results/silicon_sc36/pagecache_expert_residency.json']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    measured = json.loads(paths[2].read_text())
    N, n = measured['experts'], measured['fully_cached']
    assert N == L*E and 0 < n < N
    # Same footprint and same resident IDs; only requests differ.
    residency = np.arange(n)
    cases = {}
    for label, requests in [('all_cached', np.resize(residency, 10000)),
                            ('none_cached', np.resize(np.arange(n, N), 10000))]:
        cases[label] = float(np.isin(requests, residency).mean())
    assert cases == {'all_cached': 1.0, 'none_cached': 0.0}
    build = missed_ids(load(paths[0])).ravel()
    holdout = missed_ids(load(paths[1])).ravel()
    total = int(holdout.sum())
    assert total == 167966  # independently verified T6 cap143 holdout baseline
    # Exhaustive small-domain identity: uniform resident-set selection makes
    # expected hit fraction equal footprint, regardless of request skew.
    import itertools
    weights = np.array([1, 2, 7, 11, 19])
    cover = [float(weights[list(ids)].sum()/weights.sum())
             for ids in itertools.combinations(range(5), 2)]
    assert abs(np.mean(cover) - 2/5) < 1e-12
    sets = {'build_top': np.argsort(-build, kind='stable')[:n],
            'build_bottom': np.argsort(build, kind='stable')[:n],
            'holdout_oracle_top': np.argsort(-holdout, kind='stable')[:n],
            'holdout_oracle_bottom': np.argsort(holdout, kind='stable')[:n]}
    rows = {}
    for name, ids in sets.items():
        assert len(set(ids.tolist())) == n
        rows[name] = {'resident_experts': n,
                      'resident_expert_fraction': n/N,
                      'heldout_miss_requests_covered': int(holdout[ids].sum()),
                      'heldout_miss_request_fraction': float(holdout[ids].sum()/total)}
    q = measured['cached_GB']/measured['expert_tables_GB']
    result = {'kind': 'OFFLINE logical counterexample + SIMULATED trace traffic, NOT silicon absorption measurement',
              'source_hashes': hashes, 'measured_snapshot': measured,
              'snapshot_byte_fraction': q,
              'snapshot_full_expert_fraction': n/N,
              'same_residency_counterexamples': cases,
              'static_set_trace_weighting': rows,
              'uniform_random_resident_set_expected_request_fraction': n/N,
              'holdout_miss_requests': total,
              'interpretation': [
                  'Footprint averages over addresses; absorption averages over requested bytes.',
                  'These averages coincide only with suitable uniformity/independence assumptions.',
                  'Snapshot lacks resident identities, request weights, and time alignment; its fraction is not an absorption bound.',
                  'Static sets do not model UBC eviction, overlap with expert cache, or changes during decode.',
                  'Top holdout sets use future knowledge and are analytical bounds only; build sets avoid holdout fitting.',
                  'This does not prove actual absorption is large. It invalidates bounding it by global footprint alone.',
                  'Logical-miss interval inferred by imposing absorption <= snapshot fraction is conditional, not measured.'
              ]}
    for p in paths:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(ROOT))]
    (ROOT/'results/analysis_t6_residency_weighting.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
