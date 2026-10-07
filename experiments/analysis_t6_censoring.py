"""T6 fixed-cap trace audit: cold sentinels, finite tail, prompt boundaries.
No capacity/policy sweep; all quantities are replay statistics, not silicon.
Run: python3 experiments/analysis_t6_censoring.py
"""
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
from analysis_t6_traffic_headroom import sweep_layer

ROOT = Path(__file__).resolve().parents[1]
CAP = 143
L = 48


def replay(seq, cap=CAP):
    """Position-exact LRU; classify each reference by observed past/future."""
    flat = seq.ravel()
    seen = set()
    cache = OrderedDict()
    cold = np.zeros(flat.size, bool)
    miss = np.zeros(flat.size, bool)
    future = np.zeros(flat.size, bool)
    later = set()
    for i in range(flat.size - 1, -1, -1):
        e = int(flat[i])
        future[i] = e in later
        later.add(e)
    for i, x in enumerate(flat):
        e = int(x)
        cold[i] = e not in seen
        seen.add(e)
        miss[i] = e not in cache
        cache[e] = None
        cache.move_to_end(e)
        if len(cache) > cap:
            cache.popitem(last=False)
    return cold, miss, future


def percentiles(x):
    return {str(q): float(np.percentile(x, q)) for q in (50, 90, 95, 99)}


def self_test():
    # Fixed 10-slot shape required by existing sweep_layer; all IDs valid.
    rng = np.random.default_rng(673)
    seq = np.array([rng.choice(32, 10, replace=False) for _ in range(40)])
    cold, miss, future = replay(seq, 12)
    d = sweep_layer(seq[:, None, :], 0).ravel()
    assert np.array_equal(d >= seq.size, cold)
    assert np.array_equal(d >= 12, miss)
    flat = seq.ravel().tolist()
    for i, e in enumerate(flat):
        assert future[i] == (e in flat[i + 1:])
        prev = [j for j in range(i) if flat[j] == e]
        if prev:
            assert d[i] == len(set(flat[prev[-1] + 1:i]))
    assert int(cold.sum()) == len(set(flat))
    return {'references_checked': len(flat), 'checks': [
        'cold sentinel equals first reference', 'stack distance equals brute force',
        'distance miss mask equals LRU', 'future mask equals brute force']}


def main():
    tests = self_test()
    path = ROOT / 'results/traces_synth/holdout.npz'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    data = np.load(path)
    tags = sorted({k.rsplit('|', 1)[0] for k in data.files if k.endswith('|ids')})
    gold = np.stack([np.concatenate([data[f'{t}|L{li}_idx'] for t in tags])
                     for li in range(L)], axis=1)
    T, _, K = gold.shape
    all_d, all_cold, all_miss, all_future = [], [], [], []
    reset_misses = reset_cold = reset_steady_misses = reset_steady_tokens = 0
    concat_token_misses = np.zeros(T, int)
    matched_warm_mask = np.concatenate([
        np.arange(len(data[f'{tag}|L0_idx'])) >= 128 for tag in tags])
    for li in range(L):
        seq = gold[:, li, :]
        d = sweep_layer(gold, li).ravel()
        cold, miss, future = replay(seq)
        assert np.array_equal(d >= T*K, cold)
        assert np.array_equal(d >= CAP, miss)
        concat_token_misses += miss.reshape(T, K).sum(axis=1)
        all_d.append(d)
        all_cold.append(cold)
        all_miss.append(miss)
        all_future.append(future)
        for tag in tags:
            part = data[f'{tag}|L{li}_idx']
            c, m, _ = replay(part)
            reset_misses += int(m.sum())
            reset_cold += int(c.sum())
            reset_steady_misses += int(m.reshape(-1, K)[128:].sum())
            if li == 0:
                reset_steady_tokens += max(0, len(part)-128)
    d = np.concatenate(all_d)
    cold = np.concatenate(all_cold)
    miss = np.concatenate(all_miss)
    future = np.concatenate(all_future)
    counts = {
        'cold_with_observed_later_use': int((cold & future).sum()),
        'cold_without_observed_later_use': int((cold & ~future).sum()),
        'recurrent_miss_with_observed_later_use': int((~cold & miss & future).sum()),
        'recurrent_miss_without_observed_later_use': int((~cold & miss & ~future).sum()),
    }
    assert sum(counts.values()) == int(miss.sum())
    assert np.all(d[~cold] <= 511)
    result = {
        'kind': 'SIMULATED trace replay; no new silicon measurement',
        'trace': str(path.relative_to(ROOT)), 'trace_sha256': digest,
        'tag_order': tags, 'tokens': T, 'layers': L, 'top_k': K, 'cap': CAP,
        'tests': tests,
        'distance': {'cold_sentinel': T*K+1,
                     'unfiltered_percentiles': percentiles(d),
                     'finite_only_percentiles': percentiles(d[~cold]),
                     'finite_max': int(d[~cold].max()),
                     'cold_fraction': float(cold.mean())},
        'concatenated': {'misses_per_token': float(miss.sum()/T),
                         'hit_fraction': float(1-miss.mean()),
                         'miss_taxonomy_counts': counts,
                         'miss_taxonomy_fraction': {k: v/int(miss.sum()) for k,v in counts.items()},
                         'after_initial_128_misses_per_token': float(concat_token_misses[128:].mean()),
                         'matched_prompt_suffix_misses_per_token': float(concat_token_misses[matched_warm_mask].mean())},
        'reset_each_prompt': {'misses_per_token': reset_misses/T,
                              'hit_fraction': 1-reset_misses/gold.size,
                              'cold_misses_per_token': reset_cold/T,
                              'after_each_initial_128_misses_per_token': reset_steady_misses/reset_steady_tokens,
                              'remaining_tokens': reset_steady_tokens},
        'interpretation': [
            'Cold sentinel is not a reuse distance. Finite distance cannot exceed E-1=511.',
            'No observed future use is right-censored, not proof an expert is dead.',
            'A reuse-distance distribution describes LRU, not optimal eviction headroom.',
            'Prompt reset is a cache-lifecycle scenario, not a claim about the live server.',
            'Concatenation preserves lexicographic loader order; no subsampling.',
        ],
    }
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    out = ROOT / 'results/analysis_t6_censoring.json'
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
