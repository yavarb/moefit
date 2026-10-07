"""M3 assist: fixed-budget exclusive LRU victim hierarchy; no silicon claims.

Plain fully-associative LRU main + FIFO eviction ring, promoted on ring hit.
At equal total capacity this is exactly one LRU stack, not extra capacity.
Batch semantics refresh all pre-existing requested entries before new installs.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import random


class VictimRing:
    def __init__(self, main_capacity=127, victim_capacity=16):
        if main_capacity < 1 or victim_capacity < 1:
            raise ValueError('positive capacities required')
        self.main_capacity = main_capacity
        self.victim_capacity = victim_capacity
        self.main = OrderedDict()
        self.victim = OrderedDict()
        self.counts = dict(main_hit=0, victim_hit=0, ssd_miss=0, demotions=0,
                           victim_discards=0)

    def access(self, expert):
        if expert in self.main:
            self.main.move_to_end(expert)
            self.counts['main_hit'] += 1
            return False
        ring_hit = expert in self.victim
        self.counts['victim_hit' if ring_hit else 'ssd_miss'] += 1
        if ring_hit:
            del self.victim[expert]
        if len(self.main) == self.main_capacity:
            old, _ = self.main.popitem(last=False)
            self.victim[old] = None
            self.counts['demotions'] += 1
            if len(self.victim) > self.victim_capacity:
                self.victim.popitem(last=False)
                self.counts['victim_discards'] += 1
        self.main[expert] = None
        return not ring_hit

    def stack(self):
        return list(self.victim) + list(self.main)

    def check(self):
        assert not self.main.keys() & self.victim.keys()
        assert len(self.main) <= self.main_capacity
        assert len(self.victim) <= self.victim_capacity


def lru_access(cache, expert, capacity):
    miss = expert not in cache
    if miss and len(cache) == capacity:
        cache.popitem(last=False)
    cache[expert] = None
    cache.move_to_end(expert)
    return miss


def batch(ring, reference, ids):
    ids = list(dict.fromkeys(map(int, ids)))
    if len(ids) > ring.main_capacity:
        raise ValueError('route exceeds main capacity')
    resident = set(ring.stack())
    hit_ids = [e for e in ids if e in resident]
    miss_ids = [e for e in ids if e not in resident]
    for e in hit_ids + miss_ids:
        observed = ring.access(e)
        expected = lru_access(reference, e, ring.main_capacity + ring.victim_capacity)
        assert observed == expected
    ring.check()
    assert ring.stack() == list(reference)
    assert set(ids) <= ring.main.keys(), 'current-call demand evicted before gather'
    return len(miss_ids)


def self_test():
    rng = random.Random(634)
    ring = VictimRing(7, 3)
    ref = OrderedDict()
    for _ in range(10000):
        e = rng.randrange(40)
        assert ring.access(e) == lru_access(ref, e, 10)
        ring.check()
        assert ring.stack() == list(ref)
    for _ in range(2000):
        batch(ring, ref, [rng.randrange(40) for _ in range(7)])
    # A cyclic working set fits total capacity, but not the main cache.
    ring = VictimRing(7, 3)
    ref = OrderedDict()
    for e in list(range(10)) * 20:
        batch(ring, ref, [e])
    assert ring.counts['ssd_miss'] == 10
    assert ring.counts['victim_hit'] == 190
    assert ring.counts['main_hit'] == 0
    return dict(single_access_equalities=10000, batch_equalities=2000,
                cyclic_references=200, cyclic_ssd_misses=10,
                cyclic_victim_hits=190)


def replay(array):
    rings = [VictimRing() for _ in range(array.shape[1])]
    refs = [OrderedDict() for _ in rings]
    main_only = [OrderedDict() for _ in rings]
    main_only_misses = 0
    misses = 0
    for route in array:
        for layer, ids in enumerate(route):
            misses += batch(rings[layer], refs[layer], ids)
            ids = list(dict.fromkeys(map(int, ids)))
            cache = main_only[layer]
            hits = [e for e in ids if e in cache]
            absent = [e for e in ids if e not in cache]
            for e in hits + absent:
                main_only_misses += lru_access(cache, e, 127)
    counts = {k: sum(r.counts[k] for r in rings) for k in rings[0].counts}
    assert counts['ssd_miss'] == misses
    return dict(tokens=len(array), layer_batches=len(array)*len(rings), **counts,
                victim_ssd_misses_per_token=misses/len(array),
                equal_budget_lru143_misses_per_token=misses/len(array),
                smaller_lru127_misses_per_token=main_only_misses/len(array),
                equal_budget_miss_reduction_fraction=0.0,
                peak_payload_budget_experts_per_layer=143)


def main():
    tests = self_test()
    import numpy as np
    source = Path.home()/'.hermes/cache/scratch/xlayer.npz'
    with np.load(source, allow_pickle=False) as data:
        keys = sorted(k for k in data.files if k.endswith('|idx'))
        arrays = [data[k] for k in keys]
    assert len(keys) == 8
    assert all(a.ndim == 3 and a.shape[1:] == (48, 10) for a in arrays)
    rows = [dict(prompt=k, **replay(a)) for k, a in zip(keys, arrays)]
    total_tokens = sum(r['tokens'] for r in rows)
    misses = sum(r['ssd_miss'] for r in rows)
    result = dict(scope='OFFLINE SIMULATION on recorded real routes; no physical cache, SSD IO, or decode timing',
                  policy='Fully associative main LRU127 + exclusive FIFO victim16; promote victim hits to main MRU',
                  lifecycle='Cold reset for each prompt; all requested resident hits refreshed before installing misses',
                  accounting='143 equal-sized expert payloads total per layer, never 143+16. Metadata and transfer staging not modeled.',
                  route_array_sha256=hashlib.sha256(b''.join(a.tobytes() for a in arrays)).hexdigest(),
                  input_path=str(source), self_tests=tests, per_prompt=rows,
                  total_tokens=total_tokens, total_ssd_misses=misses,
                  misses_per_token=misses/total_tokens,
                  victim_hits=sum(r['victim_hit'] for r in rows),
                  equal_budget_lru143_miss_delta=0,
                  decision='Plain victim ring adds no SSD-miss reduction over equal-budget fully-associative LRU. No >5% promotion gate passed.',
                  limitations='Does not rule out RRIP/DRRIP, different insertion rules, compressed victims, global pooling, or avoiding install cost. Victim promotions/copies are counted events, not timed. Existing short-route dataset reused, not a new holdout.')
    output = Path(__file__).resolve().parents[1]/'results/t6_victim_ring_reference.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
