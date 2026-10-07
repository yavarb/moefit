"""Deterministic train-only co-demand path layout for expert-major packs.

Pass returned full ID permutation to reorder_pack. Does not change routing,
cache admission, or expert bytes. Unseen experts retain deterministic placement.
"""
from collections import Counter
from itertools import combinations


def co_demand_order(batches, experts):
    if type(experts) is not int or experts < 1:
        raise ValueError('experts must be positive integer')
    edges = Counter()
    for batch in batches:
        ids = list(batch)
        if any(type(e) is not int or not 0 <= e < experts for e in ids):
            raise ValueError('invalid expert ID')
        edges.update(combinations(sorted(set(ids)), 2))
    # Maximum-weight greedy degree<=2 acyclic graph; each component is a path.
    parent = list(range(experts)); adjacent = [[] for _ in parent]
    def root(e):
        while parent[e] != e:
            parent[e] = parent[parent[e]]; e = parent[e]
        return e
    for (a,b), weight in sorted(edges.items(), key=lambda x:(-x[1],x[0])):
        ra, rb = root(a), root(b)
        if ra == rb or len(adjacent[a]) == 2 or len(adjacent[b]) == 2:
            continue
        adjacent[a].append(b); adjacent[b].append(a); parent[ra] = rb
    result = []; seen = set()
    for start in range(experts):
        if start in seen or len(adjacent[start]) > 1: continue
        previous = None; current = start
        while current is not None:
            result.append(current); seen.add(current)
            nxt = next((e for e in adjacent[current] if e != previous), None)
            previous, current = current, nxt
    assert sorted(result) == list(range(experts))
    return result
