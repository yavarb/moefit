"""Read-only multi-expert adapter on unchanged safetensors component extents.
Does not own descriptors/executor. Caller must keep immutable source fds open.
No model loading, dequantization, persistent caching or tensor repacking.
"""
from .coalesced_read import read_batch


def read_experts(components, fds, expert_ids, *, executor=None, lanes=4):
    ids = list(expert_ids)
    components = list(components)
    if not components or len({c['name'] for c in components}) != len(components):
        raise ValueError('need unique named components')
    if any(type(e) is not int or e < 0 or any(e >= c['experts'] for c in components) for e in ids):
        raise ValueError('invalid expert ID')
    # Group by source descriptor, never merge across shard boundaries.
    groups = {}
    for i, eid in enumerate(ids):
        for c in components:
            fd = fds[c['file']]
            group = groups.setdefault(fd, [])
            group.append((i, c['name'], c['offset']+eid*c['size'], c['size']))
    out = [{} for _ in ids]
    totals = {'planned_reads':0, 'issued_bytes':0, 'requested_bytes':0}
    for fd, group in groups.items():
        extents = [(off, n) for _, _, off, n in group]
        # Large weight slabs must remain intact; no gap amplification.
        limit = max(1024*1024, max(n for _, n in extents))
        views, stats = read_batch(fd, extents, executor=executor, lanes=lanes,
                                  max_gap=0, max_span=limit, max_amplification=1.0)
        for (index, name, _, _), view in zip(group, views):
            out[index][name] = view
        for k in totals:
            totals[k] += stats[k]
    return out, totals
