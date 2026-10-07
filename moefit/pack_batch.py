"""Pack-aware adapter to T6's bounded coalescer; no new IO scheduler.

Preserves caller order and duplicates. Immutable pack and caller-owned lifetime.
Span bound limits each allocation, not total retained batch memory.
"""
import hashlib
from moefit.coalesced_read import read_batch


def read_pack_batch(pack, ids, *, max_experts_per_span=4, verify=False):
    ids = list(ids)
    if type(max_experts_per_span) is not int or max_experts_per_span < 1:
        raise ValueError('max_experts_per_span must be positive integer')
    m = pack.manifest
    stride = m['stride']
    extents = [(pack.positions[eid]*stride, stride) for eid in ids]
    views, stats = read_batch(pack.fd, extents, max_gap=0,
                             max_span=stride*max_experts_per_span,
                             max_amplification=1)
    output = []
    for eid, view in zip(ids, views):
        if verify and hashlib.sha256(view[:m['payload_bytes']]).hexdigest() != m['sha256'][str(eid)]:
            raise ValueError('Pack checksum mismatch')
        output.append({c['name']:view[c['pack_offset']:c['pack_offset']+c['size']]
                       for c in m['components']})
    return output, stats
