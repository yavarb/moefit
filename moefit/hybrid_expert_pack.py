"""Partial expert-major pack with exact read-only checkpoint fallback.

Caller must ensure pack and checkpoint originate from the same immutable model.
Structural metadata is verified here; identity of unpacked bytes is not hashed.
"""
import os
from moefit.expert_pack import ExpertPack, components, exact_pread


class HybridExpertPack:
    def __init__(self, model_dir, pack_path, layer=0):
        self.parts = components(model_dir, layer)
        self.pack = ExpertPack(pack_path)
        self.fds = {}; self.closed = False
        try:
            m = self.pack.manifest
            fields = ('name','size','dtype','shape','experts')
            if m['layer'] != layer or len(m['components']) != len(self.parts) or any(
                any(a[k] != b[k] for k in fields)
                for a,b in zip(self.parts,m['components'])):
                raise ValueError('Pack/checkpoint geometry mismatch')
            for c in self.parts:
                if c['file'] not in self.fds:
                    self.fds[c['file']] = os.open(c['file'],os.O_RDONLY)
        except BaseException:
            for fd in self.fds.values(): os.close(fd)
            self.pack.close()
            raise

    def read(self, eid):
        if self.closed: raise ValueError('Store is closed')
        if type(eid) is not int or not 0 <= eid < self.parts[0]['experts']:
            raise ValueError('Invalid expert ID')
        if eid in self.pack.positions:
            return self.pack.read(eid)
        return {c['name']:memoryview(exact_pread(self.fds[c['file']],c['size'],
                    c['offset']+eid*c['size'])) for c in self.parts}

    def read_many(self, ids, *, max_experts_per_span=4):
        """Deduplicate within this call, coalesce packed IDs, retain input order.

        No persistent cache/admission. Duplicate outputs share immutable payload
        buffers, but have independent dictionaries. All IDs validated before IO.
        """
        from moefit.pack_batch import read_pack_batch
        if self.closed: raise ValueError('Store is closed')
        ids = list(ids)
        if any(type(e) is not int or not 0 <= e < self.parts[0]['experts'] for e in ids):
            raise ValueError('Invalid expert ID')
        if type(max_experts_per_span) is not int or max_experts_per_span < 1:
            raise ValueError('max_experts_per_span must be positive integer')
        unique = list(dict.fromkeys(ids))
        packed = [e for e in unique if e in self.pack.positions]
        groups, _ = read_pack_batch(self.pack, packed,
                                    max_experts_per_span=max_experts_per_span)
        found = dict(zip(packed, groups))
        for e in unique:
            if e not in found: found[e] = self.read(e)
        return [dict(found[e]) for e in ids]

    def iter_batches(self, ids, *, max_bytes, max_requests=64):
        """Yield bounded lists in input order, no read-ahead across yields.

        Payload bound is unique-count * pack stride (conservative for fallback).
        Python metadata and consumer-retained prior outputs are excluded. Input
        is validated incrementally; a late invalid ID may follow yielded batches.
        At most one input ID is looked ahead. Deduplication is per chunk only.
        """
        if self.closed: raise ValueError('Store is closed')
        stride = self.pack.manifest['stride']
        if type(max_bytes) is not int or max_bytes < stride:
            raise ValueError('max_bytes must hold one expert stride')
        if type(max_requests) is not int or max_requests < 1:
            raise ValueError('max_requests must be positive integer')
        limit = max_bytes // stride
        pending = []; unique = set()
        for eid in ids:
            if type(eid) is not int or not 0 <= eid < self.parts[0]['experts']:
                raise ValueError('Invalid expert ID')
            if pending and (len(pending) == max_requests or
                            (eid not in unique and len(unique) == limit)):
                yield self.read_many(pending)
                pending = []; unique = set()
            pending.append(eid); unique.add(eid)
        if pending:
            yield self.read_many(pending)

    def close(self):
        if not self.closed:
            for fd in self.fds.values():os.close(fd)
            self.pack.close(); self.closed=True
