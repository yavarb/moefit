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

    def read_components(self, eid, names):
        """Fetch only explicit components, without dequantization or padding.

        Duplicate names collapse in first-occurrence order. Adjacent packed
        components coalesce with zero gap overread. No full-expert checksum is
        possible on partial reads; immutable matching source is required.
        """
        from moefit.coalesced_read import read_batch
        if self.closed: raise ValueError('Store is closed')
        if type(eid) is not int or not 0 <= eid < self.parts[0]['experts']:
            raise ValueError('Invalid expert ID')
        names = list(names)
        if any(type(n) is not str for n in names):
            raise ValueError('Component names must be strings')
        names = list(dict.fromkeys(names))
        parts = {c['name']:c for c in self.parts}
        if any(n not in parts for n in names):
            raise ValueError('Unknown component')
        if eid in self.pack.positions:
            packed = {c['name']:c for c in self.pack.manifest['components']}
            base = self.pack.positions[eid]*self.pack.manifest['stride']
            views, _ = read_batch(self.pack.fd,
                [(base+packed[n]['pack_offset'],packed[n]['size']) for n in names],
                max_gap=0,max_span=self.pack.manifest['stride'],max_amplification=1)
            return dict(zip(names,views))
        return {n:memoryview(exact_pread(self.fds[parts[n]['file']],parts[n]['size'],
                    parts[n]['offset']+eid*parts[n]['size'])) for n in names}

    def read_components_many(self, ids, names, *, max_experts_per_span=4):
        """Selected components for a batch, deduplicated and grouped by file.

        All inputs validated before IO. Zero gap overread; span bytes bounded
        by max_experts_per_span * stride. No persistent cache or checksum on
        partial payloads. Duplicate results have independent dictionaries and
        share immutable backing bytes. Entire batch is retained in memory.
        """
        from moefit.coalesced_read import read_batch
        if self.closed: raise ValueError('Store is closed')
        ids, names = list(ids), list(names)
        if any(type(e) is not int or not 0 <= e < self.parts[0]['experts'] for e in ids):
            raise ValueError('Invalid expert ID')
        if type(max_experts_per_span) is not int or max_experts_per_span < 1:
            raise ValueError('max_experts_per_span must be positive integer')
        if any(type(n) is not str for n in names):
            raise ValueError('Component names must be strings')
        names = list(dict.fromkeys(names))
        parts = {c['name']:c for c in self.parts}
        if any(n not in parts for n in names):
            raise ValueError('Unknown component')
        packed = {c['name']:c for c in self.pack.manifest['components']}
        stride = self.pack.manifest['stride']
        groups = {}; found = {e:{} for e in ids}
        for eid in found:
            for name in names:
                if eid in self.pack.positions:
                    c = packed[name]; fd = self.pack.fd
                    off = self.pack.positions[eid]*stride + c['pack_offset']
                else:
                    c = parts[name]; fd = self.fds[c['file']]
                    off = c['offset'] + eid*c['size']
                groups.setdefault(fd, []).append((eid, name, off, c['size']))
        for fd, requests in groups.items():
            views, _ = read_batch(fd, [(r[2], r[3]) for r in requests],
                max_gap=0, max_span=stride*max_experts_per_span,
                max_amplification=1)
            for (eid, name, _, _), view in zip(requests, views):
                found[eid][name] = view
        return [{n:found[e][n] for n in names} for e in ids]

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
