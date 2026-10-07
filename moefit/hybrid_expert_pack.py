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

    def close(self):
        if not self.closed:
            for fd in self.fds.values():os.close(fd)
            self.pack.close(); self.closed=True
