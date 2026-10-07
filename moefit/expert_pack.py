"""Lossless expert-major sidecars for affine quantized safetensors experts.
No dequantization or tensor-order changes. Original checkpoint is read-only.
"""
import hashlib
import json
import os
from pathlib import Path
import struct


def components(model_dir, layer=0):
    root = Path(model_dir)
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    prefix = f'language_model.model.layers.{layer}.mlp.switch_mlp.'
    result = []
    headers = {}
    for proj in ('gate_proj', 'up_proj', 'down_proj'):
        for part in ('weight', 'scales', 'biases'):
            key = prefix + proj + '.' + part
            file = root / index[key]
            if file not in headers:
                with file.open('rb') as f:
                    n = struct.unpack('<Q', f.read(8))[0]
                    headers[file] = (8+n, json.loads(f.read(n)))
            base, h = headers[file]
            t = h[key]
            lo, hi = t['data_offsets']
            count = t['shape'][0]
            if (hi-lo) % count:
                raise ValueError('Nonintegral expert slice')
            result.append(dict(name=key, file=str(file), offset=base+lo,
                               size=(hi-lo)//count, experts=count,
                               dtype=t['dtype'], shape=t['shape'][1:]))
    if len({c['experts'] for c in result}) != 1:
        raise ValueError('Inconsistent expert counts')
    return result


def exact_pread(fd, size, offset):
    data = os.pread(fd, size, offset)
    if len(data) != size:
        raise EOFError(f'Short read {len(data)} != {size}')
    return data


def build_pack(model_dir, output, ids, layer=0, alignment=16384) -> dict:
    ids = list(ids)
    c = components(model_dir, layer)
    if len(set(ids)) != len(ids) or not ids or any(type(i) is not int or i < 0 or i >= c[0]['experts'] for i in ids):
        raise ValueError('Invalid expert IDs')
    if alignment <= 0:
        raise ValueError('Invalid alignment')
    payload = sum(x['size'] for x in c)
    stride = ((payload+alignment-1)//alignment)*alignment
    offsets = []
    cursor = 0
    for x in c:
        offsets.append(dict(x, pack_offset=cursor))
        cursor += x['size']
    handles = {x['file']: os.open(x['file'], os.O_RDONLY) for x in c}
    hashes = {}
    try:
        with open(output, 'xb') as out:
            for eid in ids:
                data = b''.join(exact_pread(handles[x['file']], x['size'], x['offset']+eid*x['size']) for x in c)
                hashes[str(eid)] = hashlib.sha256(data).hexdigest()
                out.write(data)
                out.write(bytes(stride-payload))
            out.flush()
            os.fsync(out.fileno())
    finally:
        for fd in handles.values():
            os.close(fd)
    manifest = dict(version=1, layer=layer, expert_ids=ids, payload_bytes=payload,
                    stride=stride, alignment=alignment, components=offsets, sha256=hashes)
    Path(str(output)+'.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


class ExpertPack:
    def __init__(self, path):
        self.manifest = json.loads(Path(str(path)+'.json').read_text())
        self.positions = {eid:i for i,eid in enumerate(self.manifest['expert_ids'])}
        self.fd = os.open(path, os.O_RDONLY)

    def read(self, eid, verify=False):
        m = self.manifest
        data = exact_pread(self.fd, m['stride'], self.positions[eid]*m['stride'])
        if verify and hashlib.sha256(data[:m['payload_bytes']]).hexdigest() != m['sha256'][str(eid)]:
            raise ValueError('Pack checksum mismatch')
        view = memoryview(data)
        return {c['name']:view[c['pack_offset']:c['pack_offset']+c['size']] for c in m['components']}

    def read_into(self, eid, buffer, verify=False):
        """One synchronous libc pread directly into caller-owned writable storage.

        Returned views alias buffer: caller must finish all consumers before reuse.
        No hidden allocation of the expert payload. BF16 stays raw 16-bit bits.
        """
        import ctypes
        import errno
        m = self.manifest
        position = self.positions[eid]
        view = memoryview(buffer)
        if view.readonly or not view.c_contiguous:
            raise ValueError('Need contiguous writable buffer')
        view = view.cast('B')
        size = m['stride']
        if len(view) < size:
            raise ValueError('Buffer smaller than expert stride')
        if not hasattr(self, '_pread'):
            lib = ctypes.CDLL(None, use_errno=True)
            fn = lib.pread
            fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_longlong]
            fn.restype = ctypes.c_ssize_t
            self._pread = fn
        target = (ctypes.c_char * size).from_buffer(view)
        while True:
            n = self._pread(self.fd, target, size, position * size)
            if n >= 0:
                break
            error = ctypes.get_errno()
            if error != errno.EINTR:
                raise OSError(error, os.strerror(error))
        if n != size:
            raise EOFError(f'Short read {n} != {size}')
        if verify and hashlib.sha256(view[:m['payload_bytes']]).hexdigest() != m['sha256'][str(eid)]:
            raise ValueError('Pack checksum mismatch')
        return {c['name']: view[c['pack_offset']:c['pack_offset']+c['size']] for c in m['components']}

    def numpy_views(self, component_views):
        """Read-only zero-copy typed arrays; BF16 represented by uint16 bits.

        These arrays retain backing storage but are invalidated logically when a
        caller-owned buffer is reused. No dequantization or BF16 conversion.
        """
        import numpy as np
        types = {'U32': '<u4', 'BF16': '<u2', 'F16': '<f2', 'F32': '<f4'}
        arrays = {}
        for c in self.manifest['components']:
            a = np.frombuffer(component_views[c['name']], dtype=types[c['dtype']]).reshape(c['shape'])
            a.flags.writeable = False
            arrays[c['name']] = a
        return arrays

    def close(self):
        os.close(self.fd)


class MappedExpertPack(ExpertPack):
    """Read-only zero-copy pack views, backed by demand-paged file mapping.

    Warm-path optimization, not guaranteed SSD bypass. Views retain the mapping;
    close raises BufferError while exports exist. File must remain immutable.
    """
    def __init__(self, path):
        import mmap
        super().__init__(path)
        try:
            self.mapping = mmap.mmap(self.fd, 0, access=mmap.ACCESS_READ)
        except Exception:
            os.close(self.fd)
            raise
        self.closed = False

    def read(self, eid, verify=False):
        if self.closed:
            raise ValueError('Pack is closed')
        m = self.manifest
        start = self.positions[eid] * m['stride']
        if start + m['stride'] > len(self.mapping):
            raise EOFError('Truncated mapped expert')
        view = memoryview(self.mapping)[start:start+m['payload_bytes']]
        if verify and hashlib.sha256(view).hexdigest() != m['sha256'][str(eid)]:
            raise ValueError('Pack checksum mismatch')
        return {c['name']:view[c['pack_offset']:c['pack_offset']+c['size']] for c in m['components']}

    def close(self):
        if not self.closed:
            self.mapping.close()  # Refuse teardown with outstanding views.
            os.close(self.fd)
            self.closed = True
