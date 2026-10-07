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

    def close(self):
        os.close(self.fd)
