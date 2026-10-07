"""Lossless streaming permutation of an immutable expert-major sidecar.

Order is explicit, not learned from held-out demand. No quantization changes.
Uses one expert stride of working payload memory; refuses existing outputs.
"""
import copy
import hashlib
import json
import os
from pathlib import Path

from moefit.expert_pack import ExpertPack, exact_pread


def reorder_pack(source, output, expert_order):
    order = list(expert_order)
    output = Path(output)
    index = Path(str(output) + '.json')
    reader = ExpertPack(source)
    try:
        m = reader.manifest
        if (any(type(e) is not int for e in order) or
                len(order) != len(reader.positions) or
                set(order) != set(reader.positions)):
            raise ValueError('Order must be a full permutation of expert IDs')
        if output.exists() or index.exists():
            raise FileExistsError('Output pack or index exists')
        manifest = copy.deepcopy(m)
        manifest['expert_ids'] = order
        created = False
        index_created = False
        try:
            with output.open('xb') as f:
                created = True
                for eid in order:
                    data = exact_pread(reader.fd, m['stride'], reader.positions[eid]*m['stride'])
                    if hashlib.sha256(memoryview(data)[:m['payload_bytes']]).hexdigest() != m['sha256'][str(eid)]:
                        raise ValueError('Source pack checksum mismatch')
                    f.write(data)
                    del data
                f.flush()
                os.fsync(f.fileno())
            with index.open('x') as f:
                index_created = True
                json.dump(manifest, f, indent=2)
                f.write('\n')
                f.flush()
                os.fsync(f.fileno())
        except BaseException:
            if index_created:
                index.unlink()
            if created:
                output.unlink()
            raise
        return manifest
    finally:
        reader.close()
