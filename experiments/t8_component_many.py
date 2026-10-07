"""Tiny executed selective-batch IO counts; not a throughput benchmark."""
import json
import os
from pathlib import Path
import random
import struct
import sys
import tempfile
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moefit.expert_pack import build_pack
from moefit.hybrid_expert_pack import HybridExpertPack


def main():
    rng = random.Random(28)
    checks = 0
    guards = 0
    rows = {}
    with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as td:
        root = Path(td)
        mapping = {}
        names = []
        for p, proj in enumerate(('gate_proj', 'up_proj', 'down_proj')):
            header = {}; payload = bytearray()
            for part in ('weight', 'scales', 'biases'):
                name = f'language_model.model.layers.0.mlp.switch_mlp.{proj}.{part}'
                names.append(name)
                start = len(payload)
                payload.extend(bytes((p*61+start+i)%251 for i in range(64)))
                header[name] = dict(dtype='U32', shape=[8,2], data_offsets=[start,len(payload)])
                mapping[name] = f'{proj}.safetensors'
            h = json.dumps(header).encode()
            (root/f'{proj}.safetensors').write_bytes(struct.pack('<Q',len(h))+h+payload)
        (root/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map=mapping)))
        build_pack(root, root/'pack', [1,0], alignment=64)
        store = HybridExpertPack(root, root/'pack')
        original = os.pread
        def counted_call(fn):
            calls = []
            def counted(fd, size, off):
                calls.append((fd,size,off))
                return original(fd,size,off)
            with patch('os.pread',counted):
                value = fn()
            return value, dict(calls=len(calls),bytes=sum(c[1] for c in calls)), calls
        try:
            ids = [0,1,2,3,2,0]
            selected = names[:3]
            baseline, rows['individual'], _ = counted_call(lambda:[store.read_components(e,selected) for e in ids])
            got, rows['batch'], _ = counted_call(lambda:store.read_components_many(ids,selected))
            assert rows['individual'] == dict(calls=12,bytes=144)
            assert rows['batch'] == dict(calls=5,bytes=96)
            assert got == baseline
            assert got[0] is not got[-1]
            assert got[0][names[0]].obj is got[-1][names[0]].obj
            for _ in range(100):
                ids = [rng.randrange(8) for _ in range(rng.randrange(16))]
                selected = [rng.choice(names) for _ in range(rng.randrange(14))]
                got, counts, calls = counted_call(lambda:store.read_components_many(ids,selected,max_experts_per_span=1))
                assert counts['bytes'] == len(set(ids))*len(set(selected))*8
                assert all(c[1] <= store.pack.manifest['stride'] for c in calls)
                for eid, out in zip(ids,got):
                    reference = store.read(eid)
                    assert list(out) == list(dict.fromkeys(selected))
                    for n, v in out.items():
                        assert v == reference[n] and v.readonly
                        checks += 1
            for ids, selected, bound in [([0,-1],names,4),([True],names,4),([0],[None],4),([0],['bad'],4),([0],names,0),([0],names,True)]:
                with patch('os.pread',side_effect=AssertionError('invalid input IO')):
                    try: store.read_components_many(ids,selected,max_experts_per_span=bound)
                    except ValueError: guards += 1
                    else: raise AssertionError('invalid accepted')
            for ids, selected in [([],names),([0,2,2],[])]:
                out, counts, _ = counted_call(lambda:store.read_components_many(ids,selected))
                assert counts == dict(calls=0,bytes=0)
                assert len(out) == len(ids)
            saved = store.read_components_many([0,3],names)
        finally:
            store.close()
        assert all(v.readonly and len(v)==8 for d in saved for v in d.values())
        try: store.read_components_many([],[])
        except ValueError: guards += 1
        else: raise AssertionError('closed accepted')
    result = dict(scope='Tiny SYNTHETIC actual pread counters, NOT SSD bandwidth/decode.',rows=rows,
                  random_batches=100,component_checks=checks,guards=guards,
                  exact_selected_bytes=True,span_bound=True,shards=3,
                  caveat='Whole batch retained; metadata not bounded. Only caller-selected components, no partial checksum.')
    (ROOT/'results/t8_component_many.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__': main()
