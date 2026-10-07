"""Tiny synthetic IO correctness/count test, not a throughput benchmark."""
import json
import os
from pathlib import Path
import random
import struct
import sys
import tempfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from moefit.expert_pack import build_pack
from moefit.hybrid_expert_pack import HybridExpertPack


def main():
    with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'],prefix='t8_hybrid_many_') as td:
        root=Path(td); header={};mapping={};payload=bytearray()
        for proj in ('gate_proj','up_proj','down_proj'):
            for part in ('weight','scales','biases'):
                key=f'language_model.model.layers.0.mlp.switch_mlp.{proj}.{part}'
                start=len(payload);payload.extend(bytes((i+start)%251 for i in range(64)))
                header[key]=dict(dtype='U32',shape=[8,2],data_offsets=[start,len(payload)])
                mapping[key]='fixture.safetensors'
        h=json.dumps(header).encode()
        (root/'fixture.safetensors').write_bytes(struct.pack('<Q',len(h))+h+payload)
        (root/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map=mapping)))
        build_pack(root,root/'pack',[0,1,2,3],alignment=64)
        store=HybridExpertPack(root,root/'pack');original=os.pread
        def measured(fn):
            counts={'calls':0,'bytes':0}
            def counted(fd,size,offset):
                data=original(fd,size,offset);counts['calls']+=1;counts['bytes']+=len(data);return data
            with patch('os.pread',counted): out=fn()
            return out,counts
        try:
            ids=[0,4,1,4,0,5,2,5,3,0]
            ref,baseline=measured(lambda:[store.read(e) for e in ids])
            got,batched=measured(lambda:store.read_many(ids))
            checks=0
            for a,b in zip(ref,got):
                for k in a:assert a[k]==b[k] and b[k].readonly;checks+=1
            assert got[0] is not got[4]
            key=next(iter(got[0])); assert got[0][key].obj is got[4][key].obj
            got[0].pop(key);assert key in got[4]
            rng=random.Random(825)
            for _ in range(100):
                batch=[rng.randrange(8) for _ in range(rng.randrange(20))]
                out=store.read_many(batch)
                assert len(out)==len(batch)
                for e,group in zip(batch,out):
                    for k,v in store.read(e).items():assert v==group[k];checks+=1
            for bad in ([0,8],[0,-1],[0,True],[0,1.5]):
                def reject():
                    try:store.read_many(bad)
                    except ValueError:return
                    raise AssertionError('bad ID accepted')
                _,count=measured(reject);assert count['calls']==0
            empty,count=measured(lambda:store.read_many([]));assert empty==[] and count['calls']==0
            assert baseline=={'calls':42,'bytes':1056}
            assert batched=={'calls':19,'bytes':656}
            store.close()
            assert bytes(got[4][key])==bytes(ref[4][key])
            try:store.read_many([0])
            except ValueError:pass
            else:raise AssertionError('closed accepted')
        finally:store.close()
    result=dict(scope='Executed tiny SYNTHETIC file IO counters/correctness, NOT bandwidth or decode',
                baseline=baseline,read_many=batched,component_comparisons=checks,
                random_batches=100,invalid_id_zero_io_cases=4,empty_zero_io=True,
                independent_dicts_shared_immutable_payload=True,post_close_payload_valid=True)
    (ROOT/'results/t8_hybrid_many.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
