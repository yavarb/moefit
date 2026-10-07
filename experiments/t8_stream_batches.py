"""Tiny synthetic streaming pack execution; no throughput benchmark."""
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from moefit.expert_pack import build_pack
from moefit.hybrid_expert_pack import HybridExpertPack


def main():
    with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as td:
        root=Path(td);h={};mapping={};payload=bytearray()
        for proj in ('gate_proj','up_proj','down_proj'):
            for part in ('weight','scales','biases'):
                key=f'language_model.model.layers.0.mlp.switch_mlp.{proj}.{part}'
                start=len(payload);payload.extend(bytes((i+start)%251 for i in range(64)))
                h[key]=dict(dtype='U32',shape=[8,2],data_offsets=[start,len(payload)])
                mapping[key]='fixture.safetensors'
        hb=json.dumps(h).encode();(root/'fixture.safetensors').write_bytes(struct.pack('<Q',len(hb))+hb+payload)
        (root/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map=mapping)))
        m=build_pack(root,root/'pack',[0,1,2,3],alignment=64)
        store=HybridExpertPack(root,root/'pack')
        try:
            ids=[0,0,1,4,4,5,2,3,7,0]; consumed=[];chunks=[];checks=0;peak=0
            def source():
                for e in ids:consumed.append(e);yield e
            gen=store.iter_batches(source(),max_bytes=2*m['stride'],max_requests=3)
            original=os.pread;calls=[]
            def tracked(fd,size,off):calls.append(size);return original(fd,size,off)
            with patch('os.pread',tracked):
                first=next(gen); assert len(consumed)==4
                first_calls=len(calls)
                # Paused generator has not prefetched the next chunk.
                assert len(calls)==first_calls
                chunks.append(first)
                chunks.extend(gen)
            cursor=0
            for chunk in chunks:
                batch=ids[cursor:cursor+len(chunk)];cursor+=len(chunk)
                assert len(chunk)<=3 and len(set(batch))*m['stride']<=2*m['stride']
                objects={id(v.obj):v.obj for g in chunk for v in g.values()}
                backing=sum(len(x) for x in objects.values());peak=max(peak,backing)
                assert backing<=2*m['stride']
                for e,g in zip(batch,chunk):
                    for k,v in store.read(e).items():assert g[k]==v;checks+=1
            assert cursor==len(ids)
            dup=list(store.iter_batches([0]*100,max_bytes=m['stride'],max_requests=7))
            assert all(len(c)<=7 for c in dup) and sum(map(len,dup))==100
            guards=0
            for kwargs in (dict(max_bytes=0),dict(max_bytes=True),dict(max_bytes=m['stride'],max_requests=0)):
                try:list(store.iter_batches([0],**kwargs))
                except ValueError:guards+=1
                else:raise AssertionError('invalid bound')
            assert list(store.iter_batches([],max_bytes=m['stride']))==[]
            result=dict(scope='Tiny synthetic actual file IO and backing-buffer accounting; NOT RSS, bandwidth or decode.',
                requests=len(ids),chunks=list(map(len,chunks)),component_checks=checks,
                payload_budget=2*m['stride'],peak_chunk_backing_bytes=peak,
                actual_pread_calls=len(calls),first_chunk_pread_calls=first_calls,
                duplicate_requests=100,duplicate_chunks=len(dup),guard_cases=guards,
                caveat='Metadata excluded; consumers retaining prior chunks can exceed budget. One-ID lookahead and incremental validation; per-chunk duplicate suppression only.')
        finally:store.close()
    (ROOT/'results/t8_stream_batches.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
