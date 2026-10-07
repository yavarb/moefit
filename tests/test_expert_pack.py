"""Self-contained expert pack roundtrip tests; tiny synthetic safetensors."""
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from moefit.expert_pack import build_pack,ExpertPack


def main():
    with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as td:
        root=Path(td); tensors={}; payload=bytearray(); mapping={}
        for p in ('gate_proj','up_proj','down_proj'):
            for part in ('weight','scales','biases'):
                key=f'language_model.model.layers.0.mlp.switch_mlp.{p}.{part}'
                start=len(payload)
                payload.extend(bytes((start+i)%256 for i in range(32)))
                tensors[key]=dict(dtype='U32',shape=[4,2],data_offsets=[start,len(payload)])
                mapping[key]='fixture.safetensors'
        h=json.dumps(tensors).encode()
        (root/'fixture.safetensors').write_bytes(struct.pack('<Q',len(h))+h+payload)
        (root/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map=mapping)))
        target=root/'packed'
        m=build_pack(root,target,[3,1],alignment=64)
        reader=ExpertPack(target)
        assert list(reader.positions)==[3,1]
        checks=0
        for eid in [3,1]:
            data=reader.read(eid,verify=True)
            for key,t in tensors.items():
                start=t['data_offsets'][0]+eid*8
                assert data[key]==payload[start:start+8]; checks+=1
        try: reader.read(0)
        except KeyError: pass
        else: raise AssertionError('missing expert accepted')
        with target.open('r+b') as f: f.write(b'\xff')
        try: reader.read(3,verify=True)
        except ValueError: pass
        else: raise AssertionError('corruption not detected')
        reader.close()
        try: build_pack(root,root/'invalid',[1,1])
        except ValueError: pass
        else: raise AssertionError('duplicate IDs accepted')
        try: build_pack(root,target,[1])
        except FileExistsError: pass
        else: raise AssertionError('overwrote existing pack')
        assert m['stride']%64==0
        print(f'PASS {checks} component equalities, ID order, missing-ID, corruption, duplicates, no-overwrite, alignment')


if __name__=='__main__':main()
