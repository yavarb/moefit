"""Tiny synthetic selective-component execution; no throughput benchmark."""
import json, os, struct, sys, tempfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from moefit.expert_pack import build_pack
from moefit.hybrid_expert_pack import HybridExpertPack


def main():
    with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as td:
        root=Path(td);header={};mapping={};payload=bytearray()
        for proj in ('gate_proj','up_proj','down_proj'):
            for part in ('weight','scales','biases'):
                key=f'language_model.model.layers.0.mlp.switch_mlp.{proj}.{part}'
                start=len(payload);payload.extend(bytes((start+i)%251 for i in range(32)))
                header[key]=dict(dtype='U32',shape=[4,2],data_offsets=[start,len(payload)]);mapping[key]='f.safetensors'
        h=json.dumps(header).encode();(root/'f.safetensors').write_bytes(struct.pack('<Q',len(h))+h+payload)
        (root/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map=mapping)))
        build_pack(root,root/'pack',[0,1],alignment=64)
        store=HybridExpertPack(root,root/'pack');names=list(header);original=os.pread
        rows={};checks=0
        try:
            for eid in (0,3):
                reference=store.read(eid)
                for label,selected in [('full',names),('gate',names[:3]),('sparse',[names[0],names[8],names[0]]),('empty',[])]:
                    calls=[]
                    def counted(fd,size,off):calls.append(size);return original(fd,size,off)
                    with patch('os.pread',counted):got=store.read_components(eid,selected)
                    assert list(got)==list(dict.fromkeys(selected))
                    for n,v in got.items():assert v==reference[n] and v.readonly;checks+=1
                    assert sum(calls)==8*len(set(selected))
                    rows[f'{eid}_{label}']=dict(calls=len(calls),bytes=sum(calls))
            assert rows['0_gate']==dict(calls=1,bytes=24)
            assert rows['3_gate']==dict(calls=3,bytes=24)
            assert rows['0_sparse']==dict(calls=2,bytes=16)
            guards=0
            for eid,ns in [(0,['bad']),(0,[names[0],None]),(-1,names),(True,names)]:
                with patch('os.pread',side_effect=AssertionError('IO on invalid input')):
                    try:store.read_components(eid,ns)
                    except ValueError:guards+=1
                    else:raise AssertionError('invalid accepted')
        finally:store.close()
    result=dict(scope='Tiny SYNTHETIC actual pread counters/byte comparisons, NOT bandwidth or decode.',rows=rows,component_checks=checks,zero_io_guards=guards,
                caveat='Only caller-requested components fetched. Full inference still needs all required projections; split reads may increase calls if all components eventually consumed.')
    (ROOT/'results/t8_component_reads.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
