"""Materialize learned order on a train-selected real expert subset.
Warm local read+CRC integration test, NOT full-model/physical-SSD proof.
"""
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time
import zlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from moefit.expert_pack import build_pack, ExpertPack
from moefit.reorder_expert_pack import reorder_pack
from moefit.pack_batch import read_pack_batch


def consume(pack,batches):
    crc=calls=issued=0
    for batch in batches:
        groups,stats=read_pack_batch(pack,batch)
        calls+=stats['planned_reads'];issued+=stats['issued_bytes']
        for group in groups:
            for view in group.values():crc=zlib.crc32(view,crc)
    return crc,calls,issued


def main():
    layout=json.loads((ROOT/'results/t8_learned_layout.json').read_text())
    with np.load(Path.home()/'.hermes/cache/scratch/xlayer.npz') as z:
        keys=sorted(k for k in z.files if k.endswith('|idx'))
        arrays=[z[k] for k in keys]
    assert hashlib.sha256(b''.join(a.tobytes() for a in arrays)).hexdigest()==layout['route_sha256']
    train=np.concatenate(arrays[:4])[:,0];test=np.concatenate(arrays[4:])[:,0]
    freq=Counter(map(int,train.ravel()))
    selected=sorted(sorted(freq,key=lambda e:(-freq[e],e))[:32]);chosen=set(selected)
    order=[e for e in layout['orders'][0] if e in chosen]
    # First32 heldout positions, fixed before inspecting throughput.
    heldout=[[int(e) for e in row if int(e) in chosen] for row in test[:32]]
    shuffled=selected.copy();random.Random(823).shuffle(shuffled)
    patterns={'heldout_subset':heldout,
              'sequential_ids':[selected[i:i+4] for i in range(0,32,4)],
              'random_ids':[shuffled[i:i+4] for i in range(0,32,4)]}
    with open('/tmp/moefit-lab/state/local_ssd_bench.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'],prefix='t8_materialized_') as tmp:
            p,q=Path(tmp)/'identity',Path(tmp)/'learned'
            m=build_pack('/Users/yb/.lmstudio/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp',p,selected)
            reorder_pack(p,q,order)
            a,b=ExpertPack(p),ExpertPack(q)
            try:
                equalities=0
                for eid in selected:
                    x,y=a.read(eid,True),b.read(eid,True)
                    for key in x:assert x[key]==y[key];equalities+=1
                rows={};counters={};references={}
                for name,batches in patterns.items():
                    references[name]=consume(a,batches)[0]
                    assert consume(b,batches)[0]==references[name]
                arms=[(name,label,pack) for name in patterns for label,pack in [('identity',a),('learned',b)]]
                for trial in range(6):
                    for name,label,pack in arms[trial:]+arms[:trial]:
                        t=time.perf_counter();crc,calls,issued=consume(pack,patterns[name]);dt=time.perf_counter()-t
                        assert crc==references[name]
                        key=name+'_'+label
                        rows.setdefault(key,[]).append(dt)
                        counters[key]=dict(read_calls=calls,issued_bytes=issued)
                rates={k:sum(map(len,patterns[k.rsplit('_',1)[0]]))*m['payload_bytes']/statistics.median(v)/1e9 for k,v in rows.items()}
                result=dict(scope='MEASURED LOCAL WARM file read+CRC on train-selected32-expert integration fixture. NOT physical SSD, full512 layout, or decode.',
                            selected=selected,learned_order=order,component_equalities=equalities,
                            route_sha256=layout['route_sha256'],heldout_positions=32,
                            heldout_selected_demands=sum(map(len,heldout)),heldout_total_demands=int(test[:32].size),
                            seconds=rows,useful_GBps=rates,counters=counters,
                            ratios={name:rates[name+'_learned']/rates[name+'_identity'] for name in patterns},
                            caveats='Subset compaction changes adjacency versus full512 layout. Heldout routes filtered to train-selected experts, not cache misses. Sequential control means ascending expert IDs, not sequential offsets in learned layout. Physical device attribution unavailable.')
            finally:a.close();b.close()
    (ROOT/'results/t8_materialized_layout.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('seconds','selected','learned_order')},indent=2))

if __name__=='__main__':main()
