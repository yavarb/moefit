"""Tiny replay-cache memory-pressure lifecycle test; no throughput benchmark."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from moefit.routing_replay import RoutingReplayCache


def main():
    rows=[]
    for compact in (False,True):
        c=RoutingReplayCache(compact_prefixes=compact)
        def compute():return np.array([1,2]),np.array([.25,.75])
        def replay(tokens):
            s=c.layered_session(b'resize',2,deterministic=True)
            hits=0
            for t in tokens:
                s.begin_token(t)
                for l in range(2):
                    ids,w,h=s.route(l,compute);hits+=h
                    assert np.array_equal(ids,[1,2]) and np.array_equal(w,[.25,.75])
            return hits
        assert replay(range(4))==0
        assert replay(range(1))==2 # refresh first token to MRU
        keys=list(c.entries);keep=keys[-2:]
        old_bytes=c.bytes
        target=sum(c.entries[k][1] for k in keep)
        saved=c.entries[keep[0]][0]
        assert c.resize(target)==6
        assert list(c.entries)==keep and c.bytes==target and c.evictions==6
        assert replay(range(1))==2
        snapshot=(c.max_bytes,list(c.entries),c.bytes,c.evictions)
        for bad in (-1,1.5,float('nan'),True,None):
            try:c.resize(bad);raise AssertionError('bad budget accepted')
            except ValueError:pass
            assert snapshot==(c.max_bytes,list(c.entries),c.bytes,c.evictions)
        assert c.resize(0)==2
        assert replay(range(2))==0 and not c.entries and c.bytes==0
        assert not saved.indices.flags.writeable and np.array_equal(saved.indices,[[1,2]])
        assert c.resize(np.int64(16384))==0
        assert replay(range(4))==0
        assert replay(range(4))==8
        assert c.bytes==sum(v[1] for v in c.entries.values())<=c.max_bytes
        # Whole-token routes share the same resize eviction machinery.
        c.session(b'whole',deterministic=True).step(0,lambda:([[3]],[[1.]]))
        assert c.resize(0)==9 and c.bytes==0
        rows.append(dict(compact=compact,initial_entries=8,initial_bytes=old_bytes,
                         shrunk_entries=2,shrunk_bytes=target,survivor_hits=2,
                         zero_budget_hits=0,regrown_repeat_hits=8))
    result=dict(kind='EXECUTED tiny synthetic lifecycle checks, no timing',rows=rows,
                caveat='Accounted cache capacity, not process RSS; external route arrays and session prefixes may retain memory. No backend memory-pressure hook installed.')
    (ROOT/'results/t7_resize.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
