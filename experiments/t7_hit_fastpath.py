"""Fixed-workload CPU replay hit-path comparison, not model decode speed."""
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from moefit.routing_replay import RoutingReplayCache


def original(s,layer,compute):
    if layer != s.next_layer or layer >= s.layers:
        raise RuntimeError('layers must execute once in ascending order')
    key=((s.namespace,s.layers,int(layer)),s.prefix)
    def wrapped():
        ids,w=compute()
        return np.asarray(ids)[None,:],np.asarray(w)[None,:]
    value,hit=s.cache._resolve(key,wrapped,s.enabled)
    s.next_layer+=1
    return value.indices[0],value.weights[0],hit


def main():
    ids=np.arange(10,dtype=np.int16); w=np.full(10,.1,np.float32)
    def compute():return ids,w
    def execute(fast,repeats,budget=16*1024*1024):
        c=RoutingReplayCache(budget,compact_prefixes=True)
        # Seed cache outside timed repeat path.
        s=c.layered_session(b'fastpath',48,deterministic=True)
        for t in range(143):
            s.begin_token(t)
            for l in range(48):original(s,l,compute)
        start=time.perf_counter()
        checksum=0
        for _ in range(repeats):
            s=c.layered_session(b'fastpath',48,deterministic=True)
            for t in range(143):
                s.begin_token(t)
                for l in range(48):
                    i,v,h=s.route(l,compute) if fast else original(s,l,compute)
                    checksum+=int(i[0])+int(h)
        elapsed=time.perf_counter()-start
        return elapsed,(c.hits,c.misses,c.evictions,c.bytes,checksum),c
    trials=[]
    for trial in range(7):
        pair={}
        for fast in ([False,True] if trial%2==0 else [True,False]):
            elapsed,stats,c=execute(fast,20)
            pair['fast' if fast else 'original']=elapsed
            pair['fast_stats' if fast else 'original_stats']=stats
        assert pair['fast_stats']==pair['original_stats']
        assert pair['fast_stats'][0]==20*143*48
        trials.append(pair)
    # Capacity pressure exercises fallback/evictions, final LRU order and bytes.
    a=execute(False,2,4096);b=execute(True,2,4096)
    assert a[1]==b[1] and list(a[2].entries)==list(b[2].entries)
    for key in a[2].entries:
        av=a[2].entries[key][0];bv=b[2].entries[key][0]
        assert np.array_equal(av.indices,bv.indices) and np.array_equal(av.weights,bv.weights)
    c=RoutingReplayCache();s=c.layered_session(b'guard',1,deterministic=True)
    s.begin_token(1)
    def fail():raise ValueError('callback failure')
    try:s.route(0,fail)
    except ValueError:pass
    assert s.next_layer==0
    s.route(0,compute)
    s=c.layered_session(b'guard',1,deterministic=True);s.begin_token(1)
    i,v,hit=s.route(0,fail)
    assert hit and not i.flags.writeable and not v.flags.writeable
    a=float(np.median([p['original'] for p in trials]));b=float(np.median([p['fast'] for p in trials]))
    result=dict(kind='MEASURED CPU routing metadata only, not decode',trials=trials,
        positions_per_trial=2860,layers=48,hit_rate=1.0,
        original_positions_per_s=2860/a,fast_positions_per_s=2860/b,speedup=a/b,
        caveat='Warm full-repeat workload; tiny-cache cold fallback correctness also tested, not its speed. No namespace-prebinding retained: preserves O(1) metadata fork.')
    (ROOT/'results/t7_hit_fastpath.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='trials'},indent=2))

if __name__=='__main__':main()
