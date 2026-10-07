"""Fixed-budget exact-prefix packing benchmark; routing-only CPU, not LLM."""
import json
import runpy
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from moefit.routing_replay import RoutingReplayCache


def main():
    runpy.run_path(str(ROOT/'experiments/t7_routing_replay_bench.py'))['checks']()
    rng=np.random.default_rng(710)
    x=rng.normal(size=(512,8,64)).astype(np.float32)
    w=rng.normal(size=(8,64,256)).astype(np.float32)
    def route(t,l):
        logits=np.einsum('h,he->e',x[t,l],w[l],optimize=False)
        ids=np.argsort(-logits)[:4].astype(np.int16)
        p=np.exp(logits[ids]-logits[ids].max());p/=p.sum()
        return ids,p
    expected=[[route(t,l) for l in range(8)] for t in range(512)]
    def run(compact):
        cache=RoutingReplayCache(8*1024*1024,compact_prefixes=compact)
        rows=[]
        for rep in range(2):
            session=cache.layered_session(b'fixed-router-v1',8,deterministic=True)
            start=time.perf_counter(); hits=0; actual=[]
            for t in range(512):
                session.begin_token(t)
                for l in range(8):
                    ids,p,hit=session.route(l,lambda t=t,l=l:route(t,l))
                    hits+=hit;actual.append((ids,p))
            elapsed=time.perf_counter()-start
            for (ids,p),(ei,ep) in zip(actual,[a for row in expected for a in row]):
                assert np.array_equal(ids,ei) and np.array_equal(p,ep)
            rows.append(dict(seconds=elapsed,routing_tokens_per_s=512/elapsed,hits=hits,
                accounted_bytes=cache.bytes,entries=len(cache.entries),evictions=cache.evictions))
            assert cache.bytes<=cache.max_bytes
        return rows
    trials=[]
    for repeat in range(3):
        if repeat%2:
            compact=run(True);tuple_rows=run(False)
        else:
            tuple_rows=run(False);compact=run(True)
        assert compact[0]['hits']==0 and compact[1]['hits']==4096
        assert tuple_rows[1]['hits']==0
        trials.append(dict(tuple=tuple_rows,packed=compact))
    cache=RoutingReplayCache(compact_prefixes=True)
    payload=lambda:(np.array([[1]],dtype=np.int16),np.array([[1.]],dtype=np.float32))
    for seq in ([1,256],[256,1],[1,0,256]):
        s=cache.session(b'a',deterministic=True)
        for token in seq:s.step(token,payload)
    assert cache.session(b'a',deterministic=True).step(1,payload)[1]
    s=cache.session(b'a',deterministic=True)
    for token in (-1,2**32):
        try:s.step(token,payload);raise AssertionError('bad id accepted')
        except ValueError:pass
        assert s.prefix==b''
    # Same suffix after divergence must not alias a stored prefix.
    s=cache.session(b'a',deterministic=True)
    s.step(2,payload)
    assert not s.step(256,payload)[1]
    a=float(np.median([r['tuple'][1]['seconds'] for r in trials]))
    b=float(np.median([r['packed'][1]['seconds'] for r in trials]))
    result=dict(kind='MEASURED local routing-only CPU kernel; NOT LLM decode or SSD',
        budget_bytes=8*1024*1024,tokens=512,layers=8,trials=trials,
        repeat_tuple_tokens_per_s=512/a,repeat_packed_tokens_per_s=512/b,
        repeat_speedup=a/b,repeat_hit_rate_tuple=0.,repeat_hit_rate_packed=1.,
        caveat='Capacity effect at a fixed conservative accounted-byte budget, not RSS-equated. Packed prefixes remain quadratic total length; full bytes equality, no digest collisions. Local machine may have concurrent T1 GPU capture.')
    (ROOT/'results/t7_compact_replay.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
