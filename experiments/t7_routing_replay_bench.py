"""Execute T7 routing cache: real trace payload replay + local synthetic kernel.
No model server; kernel throughput is NOT LLM decode tok/s.
"""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from moefit.routing_replay import RoutingReplayCache


def checks():
    cache=RoutingReplayCache(100000)
    payload=lambda:(np.array([[1,2]],dtype=np.int16),np.array([[.2,.8]],dtype=np.float32))
    s=cache.session(b'model-A/context-A',deterministic=True)
    assert not s.step(1,payload)[1]
    value,_=s.step(2,payload)
    s=cache.session(b'model-A/context-A',deterministic=True)
    assert s.step(1,payload)[1] and s.step(2,payload)[1]
    s=cache.session(b'model-A/context-A',deterministic=True)
    assert s.step(1,payload)[1] and not s.step(3,payload)[1] and not s.step(2,payload)[1]
    assert not cache.session(b'model-B/context-A',deterministic=True).step(1,payload)[1]
    assert not cache.session(b'model-A/context-A').step(1,payload)[1]
    try:
        value.weights.setflags(write=True)
        raise AssertionError('mutable weights')
    except ValueError:
        pass
    tiny=RoutingReplayCache(500)
    s=tiny.session(b'a',deterministic=True)
    for t in range(20): s.step(t,payload)
    assert tiny.bytes<=500 and tiny.evictions>0
    failed=cache.session(b'failure',deterministic=True)
    def broken():
        raise RuntimeError('router failed')
    try:
        failed.step(7,broken)
        raise AssertionError('expected callback failure')
    except RuntimeError:
        pass
    assert failed.prefix == ()
    assert not failed.step(7,payload)[1]
    cache.clear()
    assert cache.bytes==0 and not cache.entries


def main():
    checks()
    path=ROOT/'results/traces/holdout.npz'
    z=np.load(path)
    name=next(k.split('|')[0] for k in z.files if k.endswith('|ids'))
    length=z[name+'|L0_idx'].shape[0]
    tokens=z[name+'|ids'][:length].tolist()
    idx=np.stack([z[f'{name}|L{l}_idx'] for l in range(48)],axis=1)
    scores=np.stack([z[f'{name}|L{l}_score'] for l in range(48)],axis=1)
    cache=RoutingReplayCache()
    trace_rows=[]
    for round_id in range(3):
        s=cache.session(b'trace-only:'+hashlib.sha256(path.read_bytes()).digest(),deterministic=True)
        hits=0
        start=time.perf_counter()
        for t,token in enumerate(tokens):
            value,hit=s.step(token,lambda t=t:(idx[t],scores[t]))
            assert np.array_equal(value.indices,idx[t]) and np.array_equal(value.weights,scores[t])
            hits+=hit
        elapsed=time.perf_counter()-start
        trace_rows.append(dict(pass_id=round_id,positions=length,hits=hits,elapsed_s=elapsed))
    assert trace_rows[0]['hits']==0 and trace_rows[1]['hits']==length
    # Actual locally executed routing kernel with constructed hidden vectors,
    # 48 layers x512 experts; not a model-forward or learned router claim.
    rng=np.random.default_rng(75)
    hidden=rng.standard_normal((64,48,128),dtype=np.float32)
    weights=rng.standard_normal((48,128,512),dtype=np.float32)
    def route(t):
        logits=np.einsum('lh,lhe->le',hidden[t],weights,optimize=False)
        chosen=np.argpartition(logits,-10,axis=1)[:,-10:]
        v=np.take_along_axis(logits,chosen,axis=1)
        order=np.argsort(-v,axis=1)
        chosen=np.take_along_axis(chosen,order,axis=1)
        v=np.take_along_axis(v,order,axis=1)
        p=np.exp(v-v.max(axis=1,keepdims=True)); p/=p.sum(axis=1,keepdims=True)
        return chosen.astype(np.int16),p
    requests=[list(range(64)),list(range(64)),list(range(32))+list(range(1000,1032))]
    expected=[]
    start=time.perf_counter()
    for request in requests:
        expected.append([route(t) for t in range(len(request))])
    baseline_s=time.perf_counter()-start
    kernel_cache=RoutingReplayCache()
    hits=0; exact=0; callback_calls=0
    def compute(t):
        nonlocal callback_calls
        callback_calls+=1
        return route(t)
    start=time.perf_counter()
    returned=[]
    for request in requests:
        s=kernel_cache.session(b'constructed-router-v1-context',deterministic=True)
        row=[]
        for t,token in enumerate(request):
            value,hit=s.step(token,lambda t=t:compute(t))
            hits+=hit; row.append(value)
        returned.append(row)
    cached_s=time.perf_counter()-start
    for a,b in zip(expected,returned):
        for (i,w),value in zip(a,b):
            assert np.array_equal(i,value.indices) and np.array_equal(w,value.weights)
            exact+=1
    assert hits==96 and callback_calls==96 and exact==192
    result=dict(kind='implemented routing cache; local CPU kernel measurement and trace replay, NOT LLM silicon decode',
        trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),trace_replay=trace_rows,
        real_trace_caveat='Same captured payload reinserted/retrieved; does not establish cross-run model determinism or weights-as-gating equivalence.',
        local_kernel=dict(positions=192,layers=48,experts=512,hidden_width=128,top_k=10,
            hits=hits,hit_rate=hits/192,compute_calls=callback_calls,exact_positions=exact,
            baseline_seconds=baseline_s,replay_seconds=cached_s,
            baseline_routing_positions_per_s=192/baseline_s,replay_routing_positions_per_s=192/cached_s,
            speedup=baseline_s/cached_s,cache_accounted_bytes=kernel_cache.bytes),
        deployment='Not wired into oMLX. Enable only deterministic execution with full context namespace; route hits do not replace expert compute or KV updates. Full-prefix equality; no speculative suffix reuse.',
        verdict='KEEP prototype for repeated-prefix routing; end-to-end LLM tok/s unmeasured.')
    (ROOT/'results/t7_routing_replay.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
