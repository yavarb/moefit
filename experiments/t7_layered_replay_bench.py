"""End-to-end constructed causal MoE reference, CPU only, NOT real LLM tps."""
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from moefit.routing_replay import RoutingReplayCache


def main():
    rng=np.random.default_rng(78)
    L,H,E,K=8,64,256,4
    emb=rng.normal(0,.1,(512,H)).astype(np.float32)
    router=rng.normal(0,.1,(L,H,E)).astype(np.float32)
    expert=rng.normal(0,.08,(L,E,H,H)).astype(np.float32)
    requests=[list(range(48)),list(range(48)),list(range(24))+list(range(200,224))]
    def run(use):
        cache=RoutingReplayCache(8*1024*1024)
        outputs=[]; calls=hits=expert_calls=0
        start=time.perf_counter()
        for req in requests:
            state=np.zeros((L,H),np.float32)
            session=cache.layered_session(b'constructed-causal-v1/zero-state',L,deterministic=True)
            out=[]
            for token in req:
                x=emb[token].copy()
                if use:
                    session.begin_token(token)
                for layer in range(L):
                    x=np.tanh(x + .2*state[layer])
                    def compute():
                        nonlocal calls
                        calls+=1
                        logits=np.einsum('h,he->e',x,router[layer],optimize=False)
                        ids=np.argsort(-logits)[:K].astype(np.int16)
                        w=np.exp(logits[ids]-np.max(logits[ids])); w/=w.sum()
                        return ids,w
                    if use:
                        ids,w,hit=session.route(layer,compute); hits+=hit
                    else:
                        ids,w=compute()
                    # These expensive/model-state operations always execute.
                    vals=np.einsum('h,khj->kj',x,expert[layer,ids],optimize=False)
                    x=np.tanh(x+np.sum(vals*w[:,None],axis=0))
                    state[layer]=x
                    expert_calls+=1
                out.append(x.copy())
            outputs.append(np.array(out))
        return outputs,dict(seconds=time.perf_counter()-start,router_calls=calls,
                            route_hits=hits,expert_layer_calls=expert_calls,cache_bytes=cache.bytes)
    # Warm kernels, then alternate baseline/replay order in five paired runs.
    run(False); run(True)
    trials=[]
    for r in range(5):
        if r%2:
            b=run(True); a=run(False)
        else:
            a=run(False); b=run(True)
        for expected,actual in zip(a[0],b[0]):
            assert np.array_equal(expected,actual)
        assert a[1]['router_calls']==1152 and b[1]['router_calls']==576
        assert b[1]['route_hits']==576
        assert a[1]['expert_layer_calls']==b[1]['expert_layer_calls']==1152
        trials.append(dict(baseline=a[1],replay=b[1]))
    # Ordered-layer guard and cross-token transaction guard.
    s=RoutingReplayCache().layered_session(b'test',2,deterministic=True)
    s.begin_token(1)
    for bad in (lambda:s.begin_token(2),lambda:s.route(1,lambda:([0],[1.]))):
        try: bad(); raise AssertionError('missing guard')
        except RuntimeError: pass
    s.route(0,lambda:(np.array([0]),np.array([1.])))
    s.route(1,lambda:(np.array([1]),np.array([1.])))
    s.begin_token(2)
    base=float(np.median([t['baseline']['seconds'] for t in trials]))
    replay=float(np.median([t['replay']['seconds'] for t in trials]))
    result=dict(kind='MEASURED local constructed causal CPU MoE, not production LLM decode',
                geometry=dict(layers=L,hidden=H,experts=E,top_k=K),tokens=144,
                trials=trials,hit_rate=.5,output_bit_exact=True,
                baseline_median_tokens_per_s=144/base,replay_median_tokens_per_s=144/replay,
                speedup=base/replay,
                caveat='Synthetic model with recurrent context, no attention/KV/SSD. Router reuse only; all expert layers and context updates executed. Not wired to oMLX.')
    (ROOT/'results/t7_layered_replay.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
