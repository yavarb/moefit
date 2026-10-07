"""Matched direct-router, disabled-wrapper and replay CPU reference arms."""
import json
import time
import numpy as np
from t7_stateful_branch_bench import CausalBranch, RoutingReplayCache, ROOT


def main():
    rng=np.random.default_rng(78)
    model=(rng.normal(0,.1,(512,64)).astype(np.float32),
           rng.normal(0,.1,(8,64,256)).astype(np.float32),
           rng.normal(0,.08,(8,256,64,64)).astype(np.float32))
    prefix=list(range(48))
    suffixes=[list(range(48,80)),list(range(200,216))+list(range(64,80))]
    c=RoutingReplayCache(compact_prefixes=True);ns=b'direct-baseline-v1'
    root=CausalBranch(model,c.layered_session(ns,8,deterministic=True))
    root.run(prefix);state=root.state.copy();root.fork().run(suffixes[0])
    entries=c.entries.copy();size=c.bytes
    expected=[]
    for suffix in suffixes:
        fresh=CausalBranch(model,None)
        expected.append((fresh.run(prefix+suffix)[48:],fresh.state.copy()))
    def run(mode):
        c.entries=entries.copy();c.bytes=size
        start=time.perf_counter()
        s=None if mode=='direct' else c.resume_layered(ns,8,prefix,deterministic=mode=='replay')
        parent=CausalBranch(model,s,state);actual=[];hits=routers=experts=0
        for suffix in suffixes:
            branch=parent.fork();out=branch.run(suffix)
            actual.append((out,branch.state.copy()))
            hits+=branch.hits;routers+=branch.routers;experts+=branch.experts
        seconds=time.perf_counter()-start
        for a,b in zip(actual,expected):
            assert np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1])
        assert experts==512 and hits==(256 if mode=='replay' else 0)
        assert routers==(256 if mode=='replay' else 512)
        assert np.array_equal(parent.state,state)
        return dict(seconds=seconds,hits=hits,routers=routers,experts=experts)
    modes=['direct','disabled','replay']
    for mode in modes:run(mode)
    trials=[]
    for i in range(5):
        order=modes[i%3:]+modes[:i%3]
        trials.append({mode:run(mode) for mode in order})
    med={m:float(np.median([t[m]['seconds'] for t in trials])) for m in modes}
    result=dict(kind='MEASURED constructed causal CPU MoE, not production LLM',
        trials=trials,tokens=64,hit_rate=.5,tokens_per_s={m:64/v for m,v in med.items()},
        replay_speedup_vs_direct=med['direct']/med['replay'],
        replay_speedup_vs_disabled=med['disabled']/med['replay'],
        disabled_overhead_ratio=med['disabled']/med['direct'],
        exact_outputs_and_states=True,
        caveat='Prefill/seeding excluded. Direct arm skips cache wrappers, immutable copies, and prefix tracking. All arms clone identical recurrent state and execute every expert layer. No attention KV/SSD/GPU/server work.')
    (ROOT/'results/t7_direct_baseline.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='trials'},indent=2))

if __name__=='__main__':main()
