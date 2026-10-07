"""Execute cached-prefix read planner on recorded routing; no SSD/model IO."""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from moefit.routing_replay import RoutingReplayCache


def main():
    p=ROOT/'results/traces/holdout.npz';z=np.load(p)
    name=next(k.split('|')[0] for k in z.files if k.endswith('|ids'))
    n=z[name+'|L0_idx'].shape[0];tokens=z[name+'|ids'][:n]
    idx=np.stack([z[f'{name}|L{l}_idx'] for l in range(48)],axis=1)
    w=np.stack([z[f'{name}|L{l}_score'] for l in range(48)],axis=1)
    cache=RoutingReplayCache(16*1024*1024,compact_prefixes=True)
    ns=b'trace-replay-plan:'+hashlib.sha256(p.read_bytes()).digest()
    s=cache.layered_session(ns,48,deterministic=True)
    for t,token in enumerate(tokens):
        s.begin_token(token)
        assert s.plan_reads({},lambda l,e:2764800,52*2764800)==()
        for l in range(48):s.route(l,lambda t=t,l=l:(idx[t,l],w[t,l]))
    residents={l:set(range(143)) for l in range(48)}
    plan_count=demand_count=hit_count=0; timings=[]
    for trial in range(3):
        s=cache.layered_session(ns,48,deterministic=True)
        elapsed=0.; local_count=0
        for t,token in enumerate(tokens):
            s.begin_token(token)
            before=list(cache.entries)
            start=time.perf_counter()
            plan=s.plan_reads(residents,lambda l,e:2764800,52*2764800)
            elapsed+=time.perf_counter()-start
            assert before==list(cache.entries) # preview never pollutes replacement
            demand={(l,int(e)) for l in range(48) for e in idx[t,l] if int(e) not in residents[l]}
            selected={(l,e) for l,e,b in plan}
            assert selected<=demand and len(selected)==len(plan)
            assert sum(b for l,e,b in plan)<=52*2764800
            local_count+=len(plan)
            if trial==0: demand_count+=len(demand)
            for l in range(48):
                _,_,hit=s.route(l,lambda t=t,l=l:(idx[t,l],w[t,l]));hit_count+=hit
        timings.append(elapsed)
        if trial==0:plan_count=local_count
    assert hit_count==3*n*48
    for namespace,token in ((b'changed-model',int(tokens[0])),(ns,int(tokens[0])+1000000)):
        s=cache.layered_session(namespace,48,deterministic=True);s.begin_token(token)
        assert s.plan_reads({},lambda l,e:1,100)==()
    s=cache.layered_session(ns,48,deterministic=False);s.begin_token(tokens[0])
    assert s.plan_reads({},lambda l,e:1,100)==()
    s=cache.layered_session(ns,48,deterministic=True);s.begin_token(tokens[0])
    assert s.plan_reads({},lambda l,e:1,0)==()
    s.route(0,lambda:(idx[0,0],w[0,0]))
    try:s.plan_reads({},lambda l,e:1,100);raise AssertionError('late plan')
    except RuntimeError:pass
    result=dict(kind='MEASURED CPU metadata planner over recorded trace, not decode or physical IO',
        trace_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),positions=n,layers=48,
        cached_route_hit_rate=hit_count/(3*n*48),planned_experts=plan_count,
        eligible_demand_experts=demand_count,demand_coverage=plan_count/demand_count,
        precision=1.0,budget_experts_per_token=52,plan_seconds=timings,
        planner_tokens_per_s=n/float(np.median(timings)),accounted_cache_bytes=cache.bytes,
        assumptions='Constructed repeated request over recorded payload; static first143 resident IDs/layer (not LRU simulation). Same-prefix determinism assumed, not model-tested. Planner issues no IO; physical waste/latency unmeasured.',
        handoff='T2: begin current actual token, plan_reads before layer0, then execute route callbacks/expert work. Not prediction of unknown t+1. Consume only under backend deterministic namespace and demand-priority scheduling.')
    (ROOT/'results/t7_replay_read_plan.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
