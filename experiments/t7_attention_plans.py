"""Integrate exact route planning with independently checked attention demand."""
import json
from collections import OrderedDict
import numpy as np
from t7_attention_replay import Decoder,RoutingReplayCache,ROOT,L,H,E

def main():
    rng=np.random.default_rng(220)
    model=tuple(rng.normal(0,.1,shape).astype(np.float32) for shape in ((256,H),(L,3,H,H),(L,H,E),(L,E,H,H),(H,256)))
    c=RoutingReplayCache(compact_prefixes=True);ns=b'attention-plan-v1'
    seed=Decoder(model,c.layered_session(ns,L,deterministic=True));seed.run(range(24))
    entries=c.entries.copy();size=c.bytes
    rows=[]
    for mode in ('partial','complete'):
        c.entries=entries.copy();c.bytes=size
        requests=[list(range(24)),list(range(12))+list(range(100,112))]
        residents={l:OrderedDict() for l in range(L)}
        issued=eligible=covered=hits=0
        for request in requests:
            oracle_routes=[]
            oracle=Decoder(model)
            expected=oracle.run(request,after_route=lambda l,ids:oracle_routes.append((l,ids.copy())))
            d=Decoder(model,c.layered_session(ns,L,deterministic=True))
            position=-1;plan=()
            def before(s):
                nonlocal position,plan,issued
                position+=1
                f=s.plan_reads if mode=='partial' else s.plan_complete_layers
                plan=f(residents,lambda l,e:128,6*128)
                assert sum(p[2] for p in plan)<=6*128
                issued+=len(plan)
                if request[12]==100 and position>=12:assert not plan
            def after(l,ids):
                nonlocal eligible,covered
                ol,oi=oracle_routes[position*L+l]
                assert ol==l and np.array_equal(ids,oi)
                missing=set(map(int,oi))-set(residents[l])
                picked={e for ll,e,b in plan if ll==l}
                assert picked<=missing
                if mode=='complete' and picked:assert picked==missing
                eligible+=len(missing);covered+=len(picked)
                for e in map(int,ids):
                    residents[l][e]=None;residents[l].move_to_end(e)
                while len(residents[l])>16:residents[l].popitem(last=False)
            actual=d.run(request,before_layers=before,after_route=after)
            assert np.array_equal(actual,expected)
            for (k,v),(ok,ov) in zip(d.kv,oracle.kv):
                assert np.array_equal(k,ok) and np.array_equal(v,ov)
            hits+=d.hits
        assert covered==issued
        rows.append(dict(planner=mode,tokens=48,route_hits=hits,route_calls=48*L,
                         planned_requests=issued,verified_nonresident_demands=covered,
                         eligible_demands=eligible,false_requests=0))
    result=dict(kind='EXECUTED constructed attention + replay planner integration; no IO or timing',rows=rows,
        caveat='Simulated LRU16 residency, fixed6-expert issue budget, random tiny model. Metadata is validated against direct router demand, but no reads are issued, landed, or installed; no tok/s/SSD gain claimed.')
    (ROOT/'results/t7_attention_plans.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
