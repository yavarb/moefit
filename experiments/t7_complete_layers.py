"""Tiny synthetic executable complete-layer planner checks; no timing claims."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from moefit.routing_replay import RoutingReplayCache


def session(routes, enabled=True):
    c=RoutingReplayCache(compact_prefixes=True)
    s=c.layered_session(b'groups',len(routes),deterministic=True);s.begin_token(1)
    for l,ids in enumerate(routes):
        s.route(l,lambda ids=ids:(np.asarray(ids,dtype=np.int16),np.ones(len(ids))))
    s=c.layered_session(b'groups',len(routes),deterministic=enabled);s.begin_token(1)
    return c,s


def main():
    routes=[[0,1,2,3],[4],[5],[6]]
    c,s=session(routes)
    old=s.plan_reads({},lambda l,e:1,3)
    new=s.plan_complete_layers({},lambda l,e:1,3)
    assert old==((0,0,1),(0,1,1),(0,2,1))
    assert new==((1,4,1),(2,5,1),(3,6,1))
    rng=np.random.default_rng(91)
    for case in range(200):
        routes=[rng.integers(0,16,8).tolist() for _ in range(4)]
        sizes=rng.integers(1,9,(4,16))
        resident={l:set(rng.choice(16,6,replace=False).tolist()) for l in range(4)}
        budget=int(rng.integers(0,65))
        c,s=session(routes)
        before=(list(c.entries),c.bytes,c.hits,c.misses,c.evictions)
        plan=s.plan_complete_layers(resident,lambda l,e:sizes[l,e],budget)
        assert before==(list(c.entries),c.bytes,c.hits,c.misses,c.evictions)
        expected=[];left=budget
        for l,ids in enumerate(routes):
            group=[(l,e,int(sizes[l,e])) for e in dict.fromkeys(ids) if e not in resident[l]]
            amount=sum(v[2] for v in group)
            if amount<=left:
                expected+=group;left-=amount
        assert plan==tuple(expected)
        assert sum(p[2] for p in plan)<=budget
        for l in {p[0] for p in plan}:
            assert {p[1] for p in plan if p[0]==l}==set(routes[l])-resident[l]
    c,s=session(routes,False)
    assert s.plan_complete_layers({},lambda l,e:1,100)==()
    c,s=session(routes);c.clear()
    assert s.plan_complete_layers({},lambda l,e:1,100)==()
    c,s=session(routes)
    for bad in (-1,1.5,float('nan')):
        try:s.plan_complete_layers({},lambda l,e:1,bad);raise AssertionError('budget guard')
        except ValueError:pass
    try:s.plan_complete_layers({},lambda l,e:0,100);raise AssertionError('size guard')
    except ValueError:pass
    s.route(0,lambda:None)
    try:s.plan_complete_layers({},lambda l,e:1,100);raise AssertionError('phase guard')
    except RuntimeError:pass
    result=dict(kind='EXECUTED synthetic planner only; no SSD/LLM timing',
        toy_budget=3,partial_planner_complete_layers=0,group_planner_complete_layers=3,
        partial_plan=old,complete_plan=new,random_reference_cases=200,
        caveat='Toy is constructed to show group selection, not representative traffic. Greedy grouping is not optimal and skips large early layers; actual reads may miss deadlines or residency may change.')
    (ROOT/'results/t7_complete_layers.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
