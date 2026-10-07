"""Fixed-workload exact read-plan acceleration; CPU metadata only."""
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from moefit.routing_replay import RoutingReplayCache


def reference(s,resident,size,budget):
    out=[];used=0
    for l in range(s.layers):
        entry=s.cache.entries.get(((s.namespace,s.layers,l),s.prefix))
        if entry is None:continue
        seen=set()
        for raw in entry[0].indices[0]:
            e=int(raw)
            if e in seen or e in resident.get(l,()):continue
            seen.add(e);n=size(l,e)
            if used+n<=budget:out.append((l,e,n));used+=n
    return tuple(out)


def main():
    z=np.load(ROOT/'results/traces/holdout.npz')
    name=next(k.split('|')[0] for k in z.files if k.endswith('|ids'))
    n=z[name+'|L0_idx'].shape[0]
    cache=RoutingReplayCache(compact_prefixes=True)
    s=cache.layered_session(b'plan-bench',48,deterministic=True)
    prefixes=[]
    for t,token in enumerate(z[name+'|ids'][:n]):
        s.begin_token(token);prefixes.append(s.prefix)
        for l in range(48):s.route(l,lambda t=t,l=l:(z[f'{name}|L{l}_idx'][t],z[f'{name}|L{l}_score'][t]))
    residents={l:set(range(143)) for l in range(48)}
    fixed=lambda l,e:2764800
    budget=52*2764800
    sessions=[]
    for prefix in prefixes:
        s=cache.layered_session(b'plan-bench',48,deterministic=True)
        s.prefix=prefix;s.next_layer=0;sessions.append(s)
        assert s.plan_reads(residents,fixed,budget)==reference(s,residents,fixed,budget)
        # Unequal payloads: oversized candidate must not stop later small fit.
        sizes=lambda l,e:1+(l*7+e)%19
        assert s.plan_reads(residents,sizes,137)==reference(s,residents,sizes,137)
        assert s.plan_reads(residents,sizes,0)==()
    rows=[]
    for trial in range(5):
        row={}
        for mode in (('reference','fast') if trial%2==0 else ('fast','reference')):
            calls=[0]
            def size(l,e):calls[0]+=1;return fixed(l,e)
            start=time.perf_counter();total=0
            for s in sessions:
                p=reference(s,residents,size,budget) if mode=='reference' else s.plan_reads(residents,size,budget)
                total+=len(p)
            row[mode]=dict(seconds=time.perf_counter()-start,size_callback_calls=calls[0],planned=total)
        assert row['reference']['planned']==row['fast']['planned']==n*52
        rows.append(row)
    a=float(np.median([r['reference']['seconds'] for r in rows]))
    b=float(np.median([r['fast']['seconds'] for r in rows]))
    result=dict(kind='MEASURED CPU metadata plan benchmark, NOT LLM/SSD',positions=n,trials=rows,
        baseline_plans_per_s=n/a,fast_plans_per_s=n/b,speedup=a/b,
        validation='143 exact uniform-plan equalities +143 nonuniform-size equalities +143 zero-budget checks',
        caveat='Short-circuit assumes positive byte sizes and pure size callback. Does not validate unused candidates after budget fills. Same route selection; no scheduling or IO change.')
    (ROOT/'results/t7_plan_saturation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
