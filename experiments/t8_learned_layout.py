"""Frozen train-only layout on real routes. Span counts, NOT bandwidth/tok/s."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moefit.pack_layout import co_demand_order
from moefit.coalesced_read import plan_reads


def score(routes, orders, cache_cap=None):
    from collections import OrderedDict
    caches = [OrderedDict() for _ in orders]
    counts = [0,0]; demands = 0
    positions = [{e:i for i,e in enumerate(order)} for order in orders]
    for route in routes:
        for layer, batch in enumerate(route):
            ids = set(map(int,batch))
            if cache_cap is not None:
                cache = caches[layer]
                misses = ids - cache.keys()
                # Protect every current route: refresh hits before miss installs.
                for e in map(int,batch):
                    if e in cache: cache.move_to_end(e)
                for e in map(int,batch):
                    if e in misses:
                        if len(cache) == cache_cap: cache.popitem(last=False)
                        cache[e] = None
                ids = misses
            demands += len(ids)
            for arm, pos in enumerate(({e:e for e in ids}, positions[layer])):
                spans = plan_reads([(pos[e],1) for e in ids],max_span=4,max_gap=0,max_amplification=1)
                assert sum(s.length for s in spans) == len(ids)
                counts[arm] += len(spans)
    return dict(identity_spans=counts[0], learned_spans=counts[1], demanded_experts=demands,
                span_reduction_fraction=1-counts[1]/counts[0])


def main():
    # Exact structural unit checks, with duplicates and disconnected components.
    assert co_demand_order([[0,2],[0,2],[2,4]],6) == [0,2,4,1,3,5]
    assert co_demand_order([],6) == list(range(6))
    for bad in [[[-1]],[[6]],[[True]]]:
        try: co_demand_order(bad,6)
        except ValueError: pass
        else: raise AssertionError('bad ID accepted')
    p=Path.home()/'.hermes/cache/scratch/xlayer.npz'
    with np.load(p) as z:
        keys=sorted(k for k in z.files if k.endswith('|idx'))
        assert len(keys)==8
        arrays=[z[k] for k in keys]
    start=time.perf_counter()
    train=np.concatenate(arrays[:4]); test=np.concatenate(arrays[4:])
    orders=[co_demand_order([list(map(int,b)) for b in train[:,l]],512) for l in range(48)]
    build_s=time.perf_counter()-start
    assert all(sorted(o)==list(range(512)) for o in orders)
    rows=[dict(prompt=k,**score(a,orders)) for k,a in zip(keys[4:],arrays[4:])]
    result=dict(scope='REAL recorded route replay, all-demand and simulated LRU-miss planner counts. NOT physical SSD bandwidth or LLM decode.',
                train_prompts=keys[:4],test_prompts=keys[4:],
                route_sha256=hashlib.sha256(b''.join(a.tobytes() for a in arrays)).hexdigest(),
                experts=512,layers=48,max_experts_per_span=4,
                build_seconds=build_s,train=score(train,orders),test=score(test,orders),
                per_test_prompt=rows,orders=orders,
                heldout_lru143=score(test,orders,143),
                cache_semantics='Empty initial cache, retained across four heldout prompts; refresh all current hits before misses installed in route order. Same misses for both layouts.',
                caveats='All-demand and fixed true-LRU143 miss-only planner counts reported separately. Heldout prompts never used for fitting. No physical layout or hardware timing this cycle; reorder_pack is existing writer.')
    out=Path(__file__).resolve().parents[1]/'results/t8_learned_layout.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='orders'},indent=2))

if __name__=='__main__':main()
