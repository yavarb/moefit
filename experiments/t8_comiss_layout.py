"""Fixed-policy co-miss layout invention; real routes, no hardware speed claim."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from moefit.pack_miss_layout import co_miss_order, miss_batches
from moefit.pack_layout import co_demand_order
from experiments.t8_learned_layout import score


def main():
    assert list(miss_batches([[0,1],[1,2],[1,2],[0,2]],2)) == [[0,1],[2],[],[0]]
    assert list(miss_batches([[0,0],[0,1]],2)) == [[0],[1]]
    # Independent list-LRU replay check, no OrderedDict implementation reuse.
    rng=np.random.default_rng(822); batches=[list(map(int,rng.choice(32,4,replace=False))) for _ in range(200)]
    cache=[]; expected=[]
    for ids in batches:
        misses=[e for e in ids if e not in cache]
        for e in ids:
            if e in cache: cache.remove(e); cache.append(e)
        for e in misses:
            if len(cache)==12:cache.pop(0)
            cache.append(e)
        expected.append(misses)
    assert list(miss_batches(batches,12)) == expected
    with np.load(Path.home()/'.hermes/cache/scratch/xlayer.npz') as z:
        keys=sorted(k for k in z.files if k.endswith('|idx')); arrays=[z[k] for k in keys]
    assert len(keys)==8
    train=np.concatenate(arrays[:4]); test=np.concatenate(arrays[4:])
    previous=json.loads((ROOT/'results/t8_learned_layout.json').read_text())
    route_hash=hashlib.sha256(b''.join(a.tobytes() for a in arrays)).hexdigest()
    assert route_hash==previous['route_sha256']
    demand_orders=previous['orders']
    start=time.perf_counter()
    orders=[co_miss_order([list(map(int,b)) for b in train[:,l]],512,143) for l in range(48)]
    elapsed=time.perf_counter()-start
    assert all(sorted(o)==list(range(512)) for o in orders)
    baseline=score(test,demand_orders,143); candidate=score(test,orders,143)
    assert baseline['demanded_experts']==candidate['demanded_experts']==47993
    assert baseline['learned_spans']==41914
    result=dict(scope='Executed planner/cache replay on REAL recorded routes; NOT measured SSD bandwidth/decode.',
                train_prompts=keys[:4],test_prompts=keys[4:],route_sha256=route_hash,
                capacity=143,max_experts_per_span=4,build_seconds=elapsed,
                co_demand_baseline=baseline,co_miss_candidate=candidate,
                incremental_span_reduction=1-candidate['learned_spans']/baseline['learned_spans'],
                training=score(train,orders,143),orders=orders,
                assertions='200 independent cache replay comparisons; duplicate/hit-only/eviction toy cases; 48 full permutations; identical heldout miss counts; reproduced previous baseline.',
                caveats='Same previously inspected holdout reused for mechanism development; not an untouched final test. Cache empty at each train/test boundary, retained within each split. Requires physical validation. No cap sweep.')
    (ROOT/'results/t8_comiss_layout.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='orders'},indent=2))

if __name__=='__main__': main()
