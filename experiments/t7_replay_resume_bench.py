"""Measured metadata resume/fork benchmark; not KV restoration or LLM tps."""
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from moefit.routing_replay import RoutingReplayCache


def main():
    cache=RoutingReplayCache(16*1024*1024,compact_prefixes=True)
    ids=np.array([1,2,3,4],np.int16);weights=np.array([.1,.2,.3,.4],np.float32)
    def compute():return ids,weights
    ns=b'resume-test/model/execution/tenant'
    def advance(s,token):
        s.begin_token(token);hits=0
        for l in range(8):
            i,w,h=s.route(l,compute);hits+=h
            assert np.array_equal(i,ids) and np.array_equal(w,weights)
        return hits
    s=cache.layered_session(ns,8,deterministic=True)
    for t in range(512):advance(s,t)
    trials=[]
    for trial in range(5):
        row={}
        for mode in (('walk','resume') if trial%2==0 else ('resume','walk')):
            start=time.perf_counter()
            if mode=='walk':
                s=cache.layered_session(ns,8,deterministic=True)
                # Setup baseline: walk known prefixes/routes, no expert compute.
                for t in range(480):
                    s.begin_token(t)
                    for l in range(8):s.route(l,compute)
            else:s=cache.resume_layered(ns,8,range(480),deterministic=True)
            row[mode+'_setup_s']=time.perf_counter()-start
            start=time.perf_counter()
            hits=sum(advance(s,t) for t in range(480,512))
            row[mode+'_continuation_s']=time.perf_counter()-start
            assert hits==256
            row[mode+'_hits']=hits
        trials.append(row)
    before=(list(cache.entries),cache.hits,cache.misses,cache.bytes)
    root=cache.resume_layered(ns,8,range(480),deterministic=True)
    assert before==(list(cache.entries),cache.hits,cache.misses,cache.bytes)
    left=root.fork();right=root.fork()
    assert advance(left,480)==8 and advance(right,9999)==0
    assert root.prefix==cache.resume_layered(ns,8,range(480)).prefix
    assert advance(right,481)==0 # matching suffix cannot repair divergence
    for compact in (False,True):
        c=RoutingReplayCache(compact_prefixes=compact)
        direct=c.layered_session(b'x',1,deterministic=True)
        for token in (0,256,65535):
            direct.begin_token(token);direct.route(0,compute)
        restored=c.resume_layered(b'x',1,[0,256,65535],deterministic=True)
        assert restored.prefix==direct.prefix
        fork=restored.fork();fork.begin_token(4)
        try:fork.fork();raise AssertionError('mid-token fork')
        except RuntimeError:pass
        for bad in ([1,-1],[1,1.5]):
            try:c.resume_layered(b'x',1,bad);raise AssertionError('invalid prefix')
            except ValueError:pass
    a=float(np.median([r['walk_setup_s'] for r in trials]));b=float(np.median([r['resume_setup_s'] for r in trials]))
    result=dict(kind='MEASURED local replay metadata setup only, not LLM/KV speed',
        skipped_tokens=480,continuation_tokens=32,layers=8,trials=trials,
        setup_speedup=a/b,walk_setup_ms=a*1000,resume_setup_ms=b*1000,
        continuation_hit_rate=1.0,restored_prefixes_per_s=1/b,
        contract='Backend must independently restore matching hidden/KV state; session fork copies metadata only. Thread safety unchanged. Timing does not include actual KV restore.')
    (ROOT/'results/t7_replay_resume.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
