"""Tiny executed lifecycle probe; no throughput/silicon benchmark."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from moefit.routing_replay import RoutingReplayCache


def main():
    rows=[]
    for compact in (False,True):
        c=RoutingReplayCache(compact_prefixes=compact)
        def layer():return np.array([2,5]),np.array([.3,.7])
        def whole():return np.array([[2,5]]),np.array([[.3,.7]])
        def seed(ns):
            s=c.session(ns,deterministic=True)
            saved=[]
            for t in range(3):saved.append(s.step(t,whole)[0])
            s=c.layered_session(ns,2,deterministic=True)
            for t in range(3):
                s.begin_token(t)
                for l in range(2):s.route(l,layer)
            return saved
        saved=seed(b'a');seed(b'ab')
        before=c.bytes
        surviving=[k for k in c.entries if (k[0] if isinstance(k[0],bytes) else k[0][0])==b'ab']
        survivor_bytes=sum(c.entries[k][1] for k in surviving)
        counters=(c.hits,c.misses,c.evictions)
        live=c.layered_session(b'a',2,deterministic=True);live.begin_token(0)
        assert c.invalidate_namespace(b'a')==9
        assert list(c.entries)==surviving and c.bytes==survivor_bytes
        assert counters==(c.hits,c.misses,c.evictions)
        assert c.invalidate_namespace(b'a')==0
        assert np.array_equal(saved[0].indices,[[2,5]]) and not saved[0].indices.flags.writeable
        assert live.plan_reads({},lambda l,e:1,4)==()
        for l in range(2):assert live.route(l,layer)[2] is False
        assert c.bytes==sum(v[1] for v in c.entries.values())
        hits=c.hits;seed(b'ab');assert c.hits-hits==9
        for bad in (b'',None,'a',1):
            try:c.invalidate_namespace(bad);raise AssertionError('invalid namespace accepted')
            except ValueError:pass
        assert c.invalidate_namespace(b'absent')==0
        c.clear();assert c.bytes==0 and c.invalidate_namespace(b'ab')==0
        rows.append(dict(compact=compact,removed_entries=9,survivor_entries=9,
                         bytes_before=before,bytes_after=survivor_bytes,
                         unaffected_hits=9,live_session_recompute_layers=2))
    result=dict(kind='EXECUTED tiny routing-cache lifecycle checks, no timing measurement',rows=rows,
        caveat='Invalidation is not revocation of returned arrays or live sessions; context mutation requires quiescence and a new namespace. No KV state managed.')
    (ROOT/'results/t7_namespace_invalidation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
