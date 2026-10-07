"""Real mixed packed/fallback read benchmark; cache-influenced, not SSD proof."""
import fcntl
import json
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time
import zlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from moefit.expert_pack import build_pack,exact_pread
from moefit.hybrid_expert_pack import HybridExpertPack


def main():
    with open('/tmp/moefit-lab/state/local_ssd_bench.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'],prefix='t8_hybrid_') as tmp:
            model='~/models/Qwen3.8-Flash-Next-oQ4e-mtp'
            path=Path(tmp)/'pack';m=build_pack(model,path,range(16))
            store=HybridExpertPack(model,path)
            def source(eid):
                return {c['name']:memoryview(exact_pread(store.fds[c['file']],c['size'],c['offset']+eid*c['size'])) for c in store.parts}
            def consume(fn,ids):
                crc=0
                for eid in ids:
                    for view in fn(eid).values():crc=zlib.crc32(view,crc)
                return crc
            try:
                checks=0
                for eid in range(32):
                    a,b=source(eid),store.read(eid)
                    for key in a:assert a[key]==b[key] and b[key].readonly;checks+=1
                guards=0
                for eid in (-1,512,True,1.5):
                    try:store.read(eid)
                    except ValueError:guards+=1
                    else:raise AssertionError('bad ID')
                seq=list(range(32));rand=seq.copy();random.Random(824).shuffle(rand)
                arms=[('source_random',source,rand),('hybrid_random',store.read,rand),
                      ('source_sequential',source,seq),('hybrid_sequential',store.read,seq)]
                expected={name:consume(fn,ids) for name,fn,ids in arms}
                assert expected['source_random']==expected['hybrid_random']
                assert expected['source_sequential']==expected['hybrid_sequential']
                rows={name:[] for name,_,_ in arms}
                for trial in range(8):
                    for name,fn,ids in arms[trial%4:]+arms[:trial%4]:
                        t=time.perf_counter();crc=consume(fn,ids);dt=time.perf_counter()-t
                        assert crc==expected[name];rows[name].append(dt)
                rates={k:32*m['payload_bytes']/statistics.median(v)/1e9 for k,v in rows.items()}
                result=dict(scope='MEASURED LOCAL cache-influenced read+full-byte CRC. NOT physical SSD or decode.',
                    packed_experts=16,fallback_experts=16,component_equalities=checks,invalid_id_guards=guards,
                    source_read_calls=32*9,hybrid_read_calls=16+16*9,
                    seconds=rows,useful_GBps=rates,
                    ratios={k:rates['hybrid_'+k]/rates['source_'+k] for k in ('random','sequential')},
                    caveat='Same immutable checkpoint required; geometry checked, full model identity not cryptographically authenticated. Mixed 50% fixture is not measured route residency.')
            finally:store.close()
            store.close()
            try:store.read(0)
            except ValueError:pass
            else:raise AssertionError('closed store read')
            result['lifetime_guards']=2
    (ROOT/'results/t8_hybrid_pack.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='seconds'},indent=2))

if __name__=='__main__':main()
