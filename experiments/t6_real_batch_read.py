"""Real unchanged oQ4e layer read benchmark; local cache-influenced file IO."""
import fcntl
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from moefit.expert_pack import components
from moefit.coalesced_read import read_exact
from moefit.expert_batch_read import read_experts


def main():
    model=Path.home()/'.lmstudio/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp'
    c=components(model,0)
    fds={x['file']:os.open(x['file'],os.O_RDONLY) for x in c}
    rng=random.Random(613)
    batches=[rng.sample(range(512),10) for _ in range(32)]
    modes=['serial','futures','lanes']
    rows=[]
    try:
        for fd in fds.values():
            fcntl.fcntl(fd,48,1)
            fcntl.fcntl(fd,45,0)
        with ThreadPoolExecutor(4) as pool:
            def fetch(args):
                eid,x=args
                return read_exact(fds[x['file']],x['size'],x['offset']+eid*x['size'])
            def run(mode,ids):
                if mode=='lanes':
                    out,s=read_experts(c,fds,ids,executor=pool,lanes=4)
                    return [x[y['name']] for x in out for y in c],s['planned_reads']
                args=[(e,x) for e in ids for x in c]
                return list(map(fetch,args) if mode=='serial' else pool.map(fetch,args)),len(args)
            checks=0
            for ids in [batches[0],[7,7,511,0]]:
                ref,_=run('serial',ids)
                for mode in modes[1:]:
                    out,_=run(mode,ids)
                    assert all(bytes(a)==b for a,b in zip(out,ref)) and len(out)==len(ref)
                    checks+=len(ref)
            for rep in range(3):
                for mode in modes[rep:]+modes[:rep]:
                    start=time.perf_counter(); checksum=calls=0
                    for ids in batches:
                        out,n=run(mode,ids)
                        checksum+=sum(x[0] for x in out); calls+=n
                    elapsed=time.perf_counter()-start
                    useful=len(batches)*10*sum(x['size'] for x in c)
                    row=dict(rep=rep,mode=mode,seconds=elapsed,useful_GBps=useful/elapsed/1e9,
                             planned_reads=calls,checksum=checksum)
                    rows.append(row); print(json.dumps(row),flush=True)
            assert len({r['checksum'] for r in rows})==1
        med={m:statistics.median(r['useful_GBps'] for r in rows if r['mode']==m) for m in modes}
        report=dict(kind='MEASURED_LOCAL_CACHE_INFLUENCED_FILE_IO_NOT_DEVICE_OR_DECODE',
                    host=platform.node(),source=str(model),components=c,batches=batches,
                    component_byte_checks=checks,rows=rows,median_useful_GBps=med,
                    lanes_vs_futures=med['lanes']/med['futures'],lanes_vs_serial=med['lanes']/med['serial'],
                    caveat='F_NOCACHE accepted, no physical counters. T1 may be using same host for capture; no isolation claim. No checkpoint writes or model loads.')
        (ROOT/'results/t6_real_batch_read.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(medians=med,lanes_vs_futures=report['lanes_vs_futures'],lanes_vs_serial=report['lanes_vs_serial']),indent=2))
    finally:
        for fd in fds.values(): os.close(fd)

if __name__=='__main__':
    with open('/tmp/moefit-lab/state/local_ssd_bench.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        main()
