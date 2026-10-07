"""Read-only real checkpoint ON/OFF with physical disk counters on Darwin.
Run with model directory and output JSON. No server/model calls or cache purge.
"""
import fcntl,hashlib,json,os,plistlib,random,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moefit.expert_pack import components
from moefit.coalesced_read import read_exact
from moefit.expert_batch_read import read_experts


def disk():
    data=plistlib.loads(subprocess.check_output(['ioreg','-a','-r','-l','-c','IOBlockStorageDriver']))
    for item in data:
        if any(c.get('BSD Name')=='disk0' for c in item.get('IORegistryEntryChildren',[])):
            return item['Statistics']['Bytes (Read)']
    raise RuntimeError('disk0 physical counter unavailable')


def main():
    model,out=map(Path,sys.argv[1:3])
    if os.environ.get('T6_SSD_SLOT_CONFIRMED') != '1':
        raise RuntimeError('Requires coordinated idle SSD slot: T6_SSD_SLOT_CONFIRMED=1')
    cs=[components(model,li) for li in range(48)]
    assert all(c['experts']==512 for group in cs for c in group)
    paths={c['file'] for group in cs for c in group}
    fds={p:os.open(p,os.O_RDONLY) for p in paths}
    rows=[]
    try:
        for fd in fds.values():
            fcntl.fcntl(fd,48,1); fcntl.fcntl(fd,45,0)
        a=disk(); t=time.perf_counter(); time.sleep(1); b=disk()
        idle=(b-a)/(time.perf_counter()-t)
        rng=random.Random(6025)
        random_batches=[(i%48,rng.sample(range(512),10)) for i in range(384)]
        sequential_batches=[(i%48,list(range((i//48)*10,(i//48)*10+10))) for i in range(384)]
        with ThreadPoolExecutor(4) as pool:
            def fetch(arg):
                e,c=arg
                return read_exact(fds[c['file']],c['size'],c['offset']+e*c['size'])
            validation=[]
            for li in (0,23,47):
                ids=[1,17,17,511]
                expected=[fetch((e,c)) for e in ids for c in cs[li]]
                got,_=read_experts(cs[li],fds,ids,executor=pool,lanes=4)
                actual=[x[c['name']] for x in got for c in cs[li]]
                assert all(a==b for a,b in zip(expected,actual))
                validation.append(hashlib.sha256(b''.join(expected)).hexdigest())
            for rep in range(2):
                arms=[('random','OFF'),('random','ON'),('sequential','OFF'),('sequential','ON')]
                if rep: arms.reverse()
                for order,arm in arms:
                    batches=random_batches if order=='random' else sequential_batches
                    p0=disk(); start=time.perf_counter(); useful=0; check=0
                    for li,ids in batches:
                        group=cs[li]
                        if arm=='OFF':
                            blocks=list(pool.map(fetch,[(e,c) for e in ids for c in group]))
                        else:
                            got,_=read_experts(group,fds,ids,executor=pool,lanes=4)
                            blocks=[x[c['name']] for x in got for c in group]
                        useful+=sum(len(x) for x in blocks); check+=sum(x[0] for x in blocks)
                    seconds=time.perf_counter()-start; p1=disk()
                    r=dict(rep=rep,order=order,arm=arm,seconds=seconds,useful_bytes=useful,
                           useful_GBps=useful/seconds/1e9,physical_bytes=p1-p0,
                           physical_over_useful=(p1-p0)/useful,
                           physical_GBps=(p1-p0)/seconds/1e9,checksum=check)
                    rows.append(r); print(json.dumps(r),flush=True)
                    out.write_text(json.dumps(dict(host=os.uname().nodename,idle_read_Bps=idle,rows=rows),indent=2))
        for order in ('random','sequential'):
            assert len({r['checksum'] for r in rows if r['order']==order})==1
        report=dict(host=os.uname().nodename,kind='REAL_CHECKPOINT_READS_WITH_PHYSICAL_DISK0_COUNTERS',
                    idle_read_Bps=idle,workers=4,experts_per_batch=10,layers=48,
                    validation_sha256=validation,
                    cache_control='F_NOCACHE=1,F_RDAHEAD=0; no global purge',rows=rows,
                    caveat='Physical counters are whole disk, include metadata/background; before/after counter subprocess overhead outside useful timing. No decode speed claim.')
        out.write_text(json.dumps(report,indent=2)+'\n')
    finally:
        for fd in fds.values():os.close(fd)

if __name__=='__main__':main()
