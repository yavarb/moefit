"""Local real-oQ4e lossless layout benchmark, NOT decode throughput."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from moefit.expert_pack import build_pack, ExpertPack, exact_pread


def main():
    model=Path.home()/'.lmstudio/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp'
    with open('/tmp/moefit-lab/state/local_ssd_bench.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(prefix='t8_oqpack_',dir=os.environ['TMPDIR']) as td:
            target=Path(td)/'layer0.pack'
            m=build_pack(model,target,range(64))
            p=ExpertPack(target)
            c=m['components']
            fds={x['file']:os.open(x['file'],os.O_RDONLY) for x in c}
            # Byte-for-byte tensor verification outside timed sections.
            checks=0
            for eid in m['expert_ids']:
                packed=p.read(eid,verify=True)
                for x in c:
                    expected=exact_pread(fds[x['file']],x['size'],x['offset']+eid*x['size'])
                    assert packed[x['name']]==expected
                    checks+=1
            nocache=[]
            for fd in list(fds.values())+[p.fd]:
                if sys.platform!='darwin':
                    raise RuntimeError('This benchmark requires Darwin F_NOCACHE')
                fcntl.fcntl(fd,48,1)  # Darwin F_NOCACHE; failure is fatal
                nocache.append(True)
            shuffled=list(range(64)); random.Random(808).shuffle(shuffled)
            seq=list(range(64))
            modes=['source_random','pack_random','pack_sequential']
            samples={name:[] for name in modes}
            def bench(name):
                order=seq if name=='pack_sequential' else shuffled
                start=time.perf_counter()
                total=0
                for eid in order:
                    if name=='source_random':
                        for x in c:
                            data=exact_pread(fds[x['file']],x['size'],x['offset']+eid*x['size'])
                            total+=len(data)
                    else:
                        data=exact_pread(p.fd,m['stride'],p.positions[eid]*m['stride'])
                        total+=len(data)
                elapsed=time.perf_counter()-start
                useful=len(order)*m['payload_bytes']
                return dict(seconds=elapsed,useful_GBps=useful/elapsed/1e9,
                            read_bytes=total,useful_bytes=useful,
                            pread_calls=len(order)*(9 if name=='source_random' else 1))
            # One untimed warmup per path then balanced rotation of three repeats.
            for name in modes:
                bench(name)
            for rotation in range(3):
                for name in modes[rotation:]+modes[:rotation]:
                    samples[name].append(bench(name))
            med={name:statistics.median(r['useful_GBps'] for r in rows) for name,rows in samples.items()}
            p.close()
            for fd in fds.values(): os.close(fd)
            report=dict(kind='MEASURED_LOCAL_FILE_IO_NOT_LLM_DECODE',host=platform.node(),
                source=str(model),layer=0,experts=64,component_byte_equalities=checks,
                f_nocache_enabled=all(nocache),
                caveat='F_NOCACHE requested successfully; no device-counter proof of physical SSD bytes, existing cache/device effects possible. Small subset, single host, Python pread allocation included. No oMLX integration.',
                payload_bytes_per_expert=m['payload_bytes'],stride=m['stride'],
                padding_fraction=(m['stride']-m['payload_bytes'])/m['payload_bytes'],
                source_headers=c,expert_sha256=m['sha256'],samples=samples,median_useful_GBps=med,
                pack_random_over_source_random=med['pack_random']/med['source_random'],
                pack_sequential_over_pack_random=med['pack_sequential']/med['pack_random'])
            (ROOT/'results/t8_oq4e_pack.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({k:report[k] for k in ['component_byte_equalities','payload_bytes_per_expert','stride','padding_fraction','median_useful_GBps','pack_random_over_source_random','pack_sequential_over_pack_random']},indent=2))


if __name__=='__main__':main()
