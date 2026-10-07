"""Guarded local real-pack random/sequential file IO with disk0 counters."""
import fcntl
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import time
import zlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from moefit.expert_pack import build_pack, ExpertPack
from t6_ssd_proof import disk


def gate():
    if Path('/tmp/omlx-prefetch-bench/CLAIMED').exists():
        raise RuntimeError('MBP reserved')
    processes=subprocess.check_output(['ps','-axo','pid,comm,args'],text=True)
    for line in processes.splitlines():
        if 'Python' in line and any(x in line for x in ('bench_xlayer_mbp.py','capture_xlayer','t6_ssd_proof.py')):
            raise RuntimeError('Competing benchmark: '+line.strip())
    a=disk(); t=time.perf_counter(); time.sleep(1); b=disk()
    rate=(b-a)/(time.perf_counter()-t)
    if not 0 <= rate < 50_000_000: raise RuntimeError(f'Background read rate {rate} B/s')
    return rate


def main():
    result: dict=dict(scope='Real oQ4e pack file reads plus whole-disk counters; NOT decode. No global purge; cached pages may persist.',rows=[])
    out=ROOT/'results/t8_pack_physical.json'
    try:
        with open('/tmp/moefit-lab/state/local_ssd_bench.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result['initial_idle_Bps']=gate()
            with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'],prefix='t8_phys_') as td:
                path=Path(td)/'pack'
                m=build_pack('/Users/yb/.lmstudio/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp',path,range(64))
                reader=ExpertPack(path)
                try:
                    fcntl.fcntl(reader.fd,48,1)
                    fcntl.fcntl(reader.fd,45,0)
                    order=list(range(64)); random.Random(3107).shuffle(order)
                    expected={}
                    for eid in range(64):
                        views=reader.read(eid,verify=True)
                        expected[eid]=zlib.crc32(b''.join(views.values()))
                    result['verified_experts']=64
                    for rep in range(3):
                        for label in (('random','sequential') if rep%2==0 else ('sequential','random')):
                            idle=gate(); p0=disk(); t=time.perf_counter()
                            checksum=0
                            for eid in (order if label=='random' else range(64)):
                                views=reader.read(eid)
                                crc=0
                                for v in views.values(): crc=zlib.crc32(v,crc)
                                assert crc==expected[eid]
                                checksum ^= crc
                            seconds=time.perf_counter()-t; physical=disk()-p0
                            issued=64*m['stride']; useful=64*m['payload_bytes']
                            row: dict=dict(rep=rep,order=label,seconds=seconds,useful_GBps=useful/seconds/1e9,
                                     issued_bytes=issued,physical_bytes=physical,physical_over_issued=physical/issued,
                                     idle_before_Bps=idle,checksum=checksum)
                            result['rows'].append(row)
                            row['idle_after_Bps']=gate()
                            row['device_attribution_gate']=0.9 <= row['physical_over_issued'] <= 1.1
                            print(json.dumps(row),flush=True)
                finally: reader.close()
            result['median_useful_GBps']={label:statistics.median(r['useful_GBps'] for r in result['rows'] if r['order']==label) for label in ('random','sequential')}
            result['physical_claim_supported']=all(r['device_attribution_gate'] for r in result['rows'])
            result['caveat']='Whole-disk counter window includes subprocess overhead; attribution ratio/idle gates are necessary checks, not proof of exclusive IO. Small freshly written/validated pack can be cache-served.'
    except (RuntimeError,BlockingIOError) as e:
        result['blocker']=str(e)
        result['physical_claim_supported']=False
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
