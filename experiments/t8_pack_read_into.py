"""Measure reusable direct-read oQ4e pack storage. Local file IO, not SSD claim."""
import fcntl
import json
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time
import tracemalloc
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from moefit.expert_pack import build_pack, ExpertPack


def main():
    model=Path.home()/'.lmstudio/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp'
    with open('/tmp/moefit-lab/state/local_ssd_bench.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(prefix='t8_into_',dir=os.environ['TMPDIR']) as td:
            path=Path(td)/'pack'
            m=build_pack(model,path,range(64)); p=ExpertPack(path)
            buf=bytearray(m['stride'])
            equalities=0
            for eid in range(64):
                a=p.read(eid,verify=True); b=p.read_into(eid,buf,verify=True)
                arrays=p.numpy_views(b)
                for c in m['components']:
                    key=c['name']
                    assert a[key]==b[key]
                    assert arrays[key].tobytes()==a[key]
                    assert not arrays[key].flags.writeable
                    assert np.shares_memory(arrays[key],np.frombuffer(buf,dtype=np.uint8))
                    equalities+=1
            for invalid in (bytes(m['stride']),bytearray(4),memoryview(buf)[::2]):
                try:p.read_into(0,invalid)
                except ValueError:pass
                else:raise AssertionError('invalid buffer accepted')
            fcntl.fcntl(p.fd,48,1)
            rand=list(range(64));random.Random(909).shuffle(rand)
            modes=['alloc_random','into_random','alloc_sequential','into_sequential']
            def bench(mode):
                order=rand if 'random' in mode else list(range(64))
                start=time.perf_counter(); checksum=0
                for eid in order:
                    v=p.read_into(eid,buf) if mode.startswith('into') else p.read(eid)
                    checksum+=sum(x[0] for x in v.values())
                elapsed=time.perf_counter()-start
                return dict(seconds=elapsed,useful_GBps=64*m['payload_bytes']/elapsed/1e9,checksum=checksum)
            for mode in modes:bench(mode)
            rows={mode:[] for mode in modes}
            for rot in range(4):
                for mode in modes[rot:]+modes[:rot]:rows[mode].append(bench(mode))
            peaks={}
            for mode in ('alloc_random','into_random'):
                tracemalloc.start();bench(mode);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop();peaks[mode]=peak
            assert len({r['checksum'] for samples in rows.values() for r in samples})==1
            med={mode:statistics.median(x['useful_GBps'] for x in samples) for mode,samples in rows.items()}
            p.close()
            out=dict(kind='MEASURED_LOCAL_CACHE_INFLUENCED_FILE_IO_NOT_DECODE',component_equalities=equalities,
                     typed_zero_copy_checks=equalities,invalid_buffer_checks=3,samples=rows,median_useful_GBps=med,
                     reusable_buffer_bytes=len(buf),tracemalloc_peak_incremental_bytes=peaks,
                     random_speedup=med['into_random']/med['alloc_random'],
                     sequential_speedup=med['into_sequential']/med['alloc_sequential'],
                     caveat='F_NOCACHE accepted; physical device traffic unverified. Existing-buffer allocation excluded from tracemalloc. All views alias buffer and must be consumed before reuse. No MLX/oMLX adapter.')
            (ROOT/'results/t8_pack_read_into.json').write_text(json.dumps(out,indent=2)+'\n')
            print(json.dumps({k:v for k,v in out.items() if k!='samples'},indent=2))


if __name__=='__main__':main()
