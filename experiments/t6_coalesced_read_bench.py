"""Local F_NOCACHE read microbench, synthetic bytes, not LLM throughput.
Unchanged tensor-major fixture: 9 component arrays x512 experts x64KiB.
Each batch selects 10 random expert IDs. Coalescer acts within that batch.
"""
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
from concurrent.futures import ThreadPoolExecutor
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moefit.coalesced_read import read_batch, read_exact, plan_reads


def main():
    if sys.platform != 'darwin':
        raise RuntimeError('This measurement explicitly requires Darwin F_NOCACHE')
    scratch = Path(os.environ['TMPDIR'])
    rng = random.Random(6501)
    E, C, size = 512, 9, 65536
    batches = []
    for _ in range(96):
        ids = rng.sample(range(E), 10)
        batches.append([(component*E*size+e*size, size) for e in ids for component in range(C)])
    methods = ['individual', 'adjacent', 'lanes']
    rows = []
    with tempfile.TemporaryDirectory(prefix='t6-read-', dir=scratch) as td:
        path = Path(td)/'fixture.bin'
        # Full materialization: no sparse holes, no repeated zero slabs.
        with path.open('wb', buffering=0) as f:
            for _ in range(E*C):
                f.write(rng.randbytes(size))
            os.fsync(f.fileno())
            fcntl.fcntl(f.fileno(), 51)  # F_FULLFSYNC
        fd = os.open(path, os.O_RDONLY)
        try:
            fcntl.fcntl(fd, 48, 1)  # F_NOCACHE; errors are fatal, no warm-cache fallback
            fcntl.fcntl(fd, 45, 0)  # F_RDAHEAD disable for all methods
            with ThreadPoolExecutor(max_workers=4) as pool:
                def execute(method, extents):
                    if method == 'individual':
                        blocks = list(pool.map(lambda x: read_exact(fd, x[1], x[0]), extents))
                        return blocks, {'planned_reads':len(extents), 'issued_bytes':sum(n for _,n in extents)}
                    return read_batch(fd, extents, executor=pool,
                                      lanes=4 if method=='lanes' else 0,
                                      max_gap=0,
                                      max_span=1024*1024, max_amplification=1.125)
                # Full-byte equality, including duplicates/overlap and caller order.
                cases = batches[:4]+[[(0,128),(32,256),(0,128),(size*7,50)]]
                for extents in cases:
                    expected, _ = execute('individual', extents)
                    for method in methods[1:]:
                        got, _ = execute(method, extents)
                        assert [bytes(x) for x in got] == expected
                assert plan_reads([])==[]
                try:
                    read_exact(fd, 1, E*C*size)
                except EOFError:
                    pass
                else:
                    raise AssertionError('EOF not detected')
                # Rotate order to reduce simple time/order bias. Include planning,
                # submission, returned buffer construction in elapsed time.
                for rep in range(3):
                    for method in methods[rep:]+methods[:rep]:
                        start=time.perf_counter()
                        issued=calls=checksum=0
                        lat=[]
                        for batch in batches:
                            t=time.perf_counter()
                            got, stat=execute(method,batch)
                            checksum += sum(x[0] for x in got)
                            calls+=stat['planned_reads']; issued+=stat['issued_bytes']
                            lat.append((time.perf_counter()-t)*1000)
                        elapsed=time.perf_counter()-start
                        useful=sum(n for b in batches for _,n in b)
                        row={'rep':rep,'method':method,'seconds':elapsed,
                             'useful_GBps':useful/elapsed/1e9,'issued_GBps':issued/elapsed/1e9,
                             'read_amplification':issued/useful,'planned_reads':calls,
                             'batch_p50_ms':statistics.median(lat),'checksum':checksum}
                        rows.append(row); print(json.dumps(row),flush=True)
            assert len({r['checksum'] for r in rows})==1
        finally:
            os.close(fd)
    med={m:statistics.median(r['useful_GBps'] for r in rows if r['method']==m) for m in methods}
    result={'kind':'MEASURED local Darwin file-read microbenchmark, synthetic fixture; NOT M4 Max 36 GB/LLM',
            'host':platform.node(),'platform':platform.platform(),
            'fixture_bytes':E*C*size,'experts':E,'components':C,'component_bytes':size,
            'batch_experts':10,'batches':len(batches),'workers':4,
            'cache_controls':'F_NOCACHE=1, F_RDAHEAD=0(disable), fsync+F_FULLFSYNC after writing',
            'caveat':'OS cache-bypass requested; SSD/controller cache and concurrent host activity uncontrolled. No physical iostat attribution.',
            'workload_sha256':hashlib.sha256(json.dumps(batches).encode()).hexdigest(),
            'rows':rows,'median_useful_GBps':med,
            'relative_to_individual':{m:med[m]/med['individual'] for m in methods},
            'verified':'5 full-byte batch equality cases per coalescer, checksum equality all9 trials, empty planner and EOF handling',
            'scratch_removed':True}
    (ROOT/'results/t6_coalesced_read_bench.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'median':med,'speedup':result['relative_to_individual']},indent=2))

if __name__=='__main__':
    with open('/tmp/moefit-lab/state/local_ssd_bench.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
