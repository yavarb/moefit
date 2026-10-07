"""Warm expert-pack read+full-byte consumption benchmark, NOT physical SSD."""
import fcntl,json,os,random,statistics,sys,tempfile,time,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from moefit.expert_pack import build_pack,ExpertPack,MappedExpertPack


def main():
    with open('/tmp/moefit-lab/state/local_ssd_bench.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'],prefix='t8_mmap_') as td:
            path=Path(td)/'pack'
            model=Path.home()/'.lmstudio/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp'
            m=build_pack(model,path,range(16)); p=ExpertPack(path); q=MappedExpertPack(path)
            count=0
            b={}
            for eid in range(16):
                a=p.read(eid,verify=True);b=q.read(eid,verify=True)
                for key in a:
                    assert a[key]==b[key] and b[key].readonly
                    count+=1
            try:q.close()
            except BufferError:pass
            else:raise AssertionError('live exports should prevent close')
            del b
            q.close();q.close()
            try:q.read(0)
            except ValueError:pass
            else:raise AssertionError('read after close')
            q=MappedExpertPack(path)
            ids=list(range(16));random.Random(1616).shuffle(ids)
            modes=['pread_random','mapped_random','pread_sequential','mapped_sequential']
            def bench(mode):
                reader=q if mode.startswith('mapped') else p
                start=time.perf_counter(); checksum=0
                for eid in (ids if mode.endswith('random') else range(16)):
                    views=reader.read(eid)
                    # CRC touches EVERY payload byte, no view-only fake BW.
                    for v in views.values():checksum=(checksum+zlib.crc32(v))%(1<<64)
                elapsed=time.perf_counter()-start
                return dict(seconds=elapsed,useful_GBps=16*m['payload_bytes']/elapsed/1e9,checksum=checksum)
            for mode in modes:bench(mode)
            rows={mode:[] for mode in modes}
            for rot in range(4):
                for mode in modes[rot:]+modes[:rot]:rows[mode].append(bench(mode))
            assert len({r['checksum'] for rows_ in rows.values() for r in rows_})==1
            med={k:statistics.median(r['useful_GBps'] for r in v) for k,v in rows.items()}
            p.close();q.close()
            out=dict(kind='MEASURED_WARM_MEMORY_PATH_NOT_SSD_OR_DECODE',component_equalities=count,
                     lifetime_guards='live exports, double close, read after close PASS',
                     samples=rows,median_useful_GBps=med,
                     random_speedup=med['mapped_random']/med['pread_random'],
                     sequential_speedup=med['mapped_sequential']/med['pread_sequential'],
                     caveat='All pages warmed; read+CRC32 includes full payload consumption. No F_NOCACHE claim for mmap, no physical-byte counters. File must remain immutable; exported views prevent close.')
            (ROOT/'results/t8_mapped_pack.json').write_text(json.dumps(out,indent=2)+'\n')
            print(json.dumps({k:v for k,v in out.items() if k!='samples'},indent=2))


if __name__=='__main__':main()
