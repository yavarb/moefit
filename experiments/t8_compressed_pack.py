"""Fixed-codec invention trial on real quantized weights, no parameter sweep."""
import fcntl,json,os,random,statistics,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from moefit.expert_pack import build_pack,ExpertPack
from moefit.compressed_expert_pack import compress_pack,CompressedExpertPack


def main():
    with open('/tmp/moefit-lab/state/local_ssd_bench.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'],prefix='t8_compressed_') as td:
            root=Path(td); raw=root/'raw'; compressed=root/'compressed'
            model=Path.home()/'.lmstudio/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp'
            m=build_pack(model,raw,range(16));start=time.perf_counter()
            cm=compress_pack(raw,compressed);build_s=time.perf_counter()-start
            readers={'raw':ExpertPack(raw),'compressed':CompressedExpertPack(compressed)}
            checks=0
            for eid in range(16):
                a=readers['raw'].read(eid,verify=True);b=readers['compressed'].read(eid,verify=True)
                for name in a:assert a[name]==b[name];checks+=1
            for r in readers.values():fcntl.fcntl(r.fd,48,1)
            ids=list(range(16));random.Random(1111).shuffle(ids)
            modes=[('raw','random'),('compressed','random'),('raw','sequential'),('compressed','sequential')]
            def trial(mode):
                kind,order=mode;start=time.perf_counter();checksum=0
                for eid in (ids if order=='random' else range(16)):
                    v=readers[kind].read(eid)
                    checksum+=sum(x[0] for x in v.values())
                s=time.perf_counter()-start
                return dict(seconds=s,useful_GBps=16*m['payload_bytes']/s/1e9,checksum=checksum)
            for mode in modes:trial(mode)
            rows={f'{a}_{b}':[] for a,b in modes}
            for rot in range(4):
                for mode in modes[rot:]+modes[:rot]:rows['_'.join(mode)].append(trial(mode))
            assert len({x['checksum'] for r in rows.values() for x in r})==1
            med={k:statistics.median(x['useful_GBps'] for x in r) for k,r in rows.items()}
            uncompressed=16*m['payload_bytes'];stored=compressed.stat().st_size
            raw_s=statistics.median(x['seconds'] for x in rows['raw_random'])
            comp_s=statistics.median(x['seconds'] for x in rows['compressed_random'])
            penalty=max(0,comp_s-raw_s)
            # Conditional serial model: saved bytes/B must exceed measured
            # extra CPU/cache-path time. Not an actual cold-SSD prediction.
            out=dict(kind='MEASURED_LOCAL_REAL_WEIGHT_CODEC_TRIAL_NOT_DECODE',experts=16,component_equalities=checks,
                     payload_bytes=uncompressed,stored_bytes=stored,stored_ratio=stored/uncompressed,
                     encode_seconds=build_s,median_useful_GBps=med,samples=rows,
                     extra_read_decode_seconds=penalty,
                     conditional_break_even_storage_GBps=(uncompressed-stored)/penalty/1e9 if penalty else None,
                     decision='DROP synchronous zlib1 speed path; retain executable negative result',
                     caveat='Warm/cache-influenced local reads with F_NOCACHE accepted; source model may be in use by T1. No device counters, GPU install or model decode. Break-even assumes serial byte savings and this CPU penalty transfer.')
            for r in readers.values():r.close()
            (ROOT/'results/t8_compressed_pack.json').write_text(json.dumps(out,indent=2)+'\n')
            print(json.dumps({k:v for k,v in out.items() if k!='samples'},indent=2))


if __name__=='__main__':main()
