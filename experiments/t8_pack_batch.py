"""T8 pack/T6 coalescer composition, warm local CRC benchmark, not SSD proof."""
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moefit.expert_pack import build_pack, ExpertPack
from moefit.pack_batch import read_pack_batch


def consume(pack, batches, batched):
    crc = calls = issued = 0
    for ids in batches:
        if batched:
            groups, stats = read_pack_batch(pack, ids)
            calls += stats['planned_reads']; issued += stats['issued_bytes']
        else:
            groups = [pack.read(eid) for eid in ids]
            calls += len(ids); issued += len(ids)*pack.manifest['stride']
        for group in groups:
            for v in group.values(): crc = zlib.crc32(v, crc)
    return crc, calls, issued


def main():
    with open('/tmp/moefit-lab/state/local_ssd_bench.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'], prefix='t8_packbatch_') as tmp:
            path = Path(tmp)/'pack'
            m = build_pack('~/models/Qwen3.8-Flash-Next-oQ4e-mtp', path, range(16))
            pack = ExpertPack(path)
            try:
                rng = random.Random(203)
                patterns = [[], [3,3,2,1], list(range(16))] + [rng.choices(range(16), k=7) for _ in range(20)]
                comparisons = 0
                for ids in patterns:
                    groups, stats = read_pack_batch(pack, ids, verify=True)
                    assert len(groups) == len(ids)
                    assert stats['issued_bytes'] == len(set(ids))*m['stride']
                    for eid, group in zip(ids, groups):
                        ref = pack.read(eid, True)
                        for key in ref:
                            assert ref[key] == group[key] and group[key].readonly
                            comparisons += 1
                guards = 0
                for invalid in [0,-1,1.5,True]:
                    try: read_pack_batch(pack, [0], max_experts_per_span=invalid)
                    except ValueError: guards += 1
                    else: raise AssertionError('invalid bound accepted')
                try: read_pack_batch(pack, [999])
                except KeyError: guards += 1
                else: raise AssertionError('missing ID accepted')
                random_order = list(range(16)); rng.shuffle(random_order)
                orders = {'random':random_order, 'sequential':list(range(16))}
                arms = [(name, [ids[i:i+4] for i in range(0,16,4)], fast)
                        for name,ids in orders.items() for fast in (False, True)]
                expected = {name:consume(pack, batches, False)[0] for name,batches,_ in arms}
                rows = {}; counters = {}
                for name,batches,fast in arms: consume(pack, batches, fast)
                for trial in range(8):
                    for name,batches,fast in arms[trial%4:]+arms[:trial%4]:
                        key = name+('_batch' if fast else '_individual')
                        start = time.perf_counter(); crc,calls,issued = consume(pack,batches,fast)
                        seconds = time.perf_counter()-start
                        assert crc == expected[name]
                        rows.setdefault(key,[]).append(seconds)
                        counters[key] = dict(read_calls=calls,issued_bytes=issued)
                rates = {k:16*m['payload_bytes']/statistics.median(v)/1e9 for k,v in rows.items()}
                result = dict(scope='MEASURED WARM local file read+CRC, NOT physical SSD or LLM decode',
                              experts=16, group_size=4, component_comparisons=comparisons,
                              correctness_patterns=len(patterns), guard_cases=guards,
                              seconds=rows, useful_GBps=rates, counters=counters,
                              ratios={k:rates[k+'_batch']/rates[k+'_individual'] for k in orders})
            finally: pack.close()
    out = Path(__file__).resolve().parents[1]/'results/t8_pack_batch.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__': main()
