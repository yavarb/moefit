"""Warm local file-path benchmark; explicit demand-order oracle, NOT SSD proof."""
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
from moefit.reorder_expert_pack import reorder_pack


def consume(reader, order):
    crc = 0
    for eid in order:
        views = reader.read(eid)
        for v in views.values():
            crc = zlib.crc32(v, crc)
    return crc


def main():
    with open('/tmp/moefit-lab/state/local_ssd_bench.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'], prefix='t8_reorder_') as tmp:
            p, q = Path(tmp)/'base', Path(tmp)/'ordered'
            order = list(range(16)); random.Random(731).shuffle(order)
            m = build_pack('/Users/yb/.lmstudio/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp', p, range(16))
            start = time.perf_counter(); reordered = reorder_pack(p, q, order)
            build_s = time.perf_counter()-start
            a, b = ExpertPack(p), ExpertPack(q)
            equalities = 0
            try:
                for eid in order:
                    av, bv = a.read(eid, True), b.read(eid, True)
                    for name in av:
                        assert av[name] == bv[name]; equalities += 1
                assert reordered['sha256'] == m['sha256']
                guards = 0
                for invalid in [order[:-1], order[:-1]+[order[0]], order[:-1]+[999]]:
                    try: reorder_pack(p, Path(tmp)/'invalid', invalid)
                    except ValueError: guards += 1
                    else: raise AssertionError('invalid order accepted')
                try: reorder_pack(p, q, order)
                except FileExistsError: guards += 1
                else: raise AssertionError('existing output accepted')
                arms = [('identity_random', a, order), ('reordered_same_random', b, order),
                        ('identity_sequential', a, list(range(16)))]
                expected = {name:consume(r, ids) for name,r,ids in arms}
                assert expected['identity_random'] == expected['reordered_same_random']
                rows = {name:[] for name,_,_ in arms}
                for trial in range(6):
                    rotated = arms[trial%3:]+arms[:trial%3]
                    for name,r,ids in rotated:
                        t = time.perf_counter(); crc = consume(r, ids); dt = time.perf_counter()-t
                        assert crc == expected[name]
                        rows[name].append(dt)
            finally:
                a.close(); b.close()
            # Corruption must not publish even a partial destination.
            with p.open('r+b') as f:
                old = f.read(1); f.seek(0); f.write(bytes([old[0]^1]))
            bad = Path(tmp)/'corrupt-output'
            try: reorder_pack(p, bad, order)
            except ValueError: guards += 1
            else: raise AssertionError('corruption accepted')
            assert not bad.exists() and not Path(str(bad)+'.json').exists()
            result: dict = dict(scope='MEASURED LOCAL WARM file read+full-byte CRC, NOT physical SSD or decode. Known demand order is an oracle layout, not an online predictor.',
                          experts=16, component_equalities=equalities, guard_cases=guards,
                          order=order, reorder_seconds=build_s, payload_bytes=m['payload_bytes']*16,
                          seconds=rows, useful_GBps={k:m['payload_bytes']*16/statistics.median(v)/1e9 for k,v in rows.items()})
            result['layout_speed_ratio'] = result['useful_GBps']['reordered_same_random']/result['useful_GBps']['identity_random']
    out = Path(__file__).resolve().parents[1]/'results/t8_reorder_pack.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__': main()
