"""Run directly with python3 tests/test_coalesced_read.py (stdlib only)."""
import os
from pathlib import Path
import random
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moefit.coalesced_read import plan_reads, read_batch, read_exact


def main():
    rng = random.Random(61)
    payload = rng.randbytes(8192)
    cases = [[], [(100,200),(0,150),(100,200)], [(0,100),(110,100)], [(0,100),(120,100)]]
    cases += [[(rng.randrange(7000),rng.randrange(1,100)) for _ in range(24)] for _ in range(100)]
    with tempfile.TemporaryFile(dir=os.environ['TMPDIR']) as f, ThreadPoolExecutor(4) as pool:
        f.write(payload); f.flush()
        for extents in cases:
            for lanes in (0,4):
                out, stat = read_batch(f.fileno(),extents,executor=pool,lanes=lanes,
                                      max_gap=32,max_span=1024,max_amplification=1.125)
                assert [bytes(x) for x in out] == [payload[o:o+n] for o,n in extents]
                for span in plan_reads(extents,max_gap=32,max_span=1024,max_amplification=1.125):
                    addresses=set()
                    for _, off, n in span.members:
                        addresses.update(range(off,off+n))
                    assert span.length<=1024 and span.length<=1.125*len(addresses)
        assert len(plan_reads([(0,100),(110,100)],max_gap=32))==1
        assert len(plan_reads([(0,100),(200,100)],max_gap=100))==2
        for extents in ([(-1,1)],[(0,0)],[(0,2**21)],[(0.5,3)]):
            try: plan_reads(extents)
            except ValueError: pass
            else: raise AssertionError('invalid extent accepted')
        with patch('moefit.coalesced_read.os.pread', side_effect=[b'ab',b'c']):
            assert read_exact(f.fileno(),3,0)==b'abc'
        with patch('moefit.coalesced_read.os.pread', return_value=b''):
            try: read_exact(f.fileno(),1,0)
            except EOFError: pass
            else: raise AssertionError('EOF ignored')
    print('PASS: 104 cases x2 execution paths, order/overlap/duplicate equality, span/amplification bounds, gap merge/refusal, invalid extents, partial-read and EOF')

if __name__=='__main__': main()
