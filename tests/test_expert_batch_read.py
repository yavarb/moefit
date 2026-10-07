"""Tiny byte-correctness preflight; NOT a throughput benchmark."""
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moefit.expert_batch_read import read_experts


def main():
    with tempfile.TemporaryDirectory() as root:
        fds={}
        cs=[]
        expected={}
        try:
            for shard in range(2):
                p=str(Path(root)/str(shard))
                payload=bytes((i+shard)%256 for i in range(4096))
                Path(p).write_bytes(payload)
                fds[p]=os.open(p,os.O_RDONLY)
                for part in range(3):
                    c=dict(name=f'{shard}:{part}',file=p,offset=part*1024,size=37+part,experts=8)
                    cs.append(c)
                    for e in range(8):
                        off=part*1024+e*(37+part)
                        expected[e,c['name']]=payload[off:off+37+part]
            checks=0
            with ThreadPoolExecutor(4) as pool:
                for executor in (None,pool):
                    for ids in ([],[7,0,7,3],list(range(8))):
                        got,stats=read_experts(cs,fds,ids,executor=executor)
                        assert len(got)==len(ids)
                        for e,entry in zip(ids,got):
                            for c in cs:
                                assert entry[c['name']]==expected[e,c['name']]
                                checks+=1
                        assert stats['issued_bytes']<=stats['requested_bytes']
                for ids in ([-1],[8],[True],[1.0]):
                    try:read_experts(cs,fds,ids)
                    except ValueError:pass
                    else:raise AssertionError('invalid ID accepted')
            print(f'PASS: {checks} component equalities, two shards, duplicate/order/empty cases, invalid-ID guards')
        finally:
            for fd in fds.values():os.close(fd)

if __name__=='__main__':main()
