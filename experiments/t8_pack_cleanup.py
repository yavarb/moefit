"""Tiny executed pack build failures and retry; not crash-safety proof."""
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from moefit.expert_pack import build_pack, ExpertPack


def main():
    cases = []
    with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as td:
        root = Path(td); source = root/'source'
        source.write_bytes(bytes(range(16)))
        parts = [dict(file=str(source),name='weight',offset=0,size=8,experts=2,dtype='U32',shape=[2])]
        with patch('moefit.expert_pack.components',return_value=parts):
            for kind in ('read','pack_fsync','index_fsync','serialize','interrupt'):
                output = root/kind; index = Path(str(output)+'.json')
                if kind == 'read': fault = patch('os.pread',return_value=b'')
                elif kind == 'pack_fsync': fault = patch('os.fsync',side_effect=OSError('injected'))
                elif kind == 'index_fsync': fault = patch('os.fsync',side_effect=[None,OSError('injected')])
                elif kind == 'serialize': fault = patch('moefit.expert_pack.json.dumps',side_effect=ValueError('injected'))
                else: fault = patch('os.pread',side_effect=KeyboardInterrupt())
                with fault:
                    try: build_pack(root,output,[1,0],alignment=16)
                    except (EOFError,OSError,ValueError,KeyboardInterrupt): pass
                    else: raise AssertionError('failure not raised')
                assert not output.exists() and not index.exists()
                build_pack(root,output,[1,0],alignment=16)
                reader = ExpertPack(output)
                try:
                    assert reader.read(0,True)['weight'] == bytes(range(8))
                    assert reader.read(1,True)['weight'] == bytes(range(8,16))
                finally: reader.close()
                cases.append(kind)
            for kind in ('existing_pack','orphan_index'):
                output = root/kind; index = Path(str(output)+'.json')
                target = output if kind=='existing_pack' else index
                target.write_bytes(b'preserve')
                with patch('os.open',side_effect=AssertionError('source opened')):
                    try: build_pack(root,output,[0],alignment=16)
                    except FileExistsError: pass
                    else: raise AssertionError('existing accepted')
                assert target.read_bytes() == b'preserve'
                assert not (index if kind=='existing_pack' else output).exists()
                cases.append(kind)
        assert source.read_bytes()==bytes(range(16))
    result = dict(scope='Tiny synthetic real file cleanup + injected exceptions, NOT crash durability/BW/decode.',
                  passed_cases=cases,cleanup_retry_cases=5,preserved_existing_cases=2,
                  verified_expert_reads=10,source_unchanged=True,
                  limitations='Not atomic across two files; process kill/power loss or concurrent destination mutation unsupported.')
    (ROOT/'results/t8_pack_cleanup.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
