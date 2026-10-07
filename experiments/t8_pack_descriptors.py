"""Execute old/new pack writers against tiny synthetic shard extents."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import moefit.expert_pack as current


def main():
    old = {}
    source = subprocess.check_output(['git','show','75db8cf:moefit/expert_pack.py'], cwd=ROOT, text=True)
    exec(compile(source, 'baseline_expert_pack', 'exec'), old)
    original_open, original_close = os.open, os.close
    with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as td:
        root = Path(td)
        parts = []
        for shard in range(3):
            path = root/f'shard{shard}'
            path.write_bytes(bytes(range(48)))
            for part in range(3):
                parts.append(dict(file=str(path), name=f'{shard}_{part}', offset=part*16,
                                  size=8, experts=2, dtype='U32', shape=[2]))
        def run(fn, ns, name, fail_open=None, fail_read=False):
            opened = []; closed = []; attempts = 0
            def tracked_open(path, flags):
                nonlocal attempts
                attempts += 1
                if attempts == fail_open: raise OSError('injected open failure')
                fd = original_open(path, flags); opened.append(fd); return fd
            def tracked_close(fd):
                closed.append(fd); return original_close(fd)
            before = ns['components']; ns['components'] = lambda *a:parts
            error = None
            try:
                with patch('os.open', tracked_open), patch('os.close', tracked_close):
                    if fail_read:
                        with patch('os.pread', return_value=b''):
                            try: fn(root,root/name,[1,0],alignment=64)
                            except EOFError: error = 'EOFError'
                    else:
                        try: fn(root,root/name,[1,0],alignment=64)
                        except OSError: error = 'OSError'
            finally:
                ns['components'] = before
            leaked = [fd for fd in opened if fd not in closed]
            for fd in leaked:
                os.fstat(fd)
                original_close(fd)  # Explicitly clean baseline leaks.
            for fd in closed:
                try: os.fstat(fd)
                except OSError: pass
                else: raise AssertionError('descriptor still open')
            result: dict = dict(opens=len(opened),closes=len(closed),leaks=len(leaked),error=error)
            return result
        baseline = run(old['build_pack'],old,'old')
        fixed = run(current.build_pack,current.__dict__,'new')
        assert baseline == dict(opens=9,closes=3,leaks=6,error=None)
        assert fixed == dict(opens=3,closes=3,leaks=0,error=None)
        assert (root/'old').read_bytes() == (root/'new').read_bytes()
        assert json.loads((root/'old.json').read_text()) == json.loads((root/'new.json').read_text())
        failure = run(current.build_pack,current.__dict__,'openfail',fail_open=2)
        assert failure == dict(opens=1,closes=1,leaks=0,error='OSError')
        assert not (root/'openfail').exists()
        short = run(current.build_pack,current.__dict__,'short',fail_read=True)
        assert short == dict(opens=3,closes=3,leaks=0,error='EOFError')
        repeats = [run(current.build_pack,current.__dict__,f'repeat{i}') for i in range(20)]
        assert all(r['leaks']==0 for r in repeats)
    result = dict(scope='Executed real descriptors on tiny synthetic extents; no bandwidth/decode claim.',
                  baseline_revision='75db8cf',baseline=baseline,fixed=fixed,open_failure=failure,
                  short_read=short,repeated_builds=20,repeated_leaks=sum(r['leaks'] for r in repeats),
                  identical_pack_and_manifest=True,
                  caveat='Short-write/read failures may still leave partial destination; this change only fixes source descriptor lifetime.')
    (ROOT/'results/t8_pack_descriptors.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__': main()
