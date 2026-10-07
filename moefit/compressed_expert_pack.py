"""Independent lossless expert frames; raw fallback if compression expands.
Experimental sidecar, not a replacement checkpoint format. Never dequantizes.
"""
import hashlib
import json
import os
from pathlib import Path
import zlib
from moefit.expert_pack import ExpertPack, exact_pread


def compress_pack(source, output):
    source=Path(source); output=Path(output)
    reader=ExpertPack(source)
    m=reader.manifest
    frames={}
    try:
        with output.open('xb') as f:
            for eid in m['expert_ids']:
                views=reader.read(eid,verify=True)
                raw=b''.join(views[c['name']] for c in m['components'])
                compressed=zlib.compress(raw,level=1)
                codec='zlib1' if len(compressed)<len(raw) else 'raw'
                data=compressed if codec=='zlib1' else raw
                frames[str(eid)]=dict(offset=f.tell(),size=len(data),codec=codec,
                                     sha256=hashlib.sha256(raw).hexdigest())
                f.write(data)
            f.flush();os.fsync(f.fileno())
    finally:reader.close()
    manifest=dict(version=1,components=m['components'],payload_bytes=m['payload_bytes'],frames=frames)
    Path(str(output)+'.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


class CompressedExpertPack:
    def __init__(self,path):
        self.manifest=json.loads(Path(str(path)+'.json').read_text())
        self.fd=os.open(path,os.O_RDONLY)

    def read(self,eid,verify=False):
        m=self.manifest; frame=m['frames'][str(eid)]
        data=exact_pread(self.fd,frame['size'],frame['offset'])
        if frame['codec']=='zlib1':
            decoder=zlib.decompressobj()
            raw=decoder.decompress(data,m['payload_bytes']+1)
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise ValueError('Invalid compressed frame')
        elif frame['codec']=='raw':raw=data
        else:raise ValueError('Unknown frame codec')
        if len(raw)!=m['payload_bytes']:raise ValueError('Frame size mismatch')
        if verify and hashlib.sha256(raw).hexdigest()!=frame['sha256']:
            raise ValueError('Frame checksum mismatch')
        v=memoryview(raw)
        return {c['name']:v[c['pack_offset']:c['pack_offset']+c['size']] for c in m['components']}

    def close(self):os.close(self.fd)
