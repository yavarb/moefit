"""Page-cache residency (mincore) of every tensor FAMILY in a safetensors model dir.

Families: expert (switch_mlp), ple_ngram (PLE n-gram embedding shards, read on
demand via mmap by oMLX's PLE SSD offload), ple_other, other (attn/shared/etc).
Use before/after a decode to attribute physical SSD reads: oMLX expert misses
use pread (cached in UBC), PLE rows are mmap gathers (also UBC).

No sudo, read-only, no server touch.
usage: python3 pagecache_families.py MODEL_DIR [--out f.json]
"""
import ctypes, ctypes.util, json, os, struct, sys, time
from pathlib import Path

libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                      ctypes.c_int, ctypes.c_longlong]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
PAGE = os.sysconf("SC_PAGE_SIZE")


def family(name):
    if "switch_mlp" in name:
        return "expert"
    if "ngram_embedding" in name:
        return "ple_ngram"
    if ".ple." in name:
        return "ple_other"
    return "other"


def scan(md):
    out = {}
    for shard in sorted(Path(md).glob("*.safetensors")):
        with open(shard, "rb") as f:
            hl = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hl))
        fd = os.open(shard, os.O_RDONLY)
        size = os.fstat(fd).st_size
        a0 = libc.mmap(None, size, 1, 1, fd, 0)
        if a0 in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), "mmap")
        npages = (size + PAGE - 1) // PAGE
        vec = ctypes.create_string_buffer(npages)
        if libc.mincore(ctypes.c_void_p(a0), size, vec) != 0:
            raise OSError(ctypes.get_errno(), "mincore")
        v = vec.raw
        for name, s in hdr.items():
            if name == "__metadata__":
                continue
            off = 8 + hl + s["data_offsets"][0]
            nb = s["data_offsets"][1] - s["data_offsets"][0]
            if nb <= 0:
                continue
            p0, p1 = off // PAGE, (off + nb - 1) // PAGE + 1
            c = v[p0:p1].count(b"\x01") + sum(1 for b in v[p0:p1] if b & 1 and b != 1)
            fam = out.setdefault(family(name), [0, 0])
            fam[0] += c
            fam[1] += p1 - p0
        libc.munmap(ctypes.c_void_p(a0), size)
        os.close(fd)
    return {k: dict(cached_GB=round(c * PAGE / 1e9, 3), total_GB=round(t * PAGE / 1e9, 2))
            for k, (c, t) in out.items()}


def main():
    md = sys.argv[1]
    t0 = time.time()
    res = dict(kind="measured", host=os.uname().nodename,
               timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"), families=scan(md),
               scan_s=None)
    res["scan_s"] = round(time.time() - t0, 1)
    print(json.dumps(res))
    if "--out" in sys.argv:
        Path(sys.argv[sys.argv.index("--out") + 1]).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
