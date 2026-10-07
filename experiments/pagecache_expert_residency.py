"""How much of the routed-expert tables sit in the macOS page cache (UBC)?

oMLX 0.7.0 reads missing experts with os.pread (no F_NOCACHE), so every slab
it has read stays in the unified buffer cache until memory pressure evicts it.
That is a second, invisible expert tier on top of the resident slots. This
script mmaps each shard read-only and calls mincore() over every expert slab
of every switch_mlp tensor, then reports per-layer/whole-model the fraction of
expert slabs that are fully / partially cached.

No sudo, no writes, no server touch. Run with any python3 on the serving Mac.

usage: python3 pagecache_expert_residency.py MODEL_DIR [--out f.json]
"""
import ctypes, ctypes.util, json, mmap, os, struct, sys, time
from pathlib import Path

libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                      ctypes.c_int, ctypes.c_longlong]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
PAGE = os.sysconf("SC_PAGE_SIZE")


def main():
    md = Path(sys.argv[1])
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    t0 = time.time()
    # per (layer, expert): [cached_pages, total_pages]
    acc = {}
    for shard in sorted(md.glob("*.safetensors")):
        with open(shard, "rb") as f:
            hl = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hl))
        ents = []
        for name, s in hdr.items():
            if (name.startswith("language_model.model.layers.") and "switch_mlp" in name
                    and len(s["shape"]) == 3 and s["shape"][0] == 512):
                li = int(name.split(".")[3])
                ents.append((li, 8 + hl + s["data_offsets"][0],
                             s["data_offsets"][1] - s["data_offsets"][0]))
        if not ents:
            continue
        fd = os.open(shard, os.O_RDONLY)
        size = os.fstat(fd).st_size
        a0 = libc.mmap(None, size, 1, 1, fd, 0)  # PROT_READ, MAP_SHARED
        if a0 in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), "mmap")
        npages = (size + PAGE - 1) // PAGE
        vec = ctypes.create_string_buffer(npages)
        if libc.mincore(ctypes.c_void_p(a0), size, vec) != 0:
            raise OSError(ctypes.get_errno(), "mincore")
        v = vec.raw
        for li, off, nb in ents:
            slab = nb // 512
            for e in range(512):
                s0 = off + e * slab
                p0, p1 = s0 // PAGE, (s0 + slab - 1) // PAGE + 1
                c = sum(1 for b in v[p0:p1] if b & 1)
                a = acc.setdefault((li, e), [0, 0])
                a[0] += c
                a[1] += p1 - p0
        libc.munmap(ctypes.c_void_p(a0), size)
        os.close(fd)
    full = sum(1 for c, t in acc.values() if c == t)
    part = sum(1 for c, t in acc.values() if 0 < c < t)
    cached_pages = sum(c for c, _ in acc.values())
    tot_pages = sum(t for _, t in acc.values())
    per_layer = {}
    for (li, e), (c, t) in acc.items():
        pl = per_layer.setdefault(li, [0, 0])
        pl[0] += c == t
        pl[1] += 1
    res = dict(kind="measured", host=os.uname().nodename,
               timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               experts=len(acc), fully_cached=full, partially_cached=part,
               cached_GB=round(cached_pages * PAGE / 1e9, 2),
               expert_tables_GB=round(tot_pages * PAGE / 1e9, 2),
               fully_cached_per_layer_mean=round(full / max(len(per_layer), 1), 1),
               fully_cached_per_layer_min=min(v[0] for v in per_layer.values()),
               fully_cached_per_layer_max=max(v[0] for v in per_layer.values()),
               scan_s=round(time.time() - t0, 1))
    print(json.dumps(res))
    if out:
        Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
