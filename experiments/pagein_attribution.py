"""Attribute physical SSD reads during one decode to tensor families via
page-cache (mincore) deltas. No sudo, no server change.

1. mincore bitmap of every shard (before)
2. one greedy streaming decode (measure_ssd_per_token.py: iostat MB/token)
3. mincore bitmap (after)
Pages that went 0->1 were read from SSD during the window (page-in) and are
attributed to the family owning them. Lower bound: a page read then evicted
inside the window is missed, so compare the total with iostat bytes.

oMLX 0.7.0 reads expert misses with os.pread (lands in UBC) and gathers PLE
n-gram rows through mmap (also UBC), so both are visible.

usage: python3 pagein_attribution.py MODEL_DIR --max-tokens 256 --out f.json
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pagecache_families import libc, PAGE, family  # noqa: E402
import ctypes, struct  # noqa: E402


def layout(md):
    shards = []
    for shard in sorted(Path(md).glob("*.safetensors")):
        with open(shard, "rb") as f:
            hl = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hl))
        ranges = []
        for name, s in hdr.items():
            if name == "__metadata__":
                continue
            off = 8 + hl + s["data_offsets"][0]
            nb = s["data_offsets"][1] - s["data_offsets"][0]
            if nb > 0:
                ranges.append((off // PAGE, (off + nb - 1) // PAGE + 1, family(name), name))
        shards.append((shard, sorted(ranges)))
    return shards


def bitmap(shard):
    fd = os.open(shard, os.O_RDONLY)
    size = os.fstat(fd).st_size
    a0 = libc.mmap(None, size, 1, 1, fd, 0)
    n = (size + PAGE - 1) // PAGE
    vec = ctypes.create_string_buffer(n)
    if libc.mincore(ctypes.c_void_p(a0), size, vec) != 0:
        raise OSError(ctypes.get_errno(), "mincore")
    raw = vec.raw
    libc.munmap(ctypes.c_void_p(a0), size)
    os.close(fd)
    return bytes(b & 1 for b in raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lay = layout(a.model_dir)
    before = {s: bitmap(s) for s, _ in lay}
    tmp = "/tmp/pagein_ssd.json"
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "measure_ssd_per_token.py"),
           "--max-tokens", str(a.max_tokens), "--out", tmp]
    if a.prompt:
        cmd += ["--prompt", a.prompt]
    t0 = time.time()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    dt = time.time() - t0
    after = {s: bitmap(s) for s, _ in lay}
    ssd = json.load(open(tmp))
    fam_new, fam_lost, layer_new = {}, {}, {}
    for shard, ranges in lay:
        b, c = before[shard], after[shard]
        for p0, p1, fam, name in ranges:
            new = sum(1 for x, y in zip(b[p0:p1], c[p0:p1]) if y and not x)
            lost = sum(1 for x, y in zip(b[p0:p1], c[p0:p1]) if x and not y)
            fam_new[fam] = fam_new.get(fam, 0) + new
            fam_lost[fam] = fam_lost.get(fam, 0) + lost
            if fam == "ple_ngram" and new:
                li = name.split(".")[3]
                layer_new[li] = layer_new.get(li, 0) + new
    ntok = ssd["tokens"]
    res = dict(
        kind="measured", host=os.uname().nodename,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"), tokens=ntok,
        decode_tps=ssd["decode_tps"], window_s=round(dt, 1),
        iostat_decode_MB_per_token=ssd["decode_disk_MB_per_token"],
        iostat_total_MB_window_approx=round(ssd["decode_disk_MBps"] * ssd["decode_s"], 0),
        pagein_MB_per_token={k: round(v * PAGE / 1e6 / ntok, 2) for k, v in fam_new.items()},
        pagein_MB_total={k: round(v * PAGE / 1e6, 1) for k, v in fam_new.items()},
        evicted_MB_total={k: round(v * PAGE / 1e6, 1) for k, v in fam_lost.items()},
        ple_layers_touched=len(layer_new),
        note="window includes prefill+TTFT; page-ins are a lower bound (read-then-evicted pages invisible)",
    )
    print(json.dumps(res, indent=1))
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
