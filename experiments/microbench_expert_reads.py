"""Microbench oMLX-style expert slab reads on the serving Mac (no server touch).

Reproduces CheckpointExpertStore's access pattern: per expert, 6-9 positional
reads (gate/up/down x weight/scales[/biases]) from the checkpoint shards,
issued on a 12-thread pool. F_NOCACHE on the fds forces real SSD reads so the
numbers are cold-miss costs. Reports per-layer latency when a decode step
misses k experts in one layer (oMLX resolves misses layer by layer, with a
device sync between layers), and aggregate throughput at large batches.

Also times the host->slot install (np.frombuffer -> mx.array -> slot
assignment) if mlx is importable (run with omlx's python).

usage: python3 microbench_expert_reads.py /path/to/model_dir
"""
import fcntl, json, os, random, struct, sys, time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

F_NOCACHE = 48


def load_specs(md):
    specs = {}
    fds = {}
    for shard in sorted(Path(md).glob("*.safetensors")):
        with open(shard, "rb") as f:
            hl = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hl))
        fd = os.open(shard, os.O_RDONLY)
        fcntl.fcntl(fd, F_NOCACHE, 1)
        fds[shard] = fd
        for name, s in hdr.items():
            if name == "__metadata__":
                continue
            specs[name] = (fd, s["dtype"], s["shape"], 8 + hl + s["data_offsets"][0],
                           s["data_offsets"][1] - s["data_offsets"][0])
    return specs


def expert_plans(specs, layer_prefix_by_layer, li, e):
    out = []
    for name in layer_prefix_by_layer[li]:
        fd, _, shape, off, nb = specs[name]
        slab = nb // shape[0]
        out.append((fd, off + e * slab, slab))
    return out


def pread_all(fd, off, nb):
    got = 0
    while got < nb:
        c = os.pread(fd, nb - got, off + got)
        if not c:
            raise OSError("short")
        got += len(c)
    return nb


def main():
    md = sys.argv[1]
    specs = load_specs(md)
    # stacked expert tensors: <...>.layers.<i>.<...>switch_mlp.<proj>.<field>
    by_layer = {}
    for name, (fd, dt, shape, off, nb) in specs.items():
        if "switch_mlp" in name and len(shape) == 3 and shape[0] == 512:
            if not name.startswith("language_model.model.layers."):
                continue  # skip MTP / other stacks sharing layer indices
            parts = name.split(".")
            li = int(parts[parts.index("layers") + 1])
            by_layer.setdefault(li, []).append(name)
    L = sorted(by_layer)
    per_expert = sum(specs[n][4] // 512 for n in by_layer[L[0]])
    print(f"layers={len(L)} tensors/layer={len(by_layer[L[0]])} expert_MB={per_expert/1e6:.3f}", flush=True)
    pool = ThreadPoolExecutor(12)
    rng = random.Random(1)
    res = {"expert_MB": per_expert / 1e6, "tensors_per_expert": len(by_layer[L[0]])}

    def step(k, layers=48):
        """one decode step: for each layer, read k random experts, wait."""
        t0 = time.perf_counter()
        for li in rng.sample(L, layers):
            futs = []
            for _ in range(k):
                e = rng.randrange(512)
                for fd, off, nb in expert_plans(specs, by_layer, li, e):
                    futs.append(pool.submit(pread_all, fd, off, nb))
            wait(futs)
        return time.perf_counter() - t0

    for k in (1, 2, 3, 4):
        ts = [step(k) for _ in range(3)]
        ms = min(ts) * 1000
        mb = k * 48 * per_expert / 1e6
        res[f"step_k{k}_ms"] = round(ms, 1)
        res[f"step_k{k}_GBps"] = round(mb / ms, 2)
        print(f"k={k}/layer x48 layers: {ms:.1f} ms/step  {mb:.0f} MB  {mb/ms:.2f} GB/s  "
              f"per-layer {ms/48:.2f} ms", flush=True)
    # bulk throughput: 96 experts at once
    t0 = time.perf_counter(); futs = []
    for _ in range(96):
        li = rng.choice(L); e = rng.randrange(512)
        for fd, off, nb in expert_plans(specs, by_layer, li, e):
            futs.append(pool.submit(pread_all, fd, off, nb))
    wait(futs); dt = time.perf_counter() - t0
    res["bulk96_GBps"] = round(96 * per_expert / 1e9 / dt, 2)
    print(f"bulk 96 experts: {res['bulk96_GBps']} GB/s", flush=True)

    try:
        import numpy as np
        import mlx.core as mx
        slots = mx.zeros((143, per_expert // 4), dtype=mx.uint32)
        mx.eval(slots)
        raw = os.urandom(per_expert)
        for _ in range(3):
            slots[0] = mx.array(np.frombuffer(raw, np.uint32))
        mx.eval(slots)
        n = 40
        t0 = time.perf_counter()
        for i in range(n):
            slots[i % 143] = mx.array(np.frombuffer(raw, np.uint32))
            mx.eval(slots)
        res["install_ms_per_expert_eval_each"] = round((time.perf_counter() - t0) / n * 1000, 2)
        t0 = time.perf_counter()
        for i in range(n):
            slots[i % 143] = mx.array(np.frombuffer(raw, np.uint32))
        mx.eval(slots)
        res["install_ms_per_expert_batched"] = round((time.perf_counter() - t0) / n * 1000, 2)
        # per-layer device sync cost (.tolist() readback of a tiny array)
        x = mx.zeros((10,), mx.int32)
        t0 = time.perf_counter()
        for _ in range(480):
            (x + 1).tolist()
        res["tiny_sync_ms"] = round((time.perf_counter() - t0) / 480 * 1000, 3)
        print({k: v for k, v in res.items() if "install" in k or "sync" in k}, flush=True)
    except ImportError as ex:
        print("mlx not available:", ex)
    print("RESULT", json.dumps(res))


if __name__ == "__main__":
    main()
