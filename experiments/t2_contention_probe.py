"""Silicon contention probe for the T2 idle-window prefetch assumption.

T2 (design_idle_prefetch.py) assumed: background expert reads issued while
oMLX decodes do NOT slow the decode (the SSD is idle during compute+sync).
This probe tests that directly on the serving Mac without touching oMLX:

  1. A/B/A decode runs (greedy, same prompt) via the running server.
  2. During B, a side process issues random expert-slab preads (F_NOCACHE,
     oMLX's 9-slab layout, 12-thread pool) at a fixed background rate
     R experts/s = budget_per_token x baseline tok/s (T2 budget = 50/token).
  3. Report tok/s(A) vs tok/s(B), achieved background GB/s, and the
     implied extra ms/token.

If decode tok/s drops by << the background share, the idle window is real
(T2 optimistic bound). If it drops ~ proportionally, demand and background
reads contend (T7 pessimistic bound). Background reads are random experts,
so they never help the decode: this measures pure interference cost.

usage (on M4 Max 36 GB, omlx python for nothing; stdlib only):
  python3 t2_contention_probe.py MODEL_DIR --budget 50 --max-tokens 512
"""
import argparse, fcntl, json, os, random, struct, subprocess, threading, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

F_NOCACHE = 48
PROMPT = ("Write a long, detailed essay on the history of unified memory "
          "architectures in personal computers, from early shared-memory designs "
          "to Apple Silicon. Cover trade-offs in bandwidth, latency and cost.")


def load_layout(md):
    specs, by_layer = {}, {}
    for shard in sorted(Path(md).glob("*.safetensors")):
        with open(shard, "rb") as f:
            hl = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hl))
        fd = os.open(shard, os.O_RDONLY)
        fcntl.fcntl(fd, F_NOCACHE, 1)
        for name, s in hdr.items():
            if name == "__metadata__":
                continue
            shp = s["shape"]
            if ("switch_mlp" in name and len(shp) == 3 and shp[0] == 512
                    and name.startswith("language_model.model.layers.")):
                nb = s["data_offsets"][1] - s["data_offsets"][0]
                li = int(name.split(".")[name.split(".").index("layers") + 1])
                by_layer.setdefault(li, []).append(
                    (fd, 8 + hl + s["data_offsets"][0], nb // 512))
    return by_layer


def pread_all(fd, off, nb):
    got = 0
    while got < nb:
        c = os.pread(fd, nb - got, off + got)
        if not c:
            raise OSError("short read")
        got += len(c)
    return nb


def background(by_layer, rate, stop, stats):
    rng = random.Random(7)
    pool = ThreadPoolExecutor(12)
    L = sorted(by_layer)
    t0 = time.perf_counter(); issued = 0; nbytes = 0
    while not stop.is_set():
        due = int((time.perf_counter() - t0) * rate)
        futs = []
        while issued < due:
            li = rng.choice(L); e = rng.randrange(512)
            for fd, off, slab in by_layer[li]:
                futs.append(pool.submit(pread_all, fd, off + e * slab, slab))
            issued += 1
        for f in futs:
            nbytes += f.result()
        time.sleep(0.002)
    stats.update(experts=issued, GB=nbytes / 1e9, s=time.perf_counter() - t0)
    pool.shutdown()


def decode(url, model, n):
    body = json.dumps(dict(model=model, stream=True, temperature=0, max_tokens=n,
                           stream_options={"include_usage": True},
                           messages=[{"role": "user", "content": PROMPT}])).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    t0 = time.time(); first = last = None; usage = None; deltas = 0
    with urllib.request.urlopen(req, timeout=1200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    now = time.time(); first = first or now; last = now; deltas += 1
    ntok = (usage or {}).get("completion_tokens") or deltas
    return dict(tokens=ntok, ttft_s=round(first - t0, 3),
                decode_s=round(last - first, 3),
                tps=round((ntok - 1) / (last - first), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next-oQ4e-mtp")
    ap.add_argument("--budget", type=float, default=50.0,
                    help="background experts per decoded token (T2 budget)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", default="t2_contention_probe.json")
    a = ap.parse_args()
    by_layer = load_layout(a.model_dir)
    runs = []
    base = None
    for phase in ("A", "B", "A2"):
        bg = {}
        stop = threading.Event(); th = None
        if phase == "B":
            rate = a.budget * base
            th = threading.Thread(target=background, args=(by_layer, rate, stop, bg))
            th.start(); time.sleep(0.5)
        r = decode(a.url, a.model, a.max_tokens)
        if th:
            stop.set(); th.join()
            r["bg_target_experts_per_s"] = round(rate, 1)
            r["bg_experts_per_s"] = round(bg["experts"] / bg["s"], 1)
            r["bg_GBps"] = round(bg["GB"] / bg["s"], 3)
            r["bg_experts_per_token"] = round(bg["experts"] / bg["s"] / r["tps"], 1)
        r["phase"] = phase
        runs.append(r); print(json.dumps(r), flush=True)
        if phase == "A":
            base = r["tps"]
    a_tps = (runs[0]["tps"] + runs[2]["tps"]) / 2
    b = runs[1]
    out = dict(kind="measured", host=subprocess.getoutput("hostname"),
               timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               budget_experts_per_token=a.budget, max_tokens=a.max_tokens,
               runs=runs, tps_A_mean=round(a_tps, 2), tps_B=b["tps"],
               slowdown_frac=round(1 - b["tps"] / a_tps, 3),
               extra_ms_per_token=round(1000 / b["tps"] - 1000 / a_tps, 2),
               bg_ms_per_token_if_serial=round(
                   b["bg_experts_per_token"] * 2.7648 / 5.0, 2),
               note="background reads are random (useless) experts: pure "
                    "interference cost of T2's idle-window budget")
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1))
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
