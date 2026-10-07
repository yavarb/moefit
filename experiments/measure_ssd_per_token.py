"""Measure physical SSD read bytes per decoded token on a running oMLX server.

Runs on the Mac that serves the model (e.g. M4 Max 36 GB). Samples `iostat -d -w 1`
for the boot disk while issuing one greedy streaming chat completion, then
attributes disk reads inside the decode window to decode.

This does NOT restart or reconfigure the server. It measures physical reads
(page-cache misses), which lower-bounds the expert bytes oMLX pread()s: an
expert slab that is still in the unified buffer cache is a logical read with no
disk traffic.

usage: python3 measure_ssd_per_token.py [--max-tokens 256] [--disk disk0] [--out f.json]
"""
import argparse, json, subprocess, threading, time, urllib.request

PROMPT = ("Write a long, detailed essay on the history of unified memory "
          "architectures in personal computers, from early shared-memory designs "
          "to Apple Silicon. Cover trade-offs in bandwidth, latency and cost.")


def sampler(disk, samples, stop):
    p = subprocess.Popen(["iostat", "-d", "-w", "1", disk], stdout=subprocess.PIPE,
                         text=True, bufsize=1)
    try:
        for line in p.stdout:
            if stop.is_set():
                break
            parts = line.split()
            if len(parts) == 3:
                try:
                    kbt, tps, mbs = map(float, parts)
                except ValueError:
                    continue
                samples.append((time.time(), kbt, tps, mbs))
    finally:
        p.terminate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next-oQ4e-mtp")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--disk", default="disk0")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    samples, stop = [], threading.Event()
    th = threading.Thread(target=sampler, args=(a.disk, samples, stop), daemon=True)
    th.start()
    time.sleep(3.5)  # idle baseline (first iostat line is since-boot; dropped below)

    body = json.dumps(dict(model=a.model, stream=True, temperature=0,
                           max_tokens=a.max_tokens,
                           stream_options={"include_usage": True},
                           messages=[{"role": "user", "content": a.prompt}])).encode()
    req = urllib.request.Request(a.url + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time(); t_first = t_last = None; usage = None; deltas = 0
    with urllib.request.urlopen(req, timeout=900) as r:
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
                    now = time.time()
                    t_first = t_first or now
                    t_last = now
                    deltas += 1
    time.sleep(3.5)
    stop.set()

    ntok = (usage or {}).get("completion_tokens") or deltas
    decode_s = t_last - t_first
    # iostat line at time ts covers (ts-1, ts]
    body_s = samples[1:]
    idle = [s for s in body_s if s[0] <= t0]
    dec = [s for s in body_s if t_first + 1.0 <= s[0] <= t_last]
    idle_mbs = sum(s[3] for s in idle) / max(len(idle), 1)
    dec_mbs = sum(s[3] for s in dec) / max(len(dec), 1)
    tps = (ntok - 1) / decode_s
    res = dict(
        kind="measured", host=subprocess.getoutput("hostname"),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        max_tokens=a.max_tokens, tokens=ntok, decode_s=round(decode_s, 3),
        decode_tps=round(tps, 2), ttft_s=round(t_first - t0, 3),
        idle_disk_MBps=round(idle_mbs, 2), decode_disk_MBps=round(dec_mbs, 1),
        decode_disk_MB_per_token=round((dec_mbs - idle_mbs) / tps, 1),
        decode_avg_KB_per_io=round(sum(s[1] for s in dec) / max(len(dec), 1), 1),
        decode_iops=round(sum(s[2] for s in dec) / max(len(dec), 1), 0),
        n_decode_samples=len(dec),
        samples=[[round(s[0] - t0, 2), s[1], s[2], s[3]] for s in body_s],
        method="iostat -d -w 1 on boot disk; samples fully inside (first+1s, last] "
               "token window; MB/token = (decode MB/s - idle MB/s) / decode tok/s. "
               "Physical reads only (page-cache hits invisible).",
    )
    print(json.dumps({k: v for k, v in res.items() if k != "samples"}, indent=1))
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
