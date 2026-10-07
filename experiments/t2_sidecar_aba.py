"""T2 sidecar replay-prefetch A/B/A on the merged oMLX patch (Santa Cruz).

Arms (same prompt, greedy, n tokens):
  A0  flag OFF: warm + record the replay script (not scored)
  A   flag OFF                       -> baseline tok/s, logical misses/token
  B   flag ON  (/tmp/omlx_sidecar_on) -> replay prefetch of t+1 routes
  A2  flag OFF
Reads /tmp/omlx_moe_stats.json (written each 1 s by the patched ExpertCache)
before/after every run: MEASURED logical hits/misses and sidecar counters.
stdlib only. Usage: python3 t2_sidecar_aba.py --out t2_sidecar_aba.json
"""
import argparse, json, os, time, urllib.request

STATS = "/tmp/omlx_moe_stats.json"
FLAG = "/tmp/omlx_sidecar_on"
PROMPT = ("Write a long, detailed essay on the history of unified memory "
          "architectures in personal computers, from early shared-memory designs "
          "to Apple Silicon. Cover trade-offs in bandwidth, latency and cost.")


def server_pid():
    import subprocess
    return subprocess.run(["pgrep", "-f", "omlx-server"], capture_output=True, text=True).stdout.split()


def stats():
    time.sleep(1.3)  # let the 1 s dumper flush
    for _ in range(30):
        if os.path.exists(STATS):
            return json.load(open(STATS))
        time.sleep(1)
    raise SystemExit("stats file missing (server restarted?)")


def decode(url, model, n):
    body = json.dumps(dict(model=model, stream=True, temperature=0, max_tokens=n,
                           stream_options={"include_usage": True},
                           messages=[{"role": "user", "content": PROMPT}])).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    t0 = time.time(); first = last = None; usage = None; finish = None; text = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                c = d.get("content") or d.get("reasoning_content") or d.get("reasoning")
                if c:
                    now = time.time(); first = first or now; last = now; text.append(c)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    ntok = (usage or {}).get("completion_tokens")
    return dict(tokens=ntok, finish_reason=finish, ttft_s=round(first - t0, 3),
                decode_s=round(last - first, 3),
                tps=round((ntok - 1) / (last - first), 2),
                text_hash=hash("".join(text)) & 0xFFFFFFFF)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next-oQ4e-mtp")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--arms", default="A0,A,B,A2,B2,A3")
    ap.add_argument("--out", default="t2_sidecar_aba.json")
    a = ap.parse_args()
    runs = []
    pid0 = server_pid()
    for arm in a.arms.split(","):
        if server_pid() != pid0:
            print("ABORT: omlx-server pid changed", pid0, server_pid(), flush=True)
            break
        on = arm.startswith("B")
        if on:
            open(FLAG, "w").close()
        elif os.path.exists(FLAG):
            os.remove(FLAG)
        time.sleep(0.6)  # flag polled every 0.25 s
        s0 = stats()
        r = decode(a.url, a.model, a.max_tokens)
        s1 = stats()
        d = {k: s1[k] - s0[k] for k in s0 if isinstance(s0[k], int) and k != "layers"}
        r.update(arm=arm, sidecar=on, **{f"d_{k}": v for k, v in d.items()})
        tot = d["hits"] + d["misses"]
        r["misses_per_token_incl_prefill"] = round(d["misses"] / max(r["tokens"], 1), 2)
        r["hit_rate"] = round(d["hits"] / tot, 4) if tot else None
        r["server_pid"] = pid0
        if server_pid() != pid0:
            r["invalid"] = "server restarted during run"
        runs.append(r); print(json.dumps(r), flush=True)
        json.dump(dict(partial=True, runs=runs), open(a.out, "w"), indent=1)
    if os.path.exists(FLAG):
        os.remove(FLAG)
    ok = [r for r in runs if not r.get("invalid")]
    A = [r["tps"] for r in ok if r["arm"].startswith("A") and r["arm"] != "A0"] or [0]
    B = [r["tps"] for r in ok if r["arm"].startswith("B")] or [0]
    out = dict(kind="measured", host=os.uname().nodename,
               timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               max_tokens=a.max_tokens, prompt=PROMPT, runs=runs,
               tps_off_mean=round(sum(A) / len(A), 2), tps_on_mean=round(sum(B) / len(B), 2),
               delta_tps=round(sum(B) / len(B) - sum(A) / len(A), 2),
               note="replay regime: identical prompt+greedy => exact route repeat; "
                    "misses include prefill; omlx prefix cache may skip prefill on repeats")
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
