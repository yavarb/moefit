#!/usr/bin/env python3
"""bench_decode.py - wall-clock decode tok/s against an OpenAI-compatible server.

Greedy (temperature 0) streaming chat completions. The first request warms the
server (it also triggers oMLX's lazy model load, which can take minutes when
experts are wrapped for offload). Each timed run reports:

  ttft_s      time to the first streamed token
  decode_tps  (tokens - 1) / (last token time - first token time)
  wall_tps    tokens / total request time

Token counts come from the server's ``usage.completion_tokens`` when it sends
one, else from the number of streamed deltas (reported as ``count_source``).

usage:
    python3 scripts/bench_decode.py --model <id> [--url http://127.0.0.1:8000]
        [--max-tokens 128] [--runs 3] [--label "measured on ..."]
        [--out results/measured_<host>.json]
"""
import argparse, json, os, platform, statistics, subprocess, sys, time
import urllib.request, urllib.error

PROMPT = ("Write a long, detailed essay on the history of unified memory "
          "architectures in personal computers, from early shared-memory "
          "designs to today's system-on-chip approaches. Do not stop early.")


def sysctl(key):
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except Exception:
        return None


def http_json(url, payload=None, timeout=3600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def stream_once(url, model, max_tokens, timeout):
    payload = {
        "model": model, "temperature": 0, "max_tokens": max_tokens,
        "stream": True, "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": PROMPT}],
    }
    req = urllib.request.Request(url + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    deltas = 0
    usage = None
    finish = None
    text = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                j = json.loads(body)
            except json.JSONDecodeError:
                continue
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices") or []:
                d = ch.get("delta") or {}
                piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning")
                if piece:
                    now = time.perf_counter()
                    first = first if first is not None else now
                    last = now
                    deltas += 1
                    text.append(piece)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    t_end = time.perf_counter()
    if first is None:
        raise RuntimeError("no tokens streamed")
    n = usage.get("completion_tokens") if usage else None
    source = "usage.completion_tokens" if n else "streamed_deltas"
    n = n or deltas
    return {
        "tokens": n, "count_source": source, "streamed_deltas": deltas,
        "ttft_s": round(first - t0, 3),
        "decode_s": round(last - first, 3),
        "decode_tps": round((n - 1) / (last - first), 2) if last > first and n > 1 else None,
        "wall_s": round(t_end - t0, 3),
        "wall_tps": round(n / (t_end - t0), 2),
        "finish_reason": finish,
        "prompt_tokens": usage.get("prompt_tokens") if usage else None,
        "sample": "".join(text)[:160],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup-tokens", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=3600, help="per request, seconds")
    ap.add_argument("--label", default="measured")
    ap.add_argument("--notes", default="")
    ap.add_argument("--out")
    a = ap.parse_args()

    models = [m["id"] for m in http_json(a.url + "/v1/models", timeout=60)["data"]]
    if a.model not in models:
        raise SystemExit(f"model {a.model!r} not served; server has {models}")

    print(f"warm-up ({a.warmup_tokens} tokens; includes model load if cold) ...", flush=True)
    t = time.perf_counter()
    w = stream_once(a.url, a.model, a.warmup_tokens, a.timeout)
    print(f"  warm-up done in {time.perf_counter() - t:.1f}s (ttft {w['ttft_s']}s)", flush=True)

    runs = []
    for i in range(a.runs):
        r = stream_once(a.url, a.model, a.max_tokens, a.timeout)
        runs.append(r)
        print(f"  run {i + 1}: {r['tokens']} tok, ttft {r['ttft_s']}s, "
              f"decode {r['decode_tps']} tok/s, wall {r['wall_tps']} tok/s "
              f"[{r['count_source']}, finish={r['finish_reason']}]", flush=True)

    decode = [r["decode_tps"] for r in runs if r["decode_tps"]]
    summary = {
        "label": a.label,
        "kind": "measured",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "chip": sysctl("machdep.cpu.brand_string"),
        "ram_gib": round(int(sysctl("hw.memsize") or 0) / 2 ** 30, 1),
        "macos": platform.mac_ver()[0],
        "server_url": a.url,
        "model": a.model,
        "max_tokens": a.max_tokens,
        "runs": runs,
        "decode_tps_median": round(statistics.median(decode), 1) if decode else None,
        "decode_tps_min": min(decode) if decode else None,
        "decode_tps_max": max(decode) if decode else None,
        "ttft_s_median": round(statistics.median(r["ttft_s"] for r in runs), 2),
        "warmup": w,
        "method": ("greedy streaming chat completion; decode_tps = (tokens-1) / "
                   "(time from first to last streamed token); wall clock on the client"),
        "notes": a.notes,
    }
    try:
        summary["omlx_version"] = subprocess.check_output(
            ["omlx", "--version"], text=True, env={**os.environ, "PATH": "/opt/homebrew/bin:" + os.environ.get("PATH", "")}).strip()
    except Exception:
        pass
    ms = os.path.expanduser("~/.omlx/model_settings.json")
    if os.path.exists(ms):
        try:
            summary["omlx_model_settings"] = json.load(open(ms))["models"].get(a.model)
        except Exception:
            pass
    try:
        summary["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))).strip()
    except Exception:
        pass
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
