"""T1 guarded cross-layer fetch A/B/A on M4 Max 36 GB (stdlib only).

Server must run patches/omlx_t1_xlayer.py with MOEFIT_XLAYER_D=1 in its env
(all other flags default off). Arms toggle ONLY the flag file /tmp/omlx_xlayer_on;
no restart between arms. Per arm: tok/s plus diffs of /tmp/omlx_moe_stats.json
(hits, misses, xl_*, ensure_s = wall time inside demand _ensure_ids).

Pre-registered (T1 notebook 2026-10-07 00:45 / 01:00 ET):
  PRIMARY  demand-wait (ensure_s) ON < OFF in every pair; MBP measured -6.3 ms/tok
  SECONDARY tok/s; ON-OFF >= +0.30 TRANSFERS, |d| < 0.30 WASH, <= -0.30 CONTENTION
            (expected band 15.5-16.6 on the 15.72 anchor prompt family)
  SANITY   misses/tok equal across arms (held-not-installed), text_hash equal.
usage: python3 t1_xlayer_aba.py --arms A0,A,B,A2,B2,A3,B3 --out t1_xlayer_aba.json
"""
import argparse, json, os, subprocess, time, urllib.request

STATS = "/tmp/omlx_moe_stats.json"
FLAG = "/tmp/omlx_xlayer_on"
PROMPT = ("Write a long, detailed essay on the history of unified memory "
          "architectures in personal computers, from early shared-memory designs "
          "to Apple Silicon. Cover trade-offs in bandwidth, latency and cost.")
KEYS = ("hits", "misses", "xl_issued", "xl_used", "xl_ready", "xl_wasted", "ensure_s")


def server_pid():
    return subprocess.run(["pgrep", "-f", "omlx-server"], capture_output=True, text=True).stdout.split()


def stats():
    time.sleep(1.3)
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
                decode_s=round(last - first, 3), tps=round((ntok - 1) / (last - first), 2),
                text_hash=hash("".join(text)) & 0xFFFFFFFF)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next-oQ4e-mtp")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--arms", default="A0,A,B,A2,B2,A3,B3")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pid0 = server_pid()
    s0 = stats()
    if "xl_issued" not in s0 or "ensure_s" not in s0:
        raise SystemExit("server is not running the T1 xlayer file (no xl_/ensure_s counters)")
    rows = []
    try:
        for arm in a.arms.split(","):
            on = arm.startswith("B")
            if on:
                open(FLAG, "w").close()
            elif os.path.exists(FLAG):
                os.remove(FLAG)
            time.sleep(0.6)  # flag is polled every 0.25 s
            before = stats()
            r = decode(a.url, a.model, a.n)
            after = stats()
            if server_pid() != pid0:
                r["ABORTED"] = "server pid changed"; rows.append(dict(arm=arm, **r)); break
            n = r["tokens"] or 1
            d = {k: after.get(k, 0) - before.get(k, 0) for k in KEYS}
            r.update(arm=arm, flag_on=on, miss_per_tok=round(d["misses"] / n, 2),
                     xl_issued_per_tok=round(d["xl_issued"] / n, 2),
                     xl_used_per_tok=round(d["xl_used"] / n, 2),
                     xl_ready_per_tok=round(d["xl_ready"] / n, 2),
                     xl_wasted_per_tok=round(d["xl_wasted"] / n, 2),
                     demand_wait_ms_per_tok=round(1000 * d["ensure_s"] / n, 2),
                     note="counter windows include the request's prefill")
            rows.append(r); print(json.dumps(r), flush=True)
            json.dump(dict(rows=rows, partial=True), open(a.out, "w"), indent=1)
    finally:
        if os.path.exists(FLAG):
            os.remove(FLAG)
    sc = [r for r in rows if r["arm"] != "A0" and "ABORTED" not in r]
    pairs = [(x, y) for x, y in zip(sc, sc[1:]) if not x["flag_on"] and y["flag_on"]]
    res = dict(kind="MEASURED M4 Max 36 GB T1 xlayer A/B/A (flag toggle, no restart)", pid=pid0,
               rows=rows,
               paired_tps=[round(y["tps"] - x["tps"], 2) for x, y in pairs],
               paired_wait_ms=[round(y["demand_wait_ms_per_tok"] - x["demand_wait_ms_per_tok"], 2) for x, y in pairs],
               text_hashes=sorted({r["text_hash"] for r in rows}))
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("paired_tps", "paired_wait_ms", "text_hashes")}))


if __name__ == "__main__":
    main()
