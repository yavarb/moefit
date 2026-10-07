"""M1 silicon A/B: LRU-insert vs BIP vs DIP on ONE oMLX process (M4 Max 36 GB).

Owner: silicon_dbuf (T9). Runs only after design_inventor's DIP patch is code-green.

Contract with the patch (to be confirmed by design_inventor):
  * policy is read from the flag file /tmp/omlx_insert_policy (content: lru|bip|dip),
    polled by the patched ExpertCache; absent file == lru == stock behavior.
  * /tmp/omlx_moe_stats.json carries cumulative hits/misses (T2 counter patch) and,
    if available, an "insert_policy" field echoing the live policy.
Arms rotate L,B,D (x rounds) inside the SAME process: no restart between arms, so
per-arm counter diffs are same-process (the BATCH1 stats0 cross-process bug cannot recur).
Prefetch OFF is asserted (T2 sidecar flag and T1 xlayer flag must be absent).
Prompt = collect_silicon_run.py prompt (family sha 87eb8e913d26cca9), greedy, n tokens.
stdlib only.
"""
import argparse, hashlib, json, os, subprocess, time, urllib.request

STATS = os.environ.get("M1_STATS", "/tmp/omlx_moe_stats.json")
POLICY = os.environ.get("M1_POLICY", "/tmp/omlx_insert_policy")
FORBIDDEN_FLAGS = ["/tmp/omlx_sidecar_on", "/tmp/omlx_xlayer_on"]
PROMPT = ("Write a long, detailed essay on the history of unified "
          "memory architectures in personal computers, from early "
          "shared-memory designs to modern unified memory fabrics.")
FAMILY_SHA = "87eb8e913d26cca9"
ARM_POLICY = {"L": "lru", "B": "bip", "D": "dip"}


def pids():
    return subprocess.run(["pgrep", "-f", "omlx-server"], capture_output=True,
                          text=True).stdout.split()


def stats():
    time.sleep(1.3)  # dumper flushes every 1 s
    for _ in range(30):
        try:
            return json.load(open(STATS))
        except (OSError, ValueError):
            time.sleep(1)
    raise SystemExit("stats file missing/unreadable")


def decode(url, model, n, prompt):
    body = json.dumps(dict(model=model, stream=True, temperature=0, max_tokens=n,
                           stream_options={"include_usage": True},
                           messages=[{"role": "user", "content": prompt}])).encode()
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
    ntok = (usage or {}).get("completion_tokens") or 0
    dec = (last - first) if (first and last and last > first) else None
    return dict(tokens=ntok, finish_reason=finish,
                ttft_s=round(first - t0, 3) if first else None,
                decode_s=round(dec, 3) if dec else None,
                decode_tps=round((ntok - 1) / dec, 2) if dec and ntok > 1 else None,
                text_sha=hashlib.sha256("".join(text).encode()).hexdigest()[:16])


def set_policy(p):
    if p == "lru":
        if os.path.exists(POLICY):
            os.remove(POLICY)
    else:
        with open(POLICY + ".tmp", "w") as f:
            f.write(p)
        os.replace(POLICY + ".tmp", POLICY)
    time.sleep(0.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next-oQ4e-mtp")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--order", default="L,B,D", help="arm order within a round; rotated each round")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="skip pid/flag guards (mock server only)")
    a = ap.parse_args()

    assert hashlib.sha256(PROMPT.encode()).hexdigest()[:16] == FAMILY_SHA, "prompt drifted"
    if not a.dry_run:
        bad = [f for f in FORBIDDEN_FLAGS if os.path.exists(f)]
        if bad:
            raise SystemExit(f"REFUSED: prefetch flag(s) present {bad}")
        if len(pids()) != 1:
            raise SystemExit(f"REFUSED: need exactly one omlx-server, have {pids()}")
    pid0 = pids()
    order = a.order.split(",")
    plan = []
    for r in range(a.rounds):
        k = r % len(order)
        plan += [f"{x}{r + 1}" for x in order[k:] + order[:k]]
    set_policy("lru")
    print("warm (lru, not scored)", flush=True)
    warm = decode(a.url, a.model, 64, PROMPT)
    runs = []
    for arm in plan:
        if not a.dry_run and (pids() != pid0 or any(os.path.exists(f) for f in FORBIDDEN_FLAGS)):
            print("ABORT: server pid changed or prefetch flag appeared", flush=True)
            break
        pol = ARM_POLICY[arm[0]]
        set_policy(pol)
        s0 = stats()
        r = decode(a.url, a.model, a.max_tokens, PROMPT)
        s1 = stats()
        d = {k: s1[k] - s0[k] for k in s0 if isinstance(s0.get(k), int) and k != "layers"}
        r.update(arm=arm, policy=pol, server_pid=pid0,
                 live_policy=s1.get("insert_policy"),
                 **{f"d_{k}": v for k, v in d.items()})
        if "misses" in d and r["tokens"]:
            r["misses_per_token_incl_prefill"] = round(d["misses"] / r["tokens"], 2)
            tot = d["misses"] + d.get("hits", 0)
            r["hit_rate"] = round(d.get("hits", 0) / tot, 4) if tot else None
        if s1.get("insert_policy") not in (None, pol):
            r["invalid"] = f"live policy {s1.get('insert_policy')} != {pol}"
        if not a.dry_run and pids() != pid0:
            r["invalid"] = "server restarted during run"
        runs.append(r); print(json.dumps(r), flush=True)
        json.dump(dict(partial=True, runs=runs), open(a.out, "w"), indent=1)
    set_policy("lru")
    summ = {}
    for pol in ("lru", "bip", "dip"):
        ok = [r for r in runs if r["policy"] == pol and not r.get("invalid")
              and r["tokens"] >= a.max_tokens and r["finish_reason"] == "length"]
        tps = sorted(r["decode_tps"] for r in ok if r["decode_tps"])
        mpt = sorted(r["misses_per_token_incl_prefill"] for r in ok
                     if "misses_per_token_incl_prefill" in r)
        summ[pol] = dict(n=len(ok), tps=tps, tps_median=tps[len(tps) // 2] if tps else None,
                         miss_per_tok=mpt, miss_median=mpt[len(mpt) // 2] if mpt else None,
                         text_sha=sorted({r["text_sha"] for r in ok}))
    out = dict(kind="measured" if not a.dry_run else "DRY-RUN (mock, not silicon)",
               host=os.uname().nodename, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               prompt_sha256=FAMILY_SHA, max_tokens=a.max_tokens, plan=plan,
               warm=warm, runs=runs, summary=summ)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
