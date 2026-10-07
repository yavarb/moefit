"""Collect an instrumented silicon decode run from an omlx/OpenAI-style
chat-completions endpoint and emit the canonical measured blob.

Plain stdlib (urllib) so it can run on Santa Cruz without pip installs.

Per-token instrumentation: every streamed chunk gets a monotonic
timestamp; per-token gaps are derived by dividing each inter-chunk gap
by the tokens the chunk carried (from stream_options
include_usage / chunk content len). The output blob is the same shape
as results/measured_santa_cruz_36gb.json PLUS per-run `per_token_ms` /
`chunk_ts_s` / `chunk_tokens`, which moefit/metrics.silicon_record_from_
measured turns into p50/p90/p95/p99 tok-gap stats.

usage:
  python3 collect_silicon_run.py --url http://127.0.0.1:8000/v1/chat/completions \
      --model Qwen3.8-Flash-Next-oQ4e-mtp --max-tokens 128 --runs 3 \
      --out results/measured_<label>.json --label "santacruz 36GB idle"
NOTE: decode_tps from chunk counting can undercount when the server
coalesces chunks; usage.completion_tokens (requested via
stream_options) is authoritative when present.
"""
import argparse, json, time, urllib.request
from pathlib import Path


def one_run(url, model, prompt, max_tokens, timeout):
    body = json.dumps(dict(
        model=model,
        messages=[dict(role="user", content=prompt)],
        max_tokens=max_tokens, temperature=0.0, stream=True,
        stream_options=dict(include_usage=True))).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    chunk_ts, chunk_tokens, n_text_chunks = [], [], 0
    completion_tokens = usage_prompt = None
    finish_reason = None
    ttft = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"":
                break
            try:
                j = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if "usage" in j and j.get("usage"):
                completion_tokens = j["usage"].get("completion_tokens")
                usage_prompt = j["usage"].get("prompt_tokens")
                continue
            ch = j.get("choices") or [{}]
            delta = (ch[0].get("delta") or {})
            # thinking models stream reasoning_content; count it as text
            txt = delta.get("content") or delta.get("reasoning_content") or ""
            n_tok = 1 if txt else 0   # omlx streams ~1 token/chunk
            now = time.monotonic()
            if txt and ttft is None:
                ttft = now - t0
            n_text_chunks += 1 if txt else 0
            chunk_ts.append(now)
            chunk_tokens.append(n_tok)
            if ch[0].get("finish_reason"):
                finish_reason = ch[0]["finish_reason"]
    wall = time.monotonic() - t0
    # per-token gaps: divide inter-chunk time by tokens in the LATER chunk
    per_token_ms = [
        1000.0 * (b - a) / max(t, 1)
        for a, b, t in zip(chunk_ts, chunk_ts[1:], chunk_tokens[1:])]
    tokens = completion_tokens if completion_tokens else n_text_chunks
    decode_s = (wall - ttft) if ttft else wall
    return dict(
        tokens=tokens,
        count_source=("usage.completion_tokens" if completion_tokens
                      else "streamed_deltas"),
        streamed_deltas=n_text_chunks,
        ttft_s=round(ttft, 4) if ttft else None,
        decode_s=round(decode_s, 3),
        decode_tps=round((tokens - 1) / decode_s, 2) if decode_s > 0
        and tokens > 1 else None,
        wall_s=round(wall, 3),
        finish_reason=finish_reason,
        prompt_tokens=usage_prompt,
        chunk_ts_s=[round(t - t0, 4) for t in chunk_ts],
        chunk_tokens=chunk_tokens,
        per_token_ms=[round(g, 3) for g in per_token_ms],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--label", default="silicon measured")
    ap.add_argument("--host", default=None)
    ap.add_argument("--ram-gib", type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    prompt = ("Write a long, detailed essay on the history of unified "
              "memory architectures in personal computers, from early "
              "shared-memory designs to modern unified memory fabrics.")

    import subprocess
    commit = None
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[1]
                                ).stdout.strip() or None
    except Exception:
        pass

    print("warmup...", flush=True)
    one_run(a.url, a.model, "Hi", 8, a.timeout)
    runs = []
    for i in range(a.runs):
        r = one_run(a.url, a.model, prompt, a.max_tokens, a.timeout)
        runs.append(r)
        print(f"run {i}: tps={r['decode_tps']} tokens={r['tokens']}"
              f" ttft={r['ttft_s']}s", flush=True)

    tpss = [r["decode_tps"] for r in runs if r["decode_tps"]]
    tpss_sorted = sorted(tpss)
    n = len(tpss_sorted)
    median = (tpss_sorted[n // 2] if n % 2
              else (tpss_sorted[n // 2 - 1] + tpss_sorted[n // 2]) / 2)
    ttfts = sorted(r["ttft_s"] for r in runs if r["ttft_s"])
    blob = dict(
        label=a.label, kind="measured",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        host=a.host, ram_gib=a.ram_gib, model=a.model,
        max_tokens=a.max_tokens, runs=runs,
        decode_tps_median=round(median, 2),
        decode_tps_min=round(min(tpss), 2),
        decode_tps_max=round(max(tpss), 2),
        ttft_s_median=ttfts[len(ttfts) // 2] if ttfts else None,
        method=("greedy streaming chat completion; decode_tps = "
                "(tokens-1)/(first->last streamed token); per_token_ms "
                "from monotonic chunk timestamps"),
        git_commit=commit,
    )
    Path(a.out).write_text(json.dumps(blob, indent=1) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
