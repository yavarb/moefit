"""Decompose decode time on silicon by regressing ms/token on SSD MB/token
across workloads (no server restart, no config change).

Different prompts route to different experts, so the same oMLX server at the
same residency produces different miss traffic. Fitting
    ms_per_token = intercept + slope * disk_MB_per_token
over several prompts gives, from measurement alone:
    intercept : decode time that does not scale with misses (compute + sync + fixed)
    slope     : ms of decode per MB of physical expert reads (inverse effective
                *serialized* read+install bandwidth)
Uses measure_ssd_per_token.py (must sit next to this file).

usage: python3 regress_prompts.py --max-tokens 256 --out f.json
"""
import argparse, json, subprocess, sys
from pathlib import Path

PROMPTS = {
    "essay_uma": "Write a long, detailed essay on the history of unified memory architectures in personal computers, from early shared-memory designs to Apple Silicon. Cover trade-offs in bandwidth, latency and cost.",
    "code_lru": "Implement a thread-safe LRU cache in Rust with generics, full documentation comments, unit tests, and a benchmark harness. Explain each design decision inline.",
    "math_proof": "Prove carefully that there are infinitely many primes congruent to 3 mod 4, then generalize the argument as far as it goes and explain where it breaks for 1 mod 4.",
    "chinese_story": "用中文写一篇长篇小说的第一章：一个在上海经营旧书店的老人发现了一本会自己改写内容的日记。描写要细腻，人物对话要多。",
    "json_tool": "You are an API. Return a JSON array of 40 fictional customer records, each with id, name, email, signup_date, plan, monthly_spend, and a nested address object. Output only JSON.",
    "sql_explain": "Explain in depth how a PostgreSQL query planner chooses between nested loop, hash join and merge join, with worked EXPLAIN ANALYZE examples on a three-table schema you define.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    here = Path(__file__).resolve().parent
    pts = []
    for name, p in PROMPTS.items():
        tmp = f"/tmp/regress_{name}.json"
        subprocess.run([sys.executable, str(here / "measure_ssd_per_token.py"),
                        "--max-tokens", str(a.max_tokens), "--prompt", p, "--out", tmp],
                       check=True, stdout=subprocess.DEVNULL)
        r = json.load(open(tmp))
        pt = dict(prompt=name, tokens=r["tokens"], decode_tps=r["decode_tps"],
                  ms_per_token=round(1000 / r["decode_tps"], 2),
                  disk_MB_per_token=r["decode_disk_MB_per_token"],
                  disk_MBps=r["decode_disk_MBps"], KB_per_io=r["decode_avg_KB_per_io"])
        pts.append(pt)
        print(json.dumps(pt), flush=True)
    xs = [p["disk_MB_per_token"] for p in pts]
    ys = [p["ms_per_token"] for p in pts]
    n = len(xs); mx_ = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx_) ** 2 for x in xs); sxy = sum((x - mx_) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx; icpt = my - slope * mx_
    ss_res = sum((y - (icpt + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    fit = dict(intercept_ms=round(icpt, 2), slope_ms_per_MB=round(slope, 4),
               serialized_read_GBps=round(1 / slope, 2) if slope > 0 else None,
               r2=round(1 - ss_res / ss_tot, 3) if ss_tot else None, n=n)
    print("FIT", json.dumps(fit))
    Path(a.out).write_text(json.dumps(dict(kind="measured", points=pts, fit=fit,
                                           method=__doc__), indent=1))


if __name__ == "__main__":
    main()
