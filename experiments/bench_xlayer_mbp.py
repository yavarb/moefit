"""MEASURED A/B of T1 guarded cross-layer expert fetch inside the real oMLX
ExpertCache (patches/omlx_t1_xlayer.py), on the MBP (not Santa Cruz).

Setup, per arm:
  * real checkpoint, oMLX 0.7.0 offload code (stock d420b305 + T2 sidecar
    counters + T3 LIP (off) + T1 xlayer hunk), residency 0.28 (cap 143)
  * F_NOCACHE on every shard fd + msync(MS_INVALIDATE) of all shards before
    each arm  ->  every miss is a physical SSD read (Santa Cruz's page cache
    held only 2.3 GB of experts, so this is the faithful regime)
  * all 48 ExpertCaches reset to empty before each arm (identical cold start)
  * same prompt, greedy, n tokens; arms alternate OFF/ON/OFF/ON...
Reports decode tok/s (first->last token wall clock), logical hits/misses per
token, xl issued/used/ready/wasted, and checks output text identity.

usage: PYTHONPATH=/Users/yb/projects/omlx-ref? (not needed) \
  ~/.hermes/cache/scratch/omlxenv/bin/python3 experiments/bench_xlayer_mbp.py \
     --pairs 3 --max-tokens 512 --out results/t1_xlayer_mbp_ab.json
"""
import argparse, ctypes, ctypes.util, fcntl, gc, importlib.util, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL = Path(os.path.expanduser("~/.lmstudio/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"))
PATCH = ROOT / "patches" / "omlx_t1_xlayer.py"
F_NOCACHE = 48

libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                      ctypes.c_int, ctypes.c_longlong]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]


def drop_ubc(files):
    for f in files:
        fd = os.open(f, os.O_RDONLY)
        n = os.fstat(fd).st_size
        p = libc.mmap(None, n, 1, 1, fd, 0)
        libc.msync(p, n, 2)  # MS_INVALIDATE
        libc.munmap(p, n)
        os.close(fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--residency", type=float, default=0.28)
    ap.add_argument("--prompt-id", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import omlx.patches  # noqa: F401  (package context for relative imports)
    spec = importlib.util.spec_from_file_location("omlx.patches.moe_t1_xlayer", PATCH)
    off = importlib.util.module_from_spec(spec)
    off.__package__ = "omlx.patches"
    sys.modules[spec.name] = off
    spec.loader.exec_module(off)
    off._XL_STATE["force"] = False

    from moefit.loader import load
    import mlx.core as mx
    from mlx_vlm.generate import stream_generate

    t0 = time.time()
    Ld = load(str(MODEL))
    wrapped = off.apply_moe_expert_offload(Ld.model, MODEL, a.residency)
    gc.collect(); mx.clear_cache()
    caches = list(off._ALL_CACHES)
    links = sum(c._xl_target is not None for c in caches)
    print(f"loaded+wrapped {time.time()-t0:.0f}s wrapped={wrapped} caches={len(caches)} "
          f"cap={caches[0].capacity} xl_links={links}", flush=True)
    assert wrapped == 48 and links == 46, (wrapped, links)
    store = caches[0].disk._store
    for fd in store._fds.values():
        fcntl.fcntl(fd, F_NOCACHE, 1)
    shards = sorted(MODEL.glob("*.safetensors"))

    prompts = [json.loads(l) for l in (ROOT / "data" / "prompts.jsonl").read_text().splitlines() if l.strip()]
    hold = [p for p in prompts if p["split"] == "holdout"]
    item = next((p for p in hold if p.get("id") == a.prompt_id), hold[0])
    tok = Ld.tokenizer
    render = tok.apply_chat_template([{"role": "user", "content": item["text"]}],
                                     add_generation_prompt=True, tokenize=False)

    keys = ("hits", "misses", "xl_issued", "xl_used", "xl_ready", "xl_wasted")

    def snap():
        return {k: sum(getattr(c, k) for c in caches) for k in keys}

    def reset():
        for c in caches:
            c.slot_of.clear(); c.free = list(range(c.capacity))
            c.map = mx.full((c.n_experts,), -1, dtype=mx.int32)
            c.warm = False; c._xl_pending = {}; c._sc_pending = {}

    def arm(on):
        reset(); gc.collect(); mx.clear_cache(); drop_ubc(shards)
        off._XL_STATE["force"] = on
        stamps, text, s0, s_first = [], [], None, None
        tstart = time.perf_counter()
        for resp in stream_generate(Ld.model, Ld.processor, render,
                                    max_tokens=a.max_tokens, temperature=0.0):
            now = time.perf_counter()
            if s_first is None:
                s_first = snap()  # counters after prefill + first token
            stamps.append(now); text.append(resp.text)
        s1 = snap()
        n = len(stamps) - 1
        d = {k: s1[k] - s_first[k] for k in keys}
        r = dict(arm="ON" if on else "OFF", decode_tokens=n,
                 ttft_s=round(stamps[0] - tstart, 2),
                 decode_tps=round(n / (stamps[-1] - stamps[0]), 3),
                 miss_per_tok=round(d["misses"] / n, 2),
                 hit_per_tok=round(d["hits"] / n, 2),
                 xl_issued_per_tok=round(d["xl_issued"] / n, 2),
                 xl_used_per_tok=round(d["xl_used"] / n, 2),
                 xl_ready_per_tok=round(d["xl_ready"] / n, 2),
                 xl_wasted_per_tok=round(d["xl_wasted"] / n, 2),
                 xl_precision=round(d["xl_used"] / max(d["xl_issued"], 1), 3),
                 text_hash=hash("".join(text)) & 0xffffffff)
        print(json.dumps(r), flush=True)
        return r, "".join(text)

    rows, texts = [], set()
    with open("/tmp/moefit-lab/state/local_ssd_bench.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        arm(False)  # warm-up (JIT/kernels), discarded
        for _ in range(a.pairs):
            for on in (False, True):
                r, t = arm(on); rows.append(r); texts.add(t)
    import statistics as st
    on = [r["decode_tps"] for r in rows if r["arm"] == "ON"]
    of = [r["decode_tps"] for r in rows if r["arm"] == "OFF"]
    res = dict(kind="MEASURED on MBP (M-series, 128 GB) with F_NOCACHE + UBC drop; NOT Santa Cruz",
               prompt=item.get("id"), residency=a.residency, cap=caches[0].capacity,
               max_tokens=a.max_tokens, d=off._XL_D, tau=off._XL_TAU, max_per_layer=off._XL_MAX,
               rows=rows, median_off=st.median(of), median_on=st.median(on),
               gain=round(st.median(on) / st.median(of) - 1, 4),
               paired_deltas=[round(rows[i + 1]["decode_tps"] - rows[i]["decode_tps"], 3)
                              for i in range(0, len(rows), 2)],
               outputs_identical=len(texts) == 1)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("median_off", "median_on", "gain", "paired_deltas", "outputs_identical")}))


if __name__ == "__main__":
    main()
