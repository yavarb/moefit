"""Real baseline numbers on this Mac: decode tps and prefill latency."""
import os, sys, time
os.environ.setdefault("MLX_MAX_WIRED_LIMIT", str(int(115e9)))
sys.path.insert(0, ".")
from specexp.loader import load
import mlx.core as mx
from mlx_vlm.generate import stream_generate

L = load()
tok = L.tokenizer
prompt = "Write a detailed 400-word essay on the history of unified memory architectures."

# warm
for _ in stream_generate(L.model, L.processor, "Hi", max_tokens=2, temperature=0.0):
    pass

# decode rate (small prompt -> mostly decode)
t0 = time.perf_counter(); n = 0; ttft = None
for r in stream_generate(L.model, L.processor, prompt, max_tokens=96, temperature=0.0):
    if ttft is None:
        ttft = time.perf_counter() - t0
    n += 1
dt = time.perf_counter() - t0
print(f"decode: {n} tokens in {dt:.1f}s = {(n-1)/(dt-ttft):.1f} tok/s; ttft={ttft*1000:.0f}ms")
print(f"peak mem {mx.get_peak_memory()/2**30:.0f} GiB")

# prefill cost at scale: pad prompt to ~4k tokens
long_prompt = (prompt + " ") * 28
ids = tok.encode(long_prompt, add_special_tokens=False)
t0 = time.perf_counter()
for r in stream_generate(L.model, L.processor, long_prompt, max_tokens=1, temperature=0.0):
    pass
pf = time.perf_counter() - t0
print(f"prefill {len(ids)} tokens in {pf:.1f}s = {len(ids)/pf:.0f} tok/s prefill")
