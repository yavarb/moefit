"""Determinism test for routing-trace replay (prefix-cache sidecar premise).

Same prompt, greedy decode, traced twice in one process: are per-layer
router decisions bit-identical run to run? If yes, a routing sidecar
stored next to the KV cache is exact, not approximate.
"""
import json, sys
from pathlib import Path
import numpy as np

import os
os.environ.setdefault("MLX_MAX_WIRED_LIMIT", str(int(115e9)))
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moefit.loader import load
from moefit.trace import RouterTrace

prompts = [json.loads(l) for l in
           (Path(__file__).resolve().parents[1] / "data/prompts.jsonl").read_text().splitlines() if l.strip()]
p = prompts[0]["text"]
L = load()
tok = L.tokenizer
try:
    render = tok.apply_chat_template([{"role": "user", "content": p}],
                                     add_generation_prompt=True, tokenize=False)
except TypeError:
    render = tok.apply_chat_template([{"role": "user", "content": p}],
                                     add_generation_prompt=True)
from mlx_vlm.generate import stream_generate

tr = RouterTrace(L.model, L.expert_layers)
tr.install()
runs = []
for r in range(2):
    tr.reset(); tr.enabled = True
    try:
        for _ in stream_generate(L.model, L.processor, render,
                                 max_tokens=32, temperature=0.0):
            pass
    finally:
        tr.enabled = False
    runs.append(tr.stacked())

same_rows = tot_rows = 0
exact_layers = 0
for li in L.expert_layers:
    (i0, _), (i1, _) = runs[0][li], runs[1][li]
    n = min(len(i0), len(i1))
    eq = (i0[:n] == i1[:n]).all(axis=1)
    same_rows += int(eq.sum()); tot_rows += n
    exact_layers += int(eq.all())
print(f"rows identical: {same_rows}/{tot_rows}; layers fully exact: {exact_layers}/48")
