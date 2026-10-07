"""Router-trace capture for mlx_lm Qwen3-Next-style MoE blocks.

Patches the SparseMoeBlock CLASS (Python special-method lookup bypasses
instance attributes) with a registry mapping module identity -> layer idx.
Router math mirrors mlx_lm qwen3_next.Qwen3NextSparseMoeBlock exactly:
softmax(precise) -> argpartition top-k -> renormalized scores. The wrapper
recomputes the router decision from block.gate(x) with the same ops, then
delegates to the original __call__ — captured indices/scores ARE the
block's real routing because gate weights and math are identical.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np


class RouterTrace:
    def __init__(self, model, expert_layers: list[int]):
        text = model.language_model.model
        self.layers = {i: text.layers[i].mlp for i in expert_layers}
        self.expert_layers = list(expert_layers)
        self.records: dict[int, list] = {i: [] for i in expert_layers}
        self.enabled = False
        self._cls = None
        self._orig = None
        self._ids = {}

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, *a):
        self.remove()

    def install(self):
        if self._cls is not None:
            return
        tr = self
        sample = next(iter(self.layers.values()))
        cls = type(sample)
        tr._ids = {id(blk): li for li, blk in self.layers.items()}
        orig = cls.__call__
        self._cls, self._orig = cls, orig

        def patched(self_blk, x, *a, **kw):
            li = tr._ids.get(id(self_blk))
            if li is not None and tr.enabled:
                gates = mx.softmax(self_blk.gate(x), axis=-1, precise=True)
                k = self_blk.top_k
                inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
                scores = mx.take_along_axis(gates, inds, axis=-1)
                if getattr(self_blk, "norm_topk_prob", True):
                    scores = scores / scores.sum(axis=-1, keepdims=True)
                mx.eval(inds, scores)
                tr.records[li].append(
                    (np.asarray(inds), np.asarray(scores.astype(mx.float32)))
                )
            return orig(self_blk, x, *a, **kw)

        cls.__call__ = patched

    def remove(self):
        if self._cls is not None:
            self._cls.__call__ = self._orig
            self._cls, self._orig, self._ids = None, None, {}

    def reset(self):
        for r in self.records.values():
            r.clear()

    def stacked(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """layer -> (idx [T,k] int16, scores [T,k] float16)."""
        out = {}
        for li, recs in self.records.items():
            if not recs:
                continue
            idx = np.concatenate([r[0].reshape(-1, r[0].shape[-1]) for r in recs])
            sc = np.concatenate([r[1].reshape(-1, r[1].shape[-1]) for r in recs])
            out[li] = (idx.astype(np.int16), sc.astype(np.float16))
        return out
