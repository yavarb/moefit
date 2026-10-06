"""PLE-trace capture: per-token PLE injection output + raw n-gram embedding.

Same class-patch strategy as trace.RouterTrace. Qwen4ExpPLELayer.__call__
returns the value injected into the residual (query-gated over n-gram
keys), and Qwen4ExpNGramEmbedding.__call__ returns the raw hashed-table
embedding (a pure function of token ids). Both are available in layer 1
before layer 1's MoE router and all later routers run, so they are
legitimate trigger inputs for speculative routing.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np


class PLETrace:
    def __init__(self, model):
        text = model.language_model.model
        self.blocks = {i: text.layers[i].ple for i in range(len(text.layers))
                       if "ple" in text.layers[i]}
        self.records: dict[str, list] = {"ple": [], "ngram": []}
        self.enabled = False
        self._saved = []

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, *a):
        self.remove()

    def install(self):
        if self._saved:
            return
        tr = self
        any_blk = next(iter(self.blocks.values()))
        cls = type(any_blk)
        emb_cls = type(any_blk.ple_embedding)
        orig_call = cls.__call__
        orig_emb = emb_cls.__call__

        def patched(self_blk, hidden_states, input_ids, cache, mask, **kw):
            out = orig_call(self_blk, hidden_states, input_ids, cache, mask, **kw)
            if tr.enabled:
                v = out[0] if isinstance(out, tuple) else out
                v = mx.reshape(v, (-1, v.shape[-1]))
                mx.eval(v)
                tr.records["ple"].append(
                    np.asarray(v.astype(mx.float16)))
            return out

        def patched_emb(self_emb, input_ids, cache):
            out = orig_emb(self_emb, input_ids, cache)
            if tr.enabled:
                v = mx.reshape(out, (-1, out.shape[-1]))
                mx.eval(v)
                tr.records["ngram"].append(
                    np.asarray(v.astype(mx.float16)))
            return out

        cls.__call__ = patched
        emb_cls.__call__ = patched_emb
        self._saved = [(cls, "__call__", orig_call),
                       (emb_cls, "__call__", orig_emb)]

    def remove(self):
        for cls, attr, orig in self._saved:
            setattr(cls, attr, orig)
        self._saved = []

    def reset(self):
        for r in self.records.values():
            r.clear()

    def stacked(self) -> dict[str, np.ndarray]:
        out = {}
        for name, recs in self.records.items():
            if recs:
                out[name] = np.concatenate(recs)
        return out
