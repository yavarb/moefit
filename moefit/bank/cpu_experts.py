"""CPU execution of 4-bit affine experts with numpy (AMX via BLAS).

A token batch routed to experts: gather hidden states per expert, run
silu(gate)*up -> down, scale by routing score, scatter-add into the
aggregate. This is the coprocessor half of the mechanism.
"""
from __future__ import annotations
import numpy as np
from ..quant import dequant


def _silu(x):
    return x / (1.0 + np.exp(-x))


class CPUExpertBank:
    """Holds dequantized-on-demand experts for one layer.

    weights: tuple (w_u32, scale, bias) each [E,*,*] as in moefit.quant
    direction: gate_up weights are [E, 2*hidden, in]; down is [E, in, hidden]
    """

    def __init__(self, gate_up, down, hidden: int, n_experts: int,
                 group_size: int = 64):
        self.gu = gate_up
        self.dn = down
        self.hidden = hidden
        self.n_experts = n_experts
        self.group_size = group_size
        self._deq_cache = {}

    def _expert_weights(self, e: int):
        if e not in self._deq_cache:
            gs = self.group_size
            gu = dequant(self.gu[0][e:e+1], self.gu[1][e:e+1],
                         self.gu[2][e:e+1], group_size=gs)
            dn = dequant(self.dn[0][e:e+1], self.dn[1][e:e+1],
                         self.dn[2][e:e+1], group_size=gs)
            self._deq_cache[e] = (gu[0], dn[0])
            if len(self._deq_cache) > 32:      # bounded LRU-ish
                self._deq_cache.pop(next(iter(self._deq_cache)))
        return self._deq_cache[e]

    def execute(self, x: np.ndarray, indices: np.ndarray,
                scores: np.ndarray) -> np.ndarray:
        """x [T,in]; indices [T,k]; scores [T,k] -> aggregate [T,in]."""
        T, inn = x.shape
        out = np.zeros_like(x, dtype=np.float32)
        k = indices.shape[1]
        flat_idx = indices.reshape(-1)
        flat_sc = scores.reshape(-1)
        tok = np.repeat(np.arange(T), k)
        order = np.argsort(flat_idx, kind="stable")
        si = flat_idx[order]
        boundaries = np.flatnonzero(np.diff(si)) + 1
        for lo, hi in zip(np.r_[0, boundaries], np.r_[boundaries, len(si)]):
            e = si[lo]
            members = order[lo:hi]
            t = tok[members]
            gu, dn = self._expert_weights(int(e))
            xi = x[t].astype(np.float32)
            h = _silu(xi @ gu[:self.hidden].T) * (xi @ gu[self.hidden:].T)
            y = h @ dn.T
            y *= flat_sc[members][:, None]
            np.add.at(out, t, y)
        return out
