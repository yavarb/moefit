"""4-bit affine packing/unpacking for the CPU expert bank.

Canonical form (matches the checkpoint's oQ/oMLX-style affine layout in
shape, not necessarily bit-identical codes):

    value = q * scale + bias,  q in [0, 15] per nibble, group-wise s/b

Packing layout mirrors checkpoint tensors so the bank can be validated
against real shapes:
  weight  : uint32 [E, out, in//8]  (8 nibbles per u32, low nibble = lowest index)
  scales  : float32 [E, out, in//group_size]
  biases  : float32 [E, out, in//group_size]

tests/test_core.py verifies pack/dequant round-trip error bounds and
bank-vs-oracle equivalence on synthetic tensors. A later experiment
(tests/test_checkpoint_quant.py) compares our dequant path against
mlx_lm's GPU dequant on real checkpoint tensors.
"""
from __future__ import annotations
import numpy as np


def pack(x: np.ndarray, group_size: int = 64, bits: int = 4):
    """Quantize [E, out, inn] fp32 -> (w_u32, scale, bias)."""
    E, o, i = x.shape
    assert i % group_size == 0 and i % 8 == 0
    qmax = (1 << bits) - 1
    g = i // group_size
    xb = x.reshape(E, o, g, group_size)
    lo = xb.min(axis=-1)
    hi = xb.max(axis=-1)
    scale = (hi - lo) / qmax
    scale = np.where(scale == 0, 1.0, scale)
    bias = lo
    q = np.clip(np.round((xb - bias[..., None]) / scale[..., None]),
                0, qmax).astype(np.uint8).reshape(E, o, i)
    words = np.zeros((E, o, i // 8), dtype=np.uint32)
    for k in range(8):
        words |= q[..., k::8].astype(np.uint32) << (4 * k)
    return words, scale.astype(np.float32), bias.astype(np.float32)


def dequant(words: np.ndarray, scale: np.ndarray, bias: np.ndarray,
            group_size: int = 64, bits: int = 4) -> np.ndarray:
    E, o, iw = words.shape
    i = iw * 8
    nib = np.zeros((E, o, i), dtype=np.uint8)
    for k in range(8):
        nib[..., k::8] = ((words >> (4 * k)) & 0xF).astype(np.uint8)
    s = np.repeat(scale, group_size, axis=-1).astype(np.float32)
    b = np.repeat(bias, group_size, axis=-1).astype(np.float32)
    return nib.astype(np.float32) * s + b


def expert_bytes(E: int, hidden: int, intermediate: int,
                 group_size: int = 64) -> int:
    """Approx bank bytes for one layer: switch-MLP experts only.

    gate_up: [E, 2*hidden, intermediate] ; down: [E, intermediate, hidden]
    4 bits/weight + fp16-scale + fp16-bias per group.
    """
    w_elems = E * (2 * hidden * intermediate + intermediate * hidden)
    groups = w_elems // group_size
    return w_elems // 2 + groups * 4
