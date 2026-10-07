"""Exact-prefix routing memoization for deterministic, context-isolated inference.

Stores routing metadata, NOT expert outputs or KV. Integration must supply a
namespace covering weights/adapters, tokenizer, positional/attention settings,
precision, execution policy, tenant and any initial hidden/KV state. Disable
reuse when batch-dependent/stochastic routing can change results. Not thread safe.
"""
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Routes:
    indices: np.ndarray
    weights: np.ndarray

    @classmethod
    def freeze(cls, indices, weights):
        i = np.asarray(indices)
        w = np.asarray(weights)
        if i.shape != w.shape or i.ndim != 2:
            raise ValueError('expected equal (layers, top_k) shapes')
        if not np.issubdtype(i.dtype, np.integer) or (i < 0).any():
            raise ValueError('invalid expert ids')
        if not np.isfinite(w).all():
            raise ValueError('nonfinite route weights')
        # Immutable bytes ownership prevents callers re-enabling write flags.
        return cls(np.frombuffer(i.tobytes(), dtype=i.dtype).reshape(i.shape),
                   np.frombuffer(w.tobytes(), dtype=w.dtype).reshape(w.shape))


class RoutingReplayCache:
    def __init__(self, max_bytes=16*1024*1024):
        if max_bytes <= 0:
            raise ValueError('max_bytes must be positive')
        self.max_bytes = max_bytes
        self.bytes = 0
        self.entries = OrderedDict()
        self.hits = self.misses = self.evictions = 0

    def session(self, namespace, *, deterministic=False):
        if not isinstance(namespace, bytes) or not namespace:
            raise ValueError('nonempty opaque context namespace required')
        return ReplaySession(self, namespace, deterministic)

    def clear(self):
        self.entries.clear()
        self.bytes = 0

    def _resolve(self, key, compute, enabled):
        if enabled and key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key][0], True
        self.misses += 1
        indices, weights = compute()
        value = Routes.freeze(indices, weights)
        if enabled:
            # Conservative accounting for Python tuple/int key overhead;
            # payload capacity is bounded, not a process RSS guarantee.
            size = value.indices.nbytes + value.weights.nbytes + len(key[0]) + 128 + 40*len(key[1])
            if size <= self.max_bytes:
                while self.bytes + size > self.max_bytes:
                    _, (_, removed) = self.entries.popitem(last=False)
                    self.bytes -= removed
                    self.evictions += 1
                self.entries[key] = value, size
                self.bytes += size
        return value, False


class ReplaySession:
    def __init__(self, cache, namespace, enabled):
        self.cache, self.namespace, self.enabled = cache, namespace, enabled
        self.prefix = ()

    def step(self, token_id, compute):
        """Consume actual input token; callback returns all-layer ids/weights.

        A differing token invalidates reuse for all subsequent positions, even
        if a later suffix matches. Callback runs only on miss. Caller still
        executes expert compute and updates KV on hits.
        """
        if not isinstance(token_id, (int, np.integer)) or token_id < 0:
            raise ValueError('token id must be a nonnegative integer')
        prefix = self.prefix + (int(token_id),)
        value, hit = self.cache._resolve((self.namespace, prefix), compute, self.enabled)
        self.prefix = prefix  # callback failure does not advance session
        return value, hit
