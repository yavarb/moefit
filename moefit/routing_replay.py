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
    def __init__(self, max_bytes=16*1024*1024, *, compact_prefixes=False):
        if max_bytes <= 0:
            raise ValueError('max_bytes must be positive')
        self.max_bytes = max_bytes
        self.compact_prefixes = compact_prefixes
        self.bytes = 0
        self.entries = OrderedDict()
        self.hits = self.misses = self.evictions = 0

    def session(self, namespace, *, deterministic=False):
        if not isinstance(namespace, bytes) or not namespace:
            raise ValueError('nonempty opaque context namespace required')
        return ReplaySession(self, namespace, deterministic)

    def layered_session(self, namespace, layers, *, deterministic=False):
        """Route each layer lazily; never skip expert/KV execution on hits."""
        if not isinstance(namespace, bytes) or not namespace or layers <= 0:
            raise ValueError('namespace and positive layer count required')
        return LayeredReplaySession(self, namespace, layers, deterministic)

    def resume_layered(self, namespace, layers, prefix_tokens, *, deterministic=False):
        """Resume AFTER a backend-restored prefix, without replaying its routes.

        Caller must restore exactly matching hidden/KV state and use the same
        namespace. Token equality cannot verify backend state. No cache entry
        is admitted, refreshed or consumed by this operation.
        """
        session = self.layered_session(namespace, layers, deterministic=deterministic)
        tokens = tuple(prefix_tokens)
        for token in tokens:
            if not isinstance(token, (int, np.integer)) or token < 0:
                raise ValueError('invalid prefix token')
            if self.compact_prefixes and token > 0xffffffff:
                raise ValueError('compact token ids must fit uint32')
        session.prefix = (np.asarray(tokens, dtype='<u4').tobytes()
                          if self.compact_prefixes else tuple(map(int, tokens)))
        return session

    def _empty_prefix(self):
        return b'' if self.compact_prefixes else ()

    def _extend_prefix(self, prefix, token_id):
        if not isinstance(token_id, (int, np.integer)) or token_id < 0:
            raise ValueError('invalid token id')
        if self.compact_prefixes:
            if token_id > 0xffffffff:
                raise ValueError('compact token ids must fit uint32')
            return prefix + int(token_id).to_bytes(4, 'little')
        return prefix + (int(token_id),)

    def clear(self):
        self.entries.clear()
        self.bytes = 0

    def invalidate_namespace(self, namespace):
        """Remove cached routes for exactly one context; return removed count.

        O(number of entries), not thread safe. Does not revoke arrays already
        returned or reset sessions/KV. Stop affected execution and use a NEW
        namespace when model/context changes; old live sessions can repopulate
        their old namespace. Hit/miss/eviction counters are unchanged.
        """
        if not isinstance(namespace, bytes) or not namespace:
            raise ValueError('nonempty opaque context namespace required')
        keys = [key for key in self.entries
                if (key[0] if isinstance(key[0], bytes) else key[0][0]) == namespace]
        for key in keys:
            _, size = self.entries.pop(key)
            self.bytes -= size
        return len(keys)

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
            namespace_size = len(key[0]) if isinstance(key[0], bytes) else len(key[0][0]) + 96
            prefix_size = len(key[1]) if isinstance(key[1], bytes) else 40*len(key[1])
            size = value.indices.nbytes + value.weights.nbytes + namespace_size + 128 + prefix_size
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
        self.prefix = cache._empty_prefix()

    def step(self, token_id, compute):
        """Consume actual input token; callback returns all-layer ids/weights.

        A differing token invalidates reuse for all subsequent positions, even
        if a later suffix matches. Callback runs only on miss. Caller still
        executes expert compute and updates KV on hits.
        """
        if not isinstance(token_id, (int, np.integer)) or token_id < 0:
            raise ValueError('token id must be a nonnegative integer')
        prefix = self.cache._extend_prefix(self.prefix, token_id)
        value, hit = self.cache._resolve((self.namespace, prefix), compute, self.enabled)
        self.prefix = prefix  # callback failure does not advance session
        return value, hit


class LayeredReplaySession:
    """Per-token transaction: layers must run in order before next token.

    On a layer failure discard the session AND roll back caller hidden/KV
    state. Completed deterministic layer entries may safely remain cached.
    Separate tagged namespaces prevent collision with whole-token entries.
    """
    def __init__(self, cache, namespace, layers, enabled):
        self.cache, self.namespace = cache, namespace
        self.layers, self.enabled = layers, enabled
        self.prefix = cache._empty_prefix()
        self.next_layer = layers

    def fork(self):
        """O(1) metadata branch at a completed token; caller forks KV separately."""
        if self.next_layer != self.layers:
            raise RuntimeError('cannot fork a partially executed token')
        child = LayeredReplaySession(self.cache, self.namespace, self.layers, self.enabled)
        child.prefix = self.prefix
        return child

    def begin_token(self, token_id):
        if self.next_layer != self.layers:
            raise RuntimeError('previous token has unfinished layers')
        if not isinstance(token_id, (int, np.integer)) or token_id < 0:
            raise ValueError('invalid token id')
        self.prefix = self.cache._extend_prefix(self.prefix, token_id)
        self.next_layer = 0

    def plan_reads(self, resident, expert_bytes, max_bytes):
        """Return (layer, expert, bytes) for cached CURRENT-prefix routes.

        Pure metadata: no IO, installs, next-token prediction, or LRU refresh.
        Caller must recheck residency before issuing IO and serialize against
        demand. Available only immediately after begin_token, before layer0.
        expert_bytes(layer, expert) supplies actual payload size. Budget is
        hard; known nonresident experts are selected in layer/route order.
        """
        if self.next_layer != 0:
            raise RuntimeError('plan before executing layer0')
        if max_bytes < 0:
            raise ValueError('negative byte budget')
        if not self.enabled or max_bytes == 0:
            return ()
        plan = []
        used = 0
        for layer in range(self.layers):
            key = ((self.namespace, self.layers, layer), self.prefix)
            entry = self.cache.entries.get(key)
            if entry is None:
                continue
            seen = set()
            for raw in entry[0].indices[0]:
                expert = int(raw)
                if expert in seen or expert in resident.get(layer, ()):
                    continue
                seen.add(expert)
                size = expert_bytes(layer, expert)
                if not isinstance(size, (int, np.integer)) or size <= 0:
                    raise ValueError('expert byte size must be positive integer')
                if used + size <= max_bytes:
                    plan.append((layer, expert, int(size)))
                    used += size
                    if used == max_bytes:
                        return tuple(plan)
        return tuple(plan)

    def route(self, layer, compute):
        if layer != self.next_layer or layer >= self.layers:
            raise RuntimeError('layers must execute once in ascending order')
        # tuple tag cannot equal the bytes namespace of whole-token entries.
        key = ((self.namespace, self.layers, int(layer)), self.prefix)
        # Hits need no callback adapter or repeated membership/value lookup.
        entry = self.cache.entries.get(key) if self.enabled else None
        if entry is not None:
            self.cache.hits += 1
            self.cache.entries.move_to_end(key)
            self.next_layer += 1
            value = entry[0]
            return value.indices[0], value.weights[0], True
        def wrapped():
            indices, weights = compute()
            return np.asarray(indices)[None, :], np.asarray(weights)[None, :]
        value, hit = self.cache._resolve(key, wrapped, self.enabled)
        self.next_layer += 1
        return value.indices[0], value.weights[0], hit
