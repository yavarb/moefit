"""Hashed n-gram -> per-layer expert candidate tables.

Build: consume routing traces (token_ids, per-layer top-k idx/scores).
Query: at decode time, key = tuple of last n token IDs (current last).
Returns per-layer candidate expert sets to speculate/prefetch.

Convention: trace row t is the routers decision for the token AT position
t (its hidden state entered the router), so the key at row t is
token_ids[: t+1][-n:] — current token plus n-1 history, the same shape of
context the models own PLE n-gram table consumes.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np


def ngram_key(ids, n: int, pad_id: int = 0) -> tuple:
    ids = list(ids)
    if len(ids) < n:
        ids = [pad_id] * (n - len(ids)) + ids
    return tuple(int(t) for t in ids[-n:])


class NgramExpertTable:
    def __init__(self, n: int = 3, cand_budget: int = 14):
        self.n = n
        self.cand_budget = cand_budget
        # (key, layer) -> {expert_id: [count, score_sum]}
        self.tab: dict[tuple, dict[int, list]] = defaultdict(
            lambda: defaultdict(lambda: [0, 0.0]))
        self.n_seen = 0
        self.cand: dict = {}

    def observe(self, layer: int, token_ids, idx, scores):
        """idx/scores: [T,k] arrays for this layer over one sequence."""
        for t in range(idx.shape[0]):
            key = ngram_key(token_ids[: t + 1], self.n)
            row = self.tab[(key, layer)]
            for e, s in zip(idx[t], scores[t]):
                cell = row[int(e)]
                cell[0] += 1
                cell[1] += float(s)
        self.n_seen += int(idx.shape[0])

    def finalize(self):
        """Rank experts per (key, layer) by (count, score_sum)."""
        self.cand = {}
        for (key, layer), experts in self.tab.items():
            ranked = sorted(experts.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))
            self.cand.setdefault(key, {})[layer] = (
                np.array([e for e, _ in ranked], dtype=np.int32),
                np.array([c for _, c in ranked], dtype=np.int32))
        return self

    def candidates(self, key: tuple, layer: int):
        d = self.cand.get(key)
        if not d or layer not in d:
            return None
        experts, counts = d[layer]
        return experts[: self.cand_budget], counts[: self.cand_budget]

    def coverage(self, eval_records):
        """eval_records: list of (token_ids, layer, idx[T,k]).

        coverage: mean fraction of true top-k contained in candidates.
        exact: fraction of positions where true top-k == candidate set
               truncated to k, ordered set equality (set semantics).
        seen: fraction of query positions whose n-gram key was observed
              during build (cold-start rate = 1 - seen).
        """
        rec = exact = seen = tot = 0
        for token_ids, layer, idx in eval_records:
            k = idx.shape[1]
            for t in range(idx.shape[0]):
                tot += 1
                key = ngram_key(token_ids[: t + 1], self.n)
                if key in self.cand and layer in self.cand[key]:
                    seen += 1
                c = self.candidates(key, layer)
                if c is None:
                    continue
                cand = set(c[0].tolist())
                gs = set(idx[t].tolist())
                rec += len(gs & cand) / len(gs)
                exact += int(gs == set(c[0][:k].tolist()))
        if not tot:
            return {"tokens": 0}
        return {"tokens": tot,
                "coverage": rec / tot,
                "exact_topk": exact / tot,
                "key_seen": seen / tot,
                "n_keys": len(self.cand),
                "n_seen_tokens": self.n_seen}
