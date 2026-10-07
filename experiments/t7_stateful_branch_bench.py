"""Execute state-restored routing replay branches in a causal CPU reference MoE.
Not attention KV, not production LLM throughput. No model or server loading.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moefit.routing_replay import RoutingReplayCache


class CausalBranch:
    """Reference integration: routing metadata and model state branch together."""
    def __init__(self, model, session, state=None):
        self.model, self.session = model, session
        self.state = np.zeros((8, 64), np.float32) if state is None else state.copy()
        self.routers = self.hits = self.experts = 0

    def fork(self):
        # Session validates the completed-token boundary before state cloning.
        return CausalBranch(self.model, self.session.fork(), self.state)

    def run(self, tokens):
        emb, router, expert = self.model
        out = []
        for token in tokens:
            self.session.begin_token(token)
            x = emb[token].copy()
            for layer in range(8):
                x = np.tanh(x + .2 * self.state[layer])
                def compute():
                    self.routers += 1
                    logits = np.einsum('h,he->e', x, router[layer], optimize=False)
                    ids = np.argsort(-logits)[:4].astype(np.int16)
                    w = np.exp(logits[ids] - np.max(logits[ids])); w /= w.sum()
                    return ids, w
                ids, w, hit = self.session.route(layer, compute)
                self.hits += hit
                vals = np.einsum('h,khj->kj', x, expert[layer, ids], optimize=False)
                x = np.tanh(x + np.sum(vals * w[:, None], axis=0))
                self.state[layer] = x
                self.experts += 1
            out.append(x.copy())
        return np.asarray(out)


def main():
    rng = np.random.default_rng(78)
    model = (rng.normal(0,.1,(512,64)).astype(np.float32),
             rng.normal(0,.1,(8,64,256)).astype(np.float32),
             rng.normal(0,.08,(8,256,64,64)).astype(np.float32))
    ns = b'causal-branch-v1/seed78/zero-initial-state/float32'
    prefix = list(range(48))
    suffixes = [list(range(48,80)), list(range(200,216))+list(range(64,80))]
    cache = RoutingReplayCache(compact_prefixes=True)
    root = CausalBranch(model, cache.layered_session(ns,8,deterministic=True))
    root.run(prefix)
    snapshot = root.state.copy()
    metadata_before = root.session.prefix
    references = []
    # Full independent execution establishes the oracle; no restored state used.
    for suffix in suffixes:
        fresh = CausalBranch(model, RoutingReplayCache().layered_session(ns,8))
        output = fresh.run(prefix + suffix)
        references.append((output[len(prefix):].copy(), fresh.state.copy()))
    # Populate only the first branch. The second shares neither continuation
    # prefix nor future cache entries, even when its suffix token IDs reconverge.
    seeded = root.fork()
    assert not np.shares_memory(root.state, seeded.state)
    seeded.run(suffixes[0])
    assert np.array_equal(root.state, snapshot)
    assert root.session.prefix == metadata_before
    # Preserve a fixed cache seed per trial so cold branch is genuinely cold.
    seed_entries = cache.entries.copy(); seed_bytes = cache.bytes

    def run(enabled):
        cache.entries = seed_entries.copy(); cache.bytes = seed_bytes
        start = time.perf_counter()
        parent = CausalBranch(model, cache.resume_layered(ns,8,prefix,deterministic=enabled), snapshot)
        outputs=[]; counters=[]
        for suffix in suffixes:
            child = parent.fork()
            outputs.append((child.run(suffix), child.state.copy()))
            counters.append(dict(hits=child.hits,routers=child.routers,experts=child.experts))
        seconds = time.perf_counter()-start
        assert np.array_equal(parent.state,snapshot)
        for actual, expected in zip(outputs,references):
            assert np.array_equal(actual[0],expected[0])
            assert np.array_equal(actual[1],expected[1])
        assert [c['experts'] for c in counters] == [256,256]
        assert [c['hits'] for c in counters] == ([256,0] if enabled else [0,0])
        return dict(seconds=seconds,branches=counters)

    run(False);run(True)
    trials=[]
    for i in range(5):
        row={}
        for mode in ([False,True] if i%2==0 else [True,False]):
            row['replay' if mode else 'recompute']=run(mode)
        trials.append(row)
    a=float(np.median([r['recompute']['seconds'] for r in trials]))
    b=float(np.median([r['replay']['seconds'] for r in trials]))
    result=dict(kind='MEASURED constructed causal CPU model, not LLM decode or attention KV',
        prefix_tokens=48,branches=2,continuation_tokens=64,layers=8,trials=trials,
        recompute_tokens_per_s=64/a,replay_tokens_per_s=64/b,speedup=a/b,
        routing_hit_rate=.5,outputs_and_final_states_bit_exact=True,
        caveat='Both arms restore the same real recurrent state snapshot; both pay snapshot cloning and route-wrapper costs. Prefix prefill/cache seeding excluded. All expert and recurrent updates retained. No production backend integration.')
    (ROOT/'results/t7_stateful_branch.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
