"""Unit tests: n-gram table semantics and 4-bit affine round-trip.

No GPU/model needed. Run: .venv/bin/python -m pytest tests -q
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moefit.trigger.ngram_table import NgramExpertTable, ngram_key
from moefit.quant import pack, dequant
from moefit.bank.cpu_experts import CPUExpertBank


def test_ngram_key_pads_and_windows():
    assert ngram_key([5, 6, 7, 8], 3) == (6, 7, 8)
    assert ngram_key([7], 3) == (0, 0, 7)
    assert ngram_key([], 2) == (0, 0)


def test_table_learns_and_hits():
    tab = NgramExpertTable(n=3, cand_budget=10)
    ids = list(range(100))
    # positions 10..40 share key tail (38,39,x)? construct repeating context
    idx = np.tile(np.arange(10), (len(ids), 1))  # every token routes to 0..9
    sc = np.full((len(ids), 10), 0.1, dtype=np.float32)
    tab.observe(layer=5, token_ids=ids, idx=idx, scores=sc)
    tab.finalize()
    key = ngram_key(ids[:20 + 1], 3)
    cands = tab.candidates(key, 5)
    assert cands is not None
    assert set(range(10)) == set(cands[0][:10].tolist())
    ev = tab.coverage([(ids, 5, idx)])
    assert ev["coverage"] == 1.0
    assert ev["exact_topk"] == 1.0


def test_table_miss_is_graceful():
    tab = NgramExpertTable(n=3, cand_budget=4)
    tab.observe(0, [1, 2, 3], np.array([[1, 2]]), np.array([[0.5, 0.5]]))
    tab.finalize()
    assert tab.candidates((9, 9, 9), 0) is None
    ev = tab.coverage([([9, 9, 9], 0, np.array([[7, 8]]))])
    assert ev["coverage"] == 0.0
    assert ev["key_seen"] == 0.0


def test_quant_roundtrip_error_bounded():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.02, size=(2, 16, 64)).astype(np.float32)
    w, s, b = pack(x, group_size=64)
    y = dequant(w, s, b, group_size=64)
    # 16 levels over each group range: max error ~ range/30 + fp margin
    err = np.abs(x - y).max()
    assert err < (x.max() - x.min()) / 20


def test_cpu_bank_matches_oracle():
    rng = np.random.default_rng(1)
    E, H, I, T, K = 8, 32, 16, 4, 2
    gu = rng.normal(0, .1, (E, 2 * H, I)).astype(np.float32)
    dn = rng.normal(0, .1, (E, I, H)).astype(np.float32)
    # bank reads quant-format via dequant; feed it real 4-bit tensors.
    from moefit.quant import pack as _pack
    gup, gus, gub = _pack(gu, group_size=8)
    dnp, dns, dnb = _pack(dn, group_size=8)
    bank = CPUExpertBank((gup, gus, gub), (dnp, dns, dnb), hidden=H, n_experts=E, group_size=8)
    x = rng.normal(0, 1, (T, I)).astype(np.float32)
    indices = np.array([[0, 1], [1, 2], [3, 3], [7, 0]], dtype=np.int64)
    scores = np.full((T, K), 0.5, dtype=np.float32)
    out = bank.execute(x, indices, scores)
    # oracle over dequantized weights
    def silu(z): return z / (1 + np.exp(-z))
    gd = dequant(gup, gus, gub, group_size=8); dd = dequant(dnp, dns, dnb, group_size=8)
    oracle = np.zeros_like(x)
    for t in range(T):
        for j in range(K):
            e = indices[t, j]
            h = silu(x[t] @ gd[e][:H].T) * (x[t] @ gd[e][H:].T)
            oracle[t] += scores[t, j] * (h @ dd[e].T)
    assert np.allclose(out, oracle, atol=2e-4, rtol=1e-4)


def test_bank_duplicate_expert_same_token_sums():
    # token 2 routes to expert 3 twice — contributions must add, not clobber
    rng = np.random.default_rng(2)
    E, H, I, T, K = 4, 16, 16, 3, 2
    gu = rng.normal(0, .1, (E, 2 * H, I)).astype(np.float32)
    dn = rng.normal(0, .1, (E, I, H)).astype(np.float32)
    from moefit.quant import pack as _pack
    gup, gus, gub = _pack(gu, group_size=8)
    dnp, dns, dnb = _pack(dn, group_size=8)
    bank = CPUExpertBank((gup, gus, gub), (dnp, dns, dnb), hidden=H, n_experts=E, group_size=8)
    x = rng.normal(0, 1, (T, I)).astype(np.float32)
    idx = np.array([[0, 1], [2, 3], [3, 3]])
    sc = np.full((T, K), 0.5)
    out = bank.execute(x, idx, sc)
    assert np.isfinite(out).all()
    # single-copy of token 2 must differ from doubled contribution
    out_single = bank.execute(x, np.array([[0, 1], [2, 3], [3, 1]]), sc)
    assert not np.allclose(out[2], out_single[2])
