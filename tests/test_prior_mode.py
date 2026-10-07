"""Regression: sim_paging mode='prior' must not NameError.

glm_fidelity found simulate() referenced an undefined `pin_counts`
(2026-10-06); shipped prior rows predate the break. This pins the fix:
prior runs, the pin_counts per-layer mix path runs, and lru is
unaffected. Uses tiny synthetic arrays - no trace files needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from experiments import sim_paging as sp


def _gold(T=64):
    rng = np.random.default_rng(7)
    # skewed routing so a hot set exists
    hot = rng.integers(0, 16, size=(T, sp.L, sp.K))
    cold = rng.integers(16, sp.E, size=(T, sp.L, sp.K))
    mask = rng.random((T, sp.L, sp.K)) < 0.7
    return np.where(mask, hot, cold).astype(np.int16)


def test_prior_mode_runs():
    gold = _gold()
    prior = {li: np.arange(sp.E) for li in range(sp.L)}
    srv, sync, async_ = sp.simulate(gold, None, prior, 32, "prior", 0, 10**9)
    assert 0 < srv <= 1.0 and sync >= 0


def test_prior_pin_counts_path():
    gold = _gold()
    prior = {li: np.arange(sp.E) for li in range(sp.L)}
    srv, _, _ = sp.simulate(gold, None, prior, 32, "prior", 0, 10**9,
                            pin_counts=[8] * sp.L)
    assert 0 < srv <= 1.0


def test_lru_unaffected():
    gold = _gold()
    srv, _, _ = sp.simulate(gold, None, None, 32, "lru", 0, 10**9)
    assert 0 < srv <= 1.0


if __name__ == "__main__":
    for fn in (test_prior_mode_runs, test_prior_pin_counts_path,
               test_lru_unaffected):
        fn()
        print("ok", fn.__name__)
