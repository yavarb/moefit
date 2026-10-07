"""Tests: within-layer double-buffer pipeline algebra
(experiments/design_double_buffer.py).

Per missing-step with k misses: serial exposes A + B*k + inst*k;
double-buffer exposes A + B*k + inst (fetch-bound, inst <= B) — ONE
trailing install latency. Verified on a fixed hand-computable matrix.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.design_double_buffer import (double_buffer_breakdown,
                                               serial_breakdown, step_stats)
from moefit.metrics import SERIAL_CONSTANTS_MEASURED as SC

# 2 layers x 3 tokens.
# layer 0: misses 3,2,0 experts -> steps 2/3, misses 5/3 per tok
# layer 1: misses 1,0,0 experts -> steps 1/3, misses 1/3 per tok
M = np.array([[3, 1], [2, 0], [0, 0]], dtype=np.int16)
st = step_stats(M)
assert abs(st["misses_per_tok"] - 6 / 3) < 1e-9
assert abs(st["steps_with_miss_frac"] - 3 / 6) < 1e-9

C = dict(SC)  # A=0.20 B=0.52 inst=0.30 sync=0.122
base = serial_breakdown(st, C, 10.0)
db = double_buffer_breakdown(st, C, 10.0)
# serial io: A*steps + B*misses per layer, averaged per token
io_expect = 0.20 * (2 / 3) + 0.52 * (5 / 3) + 0.20 * (1 / 3) + 0.52 * (1 / 3)
inst_expect = 0.30 * 2.0
sync_expect = 0.122 * 2
assert abs(base["io"] - io_expect) < 0.01, base
assert abs(base["install"] - inst_expect) < 0.01, base
assert abs(base["sync"] - sync_expect) < 0.01, base
# double-buffer: install collapses to inst * missing-steps per token
db_inst_expect = 0.30 * 1.0   # (2/3 + 1/3) missing steps per token
assert abs(db["install"] - db_inst_expect) < 0.01, db
assert db["pipeline_bound"] == "fetch"
# DB never slower than serial when inst <= B (install hides under fetch)
assert db["total_ms"] <= base["total_ms"] + 1e-9, (base, db)
# the saving is exactly inst * (misses - missing_steps) per token
saving = base["install"] - db["install"]
assert abs(saving - 0.30 * (2.0 - 1.0)) < 0.01, saving
# GPU idle fraction is (total-compute)/total
assert abs(base["gpu_idle_frac"]
           - (base["total_ms"] - 10.0) / base["total_ms"]) < 0.005
# install-bound branch: inst > B hides B*(k-1) per step instead
C2 = dict(SC, install_ms=0.80)   # > B=0.52
db2 = double_buffer_breakdown(st, C2, 10.0)
assert db2["pipeline_bound"] == "install"
io_saving = base["io"] - db2["io"]
# per token: layer0 k=3,2 on missing steps -> hidden B*(k-1): (2+1)/3... 
# hidden = B*(misses - missing_steps) = 0.52*(2.0-1.0)
assert abs(io_saving - 0.52 * 1.0) < 0.01, (base, db2)

if __name__ == "__main__":
    print("ok test_all")
