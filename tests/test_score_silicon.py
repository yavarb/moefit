"""Tests: score_silicon_run.py implements the PRE-REGISTERED protocol's
decision rules (results/SILICON_TEST_PROTOCOL_PREREGISTERED.md).
Thresholds are fixed by the protocol — if one changes, the protocol must
be re-registered; do not relax a threshold to make a test pass.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "score_silicon_run.py"


def _blob(tps, tokens=1024, gaps=None):
    return dict(
        kind="measured", host="fixture", model="m", timestamp="t",
        max_tokens=tokens, tokens=tokens,
        decode_tps_median=tps, decode_tps_min=tps - 0.5,
        decode_tps_max=tps + 0.5, ttft_s_median=0.7, omlx_version="0.7.0",
        runs=[dict(tokens=tokens, decode_tps=tps, ttft_s=0.7,
                   per_token_ms=gaps)],
        memory_during_run=dict(omlx_actual_gb=28.0,
                               expert_tables_resident_gb=24.5,
                               omlx_full_model_gb=104.0,
                               resident_experts_per_layer_approx=192),
        omlx_model_settings=dict(
            moe_expert_offload_resident_fraction=0.375))


def _run(args):
    r = subprocess.run([sys.executable, str(SCRIPT)] + args,
                      capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def _path(d, obj):
    p = Path(d) / f"{len(list(Path(d).iterdir()))}.json"
    p.write_text(json.dumps(obj))
    return str(p)


def test_all():
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        lru = _path(td, _blob(15.9))
        prior = _path(td, _blob(14.2))
        out = _run(["--test", "A", "--lru", lru, "--prior", prior])
        assert "TRANSFERS" in out["verdicts"][0]
        assert "serial ranking" in out["verdicts"][1]

        out = _run(["--test", "B", "--measured", prior])
        assert "IN the pre-registered 13.8-15.7" in out["verdicts"][0]

        c = _path(td, _blob(12.7, tokens=256,
                            gaps=[70] * 180 + [150] * 20 + [110, 170, 120]))
        out = _run(["--test", "C", "--measured", c])
        v0 = out["verdicts"][0]
        assert "shape-compatible" in v0 and "transport-provisional" in v0, v0

        # n=128 at cap224 must trip the anti-rule warning
        short = _path(td, _blob(16.0, tokens=128))
        out = _run(["--test", "D", "--measured", short])
        assert any("FORBIDS" in w for w in out["warnings"])
        # and a valid-length run yields the compute-term verdict
        out = _run(["--test", "D", "--measured", lru])
        assert "compute term follows" in out["verdicts"][0]


if __name__ == "__main__":
    test_all()
    print("ok test_all")
