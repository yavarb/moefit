"""Tests: score_silicon_run.py implements the PRE-REGISTERED protocol's
decision rules (results/SILICON_TEST_PROTOCOL_PREREGISTERED.md).
Thresholds are fixed by the protocol — if one changes, the protocol must
be re-registered; do not relax a threshold to make a test pass.

Eligibility is FAIL-CLOSED (T8 audit d778463): the adversarial fixtures
below reproduce the audit's protocol-invalid inputs and must all receive
SCORING REFUSED with no model verdict.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "score_silicon_run.py"


def _blob(tps, tokens=1024, gaps=None, n_runs=3, max_tokens=None,
          drop_run_tokens=False, host="fixture", model="m",
          drop_run_finish=False):
    runs = []
    for i in range(n_runs):
        r = dict(tokens=tokens, decode_tps=tps, ttft_s=0.7,
                 per_token_ms=gaps, finish_reason="stop")
        if drop_run_tokens:
            del r["tokens"]   # T8: missing per-run counts
        if drop_run_finish:
            del r["finish_reason"]   # T8: incomplete stream
        runs.append(r)
    return dict(
        kind="measured", host=host, model=model, timestamp="t",
        max_tokens=(max_tokens if max_tokens is not None else tokens),
        tokens=tokens,
        decode_tps_median=tps, decode_tps_min=tps - 0.5,
        decode_tps_max=tps + 0.5, ttft_s_median=0.7, omlx_version="0.7.0",
        runs=runs,
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
        assert out["eligibility"] == []

        out = _run(["--test", "B", "--measured", prior])
        assert "IN the pre-registered 13.8-15.7" in out["verdicts"][0]

        c = _path(td, _blob(12.7, tokens=256,
                            gaps=[70] * 180 + [150] * 20 + [110, 170, 120]))
        out = _run(["--test", "C", "--measured", c])
        v0 = out["verdicts"][0]
        assert "shape-compatible" in v0 and "transport-provisional" in v0, v0

        out = _run(["--test", "D", "--measured", lru])
        assert "compute term follows" in out["verdicts"][0]

        # ---- fail-closed eligibility (T8 adversarial fixtures) ----

        # T8 critical case: max_tokens=1024 request masks actual
        # 128-token runs -> must REFUSE (old scorer passed silently)
        masked = _path(td, _blob(16.0, tokens=128, max_tokens=1024))
        out = _run(["--test", "D", "--measured", masked])
        assert out["verdicts"][0].startswith("SCORING REFUSED"), out
        assert "actual n=128" in out["verdicts"][0]
        assert not any("compute term follows" in v
                       for v in out["verdicts"])

        # unknown counts (runs[*].tokens missing) -> fail closed
        unknown = _path(td, _blob(16.0, drop_run_tokens=True))
        out = _run(["--test", "D", "--measured", unknown])
        assert out["verdicts"][0].startswith("SCORING REFUSED"), out
        assert "UNKNOWN" in out["verdicts"][0]

        # only 1 run vs required 3 -> refuse
        one_run = _path(td, _blob(16.0, tokens=1024, n_runs=1))
        out = _run(["--test", "D", "--measured", one_run])
        assert out["verdicts"][0].startswith("SCORING REFUSED"), out
        assert ">= 3 contributing runs" in out["verdicts"][0]

        # short n at cap>=180 -> refuse even when explicitly requested
        short = _path(td, _blob(16.0, tokens=128))
        out = _run(["--test", "B", "--measured", short])
        assert out["verdicts"][0].startswith("SCORING REFUSED"), out
        assert "FORBIDS scoring" in out["verdicts"][0]
        assert "IN the pre-registered" not in out["verdicts"][0]

        # mismatched Test A pair (host/model) -> refuse
        out = _run(["--test", "A", "--lru", lru,
                    "--prior", _path(td, _blob(14.2, host="otherbox"))])
        assert out["verdicts"][0].startswith("SCORING REFUSED"), out
        assert "matched pair" in out["verdicts"][0]

        # Test C needs known counts but a single valid run is admissible
        c1 = _path(td, _blob(12.7, tokens=256, n_runs=1,
                             gaps=[70] * 180 + [150] * 20))
        out = _run(["--test", "C", "--measured", c1])
        assert not out["verdicts"][0].startswith("SCORING REFUSED"), out
        # ...but unknown counts still refuse
        out = _run(["--test", "C", "--measured",
                    _path(td, _blob(12.7, drop_run_tokens=True))])
        assert out["verdicts"][0].startswith("SCORING REFUSED"), out

        # stream-integrity gate (T8 stream-integrity probe): a run with
        # no finish_reason is an incomplete stream -> refuse scoring
        out = _run(["--test", "D", "--measured",
                    _path(td, _blob(16.0, drop_run_finish=True))])
        assert out["verdicts"][0].startswith("SCORING REFUSED"), out
        assert "stream incomplete" in out["verdicts"][0]
        # ...and a recorded integrity problem (e.g. truncated stream)
        bad = _blob(16.0)
        bad["runs"][0]["stream_integrity"] = dict(
            eligible=False, problems=["missing_or_unsupported_finish"])
        out = _run(["--test", "D", "--measured", _path(td, bad)])
        assert out["verdicts"][0].startswith("SCORING REFUSED"), out
        assert "integrity problems" in out["verdicts"][0]


if __name__ == "__main__":
    test_all()
    print("ok test_all")
