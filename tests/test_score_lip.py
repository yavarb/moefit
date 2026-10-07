import json, subprocess, sys, tempfile, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "experiments" / "score_lip_aba.py"


def blob(tps, tok=1024, ph="abc123"):
    return {"host": "santacruz", "model": "m", "prompt_sha256": ph,
            "runs": [{"decode_tps": t, "tokens": tok, "finish_reason": "stop",
                      "stream_integrity": {"problems": []}} for t in tps]}


def run(*extra):
    return subprocess.run([sys.executable, str(SCORER), *extra],
                          capture_output=True, text=True)


def test_wash_with_counters():
    with tempfile.TemporaryDirectory() as d:
        def w(n, o):
            p = os.path.join(d, n)
            json.dump(o, open(p, "w"))
            return p
        off = w("off.json", blob([15.50, 15.55, 15.60]))
        on = w("on.json", blob([15.55, 15.60, 15.65]))
        s0 = w("s0.json", {"hits": 1000, "misses": 5000, "tokens": 0, "t": 1})
        s1 = w("s1.json", {"hits": 1400, "misses": 5300, "tokens": 100, "t": 2})
        out = json.loads(run("--off", off, "--on", on,
                             "--stats-off", s0, s1, "--stats-on", s0, s1)
                         .stdout)
        assert out["verdict"].startswith("WASH"), out["verdict"]
        assert out["delta_tps"] == 0.05
        assert out["counters"]["on_minus_off_miss_per_tok"] == 0.0
        assert not out["errors"]


def test_short_run_refused():
    with tempfile.TemporaryDirectory() as d:
        off = os.path.join(d, "off.json")
        json.dump(blob([15.5], tok=128), open(off, "w"))
        on = os.path.join(d, "on.json")
        json.dump(blob([15.5, 15.6, 15.7]), open(on, "w"))
        out = json.loads(run("--off", off, "--on", on).stdout)
        assert "SCORING REFUSED" in out["verdict"]
        assert out["delta_tps"] is None


def test_unhashed_prompt_gate_and_attestation():
    with tempfile.TemporaryDirectory() as d:
        off = os.path.join(d, "off.json")
        b = blob([15.5, 15.6, 15.7]); b.pop("prompt_sha256")
        json.dump(b, open(off, "w"))
        on = os.path.join(d, "on.json")
        json.dump(blob([15.5, 15.6, 15.7]), open(on, "w"))
        out = json.loads(run("--off", off, "--on", on).stdout)
        assert "SCORING REFUSED" in out["verdict"]
        out2 = json.loads(run("--off", off, "--on", on,
                              "--accept-unhashed-prompt").stdout)
        assert "NOT mechanically verified" in out2["prompt_note"]
        assert out2["verdict"].startswith("WASH")


def test_transfers_and_falsifies():
    with tempfile.TemporaryDirectory() as d:
        off = os.path.join(d, "off.json")
        json.dump(blob([15.50, 15.55, 15.60]), open(off, "w"))
        on = os.path.join(d, "on.json")
        json.dump(blob([16.2, 16.3, 16.4]), open(on, "w"))
        out = json.loads(run("--off", off, "--on", on).stdout)
        assert out["verdict"] == "TRANSFERS" and out["delta_tps"] == 0.75
        on2 = os.path.join(d, "on2.json")
        json.dump(blob([14.8, 14.9, 15.0]), open(on2, "w"))
        out2 = json.loads(run("--off", off, "--on", on2).stdout)
        assert "FALSIFIES" in out2["verdict"]


def test_bad_stats_snapshot_flagged():
    with tempfile.TemporaryDirectory() as d:
        off = os.path.join(d, "off.json")
        json.dump(blob([15.50, 15.55, 15.60]), open(off, "w"))
        on = os.path.join(d, "on.json")
        json.dump(blob([15.55, 15.60, 15.65]), open(on, "w"))
        s0 = os.path.join(d, "s0.json")
        json.dump({"hits": 1}, open(s0, "w"))
        s1 = os.path.join(d, "s1.json")
        json.dump({"hits": 2, "misses": 3, "tokens": 4, "t": 5}, open(s1, "w"))
        out = json.loads(run("--off", off, "--on", on,
                             "--stats-on", s0, s1).stdout)
        assert any("moe_offload_stats" in str(x) for x in out["errors"])
