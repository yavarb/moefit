"""Regression tests for experiments/score_dbuf_aba.py (T4 DBUF A/B/A
scorer, Amendment 2 semantics).

Covers the three pre-registered verdicts and the fail-closed
eligibility gates: masked short runs (request max_tokens never gates),
prompt mismatch, unhashed-prompt gating, stream integrity, arm run
count, and A/B/A pooling.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "score_dbuf_aba.py"


def blob(tpss, prompt="abc123", tokens=1024, host="santacruz",
         model="qwen4", max_tokens=1024, finish="stop"):
    return dict(
        label="t", kind="measured", timestamp="x", host=host,
        ram_gib=36, model=model, prompt_sha256=prompt,
        max_tokens=max_tokens,
        runs=[dict(run=i, decode_tps=t, tokens=tokens,
                   finish_reason=finish, prompt_tokens=10)
              for i, t in enumerate(tpss)],
        decode_tps_median=sorted(tpss)[len(tpss) // 2],
        decode_tps_min=min(tpss), decode_tps_max=max(tpss),
        ttft_s_median=1.0, method="m", git_commit="t")


class ScoreDbuf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.T = Path(cls.tmp.name)
        cls.on = cls.w("on.json", blob([15.55, 15.42, 15.73]))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def w(cls, name, d):
        p = cls.T / name
        p.write_text(json.dumps(d))
        return str(p)

    def run_scorer(self, on, off, extra=()):
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "--on", on, "--off", off, *extra],
            capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)

    def test_transfers_in_band(self):
        out = self.run_scorer(self.on, self.w(
            "off_t.json", blob([15.0, 14.9, 14.8])))
        self.assertIn("TRANSFERS", out["verdicts"][0])
        self.assertIn("INSIDE the band", out["verdicts"][0])
        self.assertEqual(out["prompt_identity"],
                         "prompt_sha256 abc123 (verified identical)")
        self.assertEqual(out["measured"]["delta_on_minus_off"], 0.65)

    def test_wash_carries_amendment2_reading(self):
        out = self.run_scorer(self.on, self.w(
            "off_w.json", blob([15.5, 15.4, 15.45])))
        self.assertIn("WASH", out["verdicts"][0])
        self.assertIn("re-priced", out["verdicts"][0])

    def test_contention(self):
        out = self.run_scorer(self.on, self.w(
            "off_c.json", blob([16.1, 16.2, 16.3])))
        self.assertIn("CONTENTS", out["verdicts"][0])

    def test_refused_masked_short_runs(self):
        # T8's critical case: max_tokens=1024 masking 128-token runs
        out = self.run_scorer(self.on, self.w(
            "bad_n.json", blob([14.0, 14.1, 14.2], tokens=128)))
        self.assertTrue(out["verdicts"][0].startswith("SCORING REFUSED"))
        self.assertTrue(any("does NOT gate" in e for e in out["eligibility"]))

    def test_refused_prompt_mismatch(self):
        out = self.run_scorer(self.on, self.w(
            "mismatch.json", blob([15.0, 14.9, 14.8], prompt="zzz")))
        self.assertTrue(any("DIFFERENT prompts" in e
                            for e in out["eligibility"]))
        self.assertTrue(out["verdicts"][0].startswith("SCORING REFUSED"))

    def test_unhashed_prompt_gated_then_attested(self):
        p = self.w("unhashed.json", {**blob([15.0, 14.9, 14.8]),
                                     "prompt_sha256": None})
        out = self.run_scorer(self.on, p)
        self.assertTrue(any("prompt_sha256 missing" in e
                            for e in out["eligibility"]))
        out = self.run_scorer(self.on, p, ["--accept-unhashed-prompt"])
        self.assertIn("TRANSFERS", out["verdicts"][0])
        self.assertIn("attests same prompt", out["prompt_identity"])

    def test_refused_incomplete_stream(self):
        out = self.run_scorer(self.on, self.w(
            "incomplete.json", blob([15.0, 14.9, 14.8], finish=None)))
        self.assertTrue(any("stream incomplete" in e
                            for e in out["eligibility"]))

    def test_refused_too_few_runs(self):
        out = self.run_scorer(self.on, self.w(
            "few.json", blob([15.0, 15.1])))
        self.assertTrue(any("requires >= 3" in e
                            for e in out["eligibility"]))

    def test_aba_pooling(self):
        on2 = self.w("on2.json", blob([15.60, 15.5, 15.68]))
        off = self.w("off_p.json", blob([15.0, 14.9, 14.8]))
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "--on", self.on, on2,
             "--off", off],
            capture_output=True, text=True, cwd=str(ROOT))
        out = json.loads(r.stdout)
        self.assertEqual(len(out["measured"]["on_tps"]), 6)
        self.assertIn("TRANSFERS", out["verdicts"][0])


if __name__ == "__main__":
    unittest.main()
