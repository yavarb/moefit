"""Score a silicon run against the PRE-REGISTERED test protocol.

Implements the decision rules of results/SILICON_TEST_PROTOCOL_PREREGISTERED.md
(T3, commit 2ce6a32) as code, so results cannot be scored post-hoc: the
verdicts and thresholds below were fixed BEFORE any of these runs execute.
Amendment 1 (T3, commit 9640761): Test C verdict labels follow the
downgrade to shape-compatibility — thresholds are UNCHANGED, only the
semantics of what they gate. Do not edit thresholds without re-registering
the protocol.

usage:
  Test A (48GB milestone, needs BOTH policies):
    score_silicon_run.py --test A --lru measured48_lru.json \
        --prior measured48_prior.json
  Test B (36GB cap180):  score_silicon_run.py --test B --measured m.json
  Test C (collector S-vs-Q):  score_silicon_run.py --test C --measured m.json \
        [--vmstat-absorbed-mb-per-tok 18.0]
  Test D (compute term, cap224): score_silicon_run.py --test D --measured m.json

All verdicts quote the pre-registered numbers from the protocol file.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moefit.metrics import (silicon_record_from_measured,
                            signature_from_silicon_record)


def _rec(path, label):
    r = silicon_record_from_measured(path, f"{label}:{path}")
    tokens = None
    try:
        d = json.loads(Path(path).read_text())
        tokens = d.get("tokens") or d.get("max_tokens")
    except Exception:
        pass
    return r, tokens


def _anti_rule_warnings(tokens, cap):
    w = []
    if cap is not None and cap >= 180 and tokens is not None and tokens < 1024:
        w.append(f"n={tokens} < 1024 at cap {cap} >= 180: protocol FORBIDS "
                 "scoring this run (short runs understate; T8 ramp)")
    return w


def test_a(a):
    if not a.lru or not a.prior:
        sys.exit("Test A needs --lru and --prior (both policies, same box)")
    lru, tl = _rec(a.lru, "lru@192")
    prior, tp = _rec(a.prior, "prior@192")
    out = dict(test="A", kind="48GB milestone cap192, both policies",
               prediction=dict(lru_serial=15.9, lru_band="14.1-16.5",
                               prior_serial=14.5, prior_band="13-15.5",
                               bw_model_lru=52.0, bw_model_prior=46.7,
                               do_not_score="54.8 (bandwidth artifact)"),
               measured=dict(lru=lru["tps"], prior=prior["tps"]))
    out["warnings"] = _anti_rule_warnings(tl, 192) + _anti_rule_warnings(tp, 192)
    v = []
    if 12.5 <= lru["tps"] <= 18.5:
        v.append(f"lru {lru['tps']} in [12.5, 18.5] -> serial model "
                 "TRANSFERS to the 48GB box; score vs this table, NOT 54.8")
    elif lru["tps"] > 25:
        v.append(f"lru {lru['tps']} > 25 -> serial constants do NOT "
                 "transfer; re-microbench that box before concluding")
    elif lru["tps"] < 10:
        v.append(f"lru {lru['tps']} < 10 -> memory pressure / crowded box; "
                 "rerun idle")
    else:
        v.append(f"lru {lru['tps']} in the gap between pre-registered "
                 "bands (18.5, 25) -> inconclusive, investigate")
    if prior["tps"] > lru["tps"] + 1:
        v.append(f"prior {prior['tps']} beats lru {lru['tps']} by >1 tps "
                 "-> serial ranking flip is WRONG; falsifies BOTH models' "
                 "ranking logic (bandwidth model also ranks lru first)")
    else:
        v.append(f"prior {prior['tps']} <= lru {lru['tps']}+1 -> "
                 "serial ranking (lru > prior) holds")
    out["verdicts"] = v
    return out


def test_b(a):
    rec, tokens = _rec(a.measured, "cap180")
    in_band = 13.8 <= rec["tps"] <= 15.7
    return dict(test="B", kind="36GB cap180 point",
                prediction=dict(serial_band="13.8-15.7",
                                note="tps alone does NOT discriminate "
                                     "S vs Q (same median)"),
                measured=dict(tps=rec["tps"]),
                warnings=_anti_rule_warnings(tokens, 180),
                verdicts=[f"cap180 tps {rec['tps']} "
                          f"{'IN' if in_band else 'OUTSIDE'} the "
                          "pre-registered 13.8-15.7 band"])


def test_c(a):
    rec, tokens = _rec(a.measured, "collector n256")
    out = dict(test="C", kind="collector run: tok-gap SHAPE (Amendment 1)",
               prediction=dict(s_p95_over_mean=1.51,
                               s_p50_ms=72, s_p95_ms=120,
                               bursty_q_p95_over_mean="1.83 (shares the "
                               "miss-burst shape; astra_analysis_b)",
                               threshold=1.3,
                               note=">=1.3 is shape-compatibility ONLY "
                                    "(no mechanism verdict); <=1.1 "
                                    "falsifies BOTH S and bursty-Q, "
                                    "decisive only with verified "
                                    "token-level provenance"))
    v = []
    tg = rec.get("tok_gap_ms")
    if tg:
        sig = signature_from_silicon_record(rec)
        out["measured"] = dict(p50=tg["p50"], p95=tg["p95"], mean=tg["mean"],
                               p95_over_mean=sig["p95_over_mean"])
        v.append(f"p95/mean {sig['p95_over_mean']} -> {sig['verdict']}")
    else:
        out["measured"] = None
        v.append("NO per-token gaps in the blob - run "
                 "collect_silicon_run.py, not the plain bench "
                 "(readout 1 unavailable)")
    if a.vmstat_absorbed_mb_per_tok is not None:
        x = a.vmstat_absorbed_mb_per_tok
        out["vmstat_absorbed_mb_per_tok"] = x
        if x >= 10:
            v.append(f"{x} MB/tok absorbed -> buffer-cache reuse / "
                     "non-F_NOCACHE reads explain the 0.885 phys/logical")
        else:
            v.append(f"{x} MB/tok absorbed ~= 0 -> oMLX admission beats "
                     "LRU by ~12% (misses 50.8 floor vs 58.3 true-LRU) "
                     "-> admission policy is design territory")
    out["verdicts"] = v
    return out


def test_d(a):
    rec, tokens = _rec(a.measured, "cap224")
    t = rec["tps"]
    d18 = abs(t - 17.4)
    d24 = abs(t - 15.8)
    closer = "18.1 ms (DRAM_EFF@546)" if d18 <= d24 else "24.1 ms (DRAM_EFF@410)"
    return dict(test="D", kind="compute-term separation, cap224",
                prediction=dict(tps_at_compute_18_1=17.4,
                                tps_at_compute_24_1=15.8,
                                delta=1.6,
                                breakdown="io 22.9 + install 10.5 + sync 5.9"),
                measured=dict(tps=t),
                warnings=_anti_rule_warnings(tokens, 224),
                verdicts=[f"measured {t} tps is closer to the "
                         f"{closer} prediction (|d| {min(d18, d24):.1f} vs "
                         f"{max(d18, d24):.1f}) -> compute term follows "
                         f"{closer}; pin every tier prediction by the "
                         "same offset"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, choices=list("ABCD"))
    ap.add_argument("--measured", default=None)
    ap.add_argument("--lru", default=None, help="Test A lru@192 blob")
    ap.add_argument("--prior", default=None, help="Test A prior@192 blob")
    ap.add_argument("--vmstat-absorbed-mb-per-tok", type=float, default=None,
                    help="Test C readout 2: page-cache-absorbed MB/tok")
    a = ap.parse_args()
    fn = {"A": test_a, "B": test_b, "C": test_c, "D": test_d}[a.test]
    if a.test != "A" and not a.measured:
        sys.exit(f"Test {a.test} needs --measured")
    out = fn(a)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
