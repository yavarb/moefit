"""Score a silicon run against the PRE-REGISTERED test protocol.

Implements the decision rules of results/SILICON_TEST_PROTOCOL_PREREGISTERED.md
(T3, commit 2ce6a32) as code, so results cannot be scored post-hoc: the
verdicts and thresholds below were fixed BEFORE any of these runs execute.
Amendment 1 (T3, commit 9640761): Test C verdict labels follow the
downgrade to shape-compatibility — thresholds are UNCHANGED, only the
semantics of what they gate. Do not edit thresholds without re-registering
the protocol.

ELIGIBILITY IS FAIL-CLOSED (T8 audit d778463 + T3 protocol-owner position,
2026-10-08): scoring eligibility gates on ACTUAL per-run completion counts
(runs[*].tokens), never on the requested max_tokens. Unknown counts, fewer
than 3 contributing runs (Tests A/B/D), or actual n<1024 at cap>=180
=> the test refuses to emit ANY model verdict ("SCORING REFUSED") and
returns only descriptive data + the pre-registered predictions. Test A
additionally requires the two blobs to be a matched pair (same host and
model). Test C gates only on known counts (its readouts are shape/
admissibility, not mean scoring), single run admissible.

usage:
  Test A (48GB milestone, needs BOTH policies):
    score_silicon_run.py --test A --lru measured48_lru.json \
        --prior measured48_prior.json
  Test B (36GB cap180):  score_silicon_run.py --test B --measured m.json
  Test C (collector tok-gap shape):  score_silicon_run.py --test C \
        --measured m.json [--vmstat-absorbed-mb-per-tok 18.0]
  Test D (compute term, cap224): score_silicon_run.py --test D --measured m.json

All verdicts quote the pre-registered numbers from the protocol file.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moefit.metrics import (silicon_record_from_measured,
                            signature_from_silicon_record)


def _load(path, label):
    """Load a measured blob + its ACTUAL eligibility facts.

    min_tokens is the smallest runs[*].tokens across runs, or None if
    ANY run's count is missing/empty (fail closed on unknown counts).
    runs[*].tokens — the ACTUAL completion count — is authoritative over
    the top-level max_tokens, which is only the REQUEST (T8: a 1024
    request with 128-token runs must not pass).
    """
    d = json.loads(Path(path).read_text())
    r = silicon_record_from_measured(d, f"{label}:{path}")
    runs = d.get("runs") or []
    per_run = [x.get("tokens") for x in runs]
    min_tokens = None
    if per_run and all(t for t in per_run):
        min_tokens = min(per_run)
    return dict(rec=r, n_runs=len(runs), per_run_tokens=per_run,
                min_tokens=min_tokens, host=d.get("host"),
                model=d.get("model"),
                per_run_finish=[x.get("finish_reason") for x in runs],
                per_run_integrity=[x.get("stream_integrity")
                                   for x in runs])


def eligibility(b, cap, need_runs=3):
    """Fail-closed scoring eligibility (pre-registered anti-rules)."""
    e = []
    if b["n_runs"] < need_runs:
        e.append(f"only {b['n_runs']} run(s); protocol requires "
                 f">= {need_runs} contributing runs")
    if b["min_tokens"] is None:
        e.append("actual per-run completion counts UNKNOWN "
                 "(runs[*].tokens missing or empty) - FAIL CLOSED")
    elif cap is not None and cap >= 180 and b["min_tokens"] < 1024:
        e.append(f"actual n={b['min_tokens']} < 1024 at cap {cap} >= 180: "
                 "protocol FORBIDS scoring (short runs understate; T8 ramp)")
    # stream integrity (T8 stream-integrity probe): incomplete/damaged
    # runs keep descriptive data but must never enter model scoring
    for i, fin in enumerate(b.get("per_run_finish") or []):
        if fin not in ("stop", "length"):
            e.append(f"run {i}: stream incomplete (finish_reason="
                     f"{fin!r}) - FAIL CLOSED")
    for i, si in enumerate(b.get("per_run_integrity") or []):
        if si and si.get("problems"):
            e.append(f"run {i}: stream integrity problems: "
                     f"{'; '.join(si['problems'])}")
    return e


def _refused(problems):
    return [f"SCORING REFUSED - INELIGIBLE: {'; '.join(problems)}. "
            "No model verdict is emitted; descriptive data and the "
            "pre-registered predictions are returned for the record. "
            "Fix eligibility (more/longer runs, actual counts) and rerun."]


def test_a(a):
    if not a.lru or not a.prior:
        sys.exit("Test A needs --lru and --prior (both policies, same box)")
    lru = _load(a.lru, "lru@192")
    prior = _load(a.prior, "prior@192")
    out = dict(test="A", kind="48GB milestone cap192, both policies",
               prediction=dict(lru_serial=15.9, lru_band="14.1-16.5",
                               prior_serial=14.5, prior_band="13-15.5",
                               bw_model_lru=52.0, bw_model_prior=46.7,
                               do_not_score="54.8 (bandwidth artifact)"),
               measured=dict(lru=lru["rec"]["tps"],
                             prior=prior["rec"]["tps"]))
    e = eligibility(lru, 192) + eligibility(prior, 192)
    # matched-pair provenance (T8: mismatched host/model must not compare)
    if (lru["host"] and prior["host"] and lru["host"] != prior["host"]) or \
       (lru["model"] and prior["model"]
                and lru["model"] != prior["model"]):
        e.append(f"not a matched pair: host {lru['host']!r} vs "
                 f"{prior['host']!r}, model {lru['model']!r} vs "
                 f"{prior['model']!r}")
    out["eligibility"] = e
    if e:
        out["verdicts"] = _refused(e)
        return out
    v = []
    t = lru["rec"]["tps"]
    if 12.5 <= t <= 18.5:
        v.append(f"lru {t} in [12.5, 18.5] -> serial model "
                 "TRANSFERS to the 48GB box; score vs this table, NOT 54.8")
    elif t > 25:
        v.append(f"lru {t} > 25 -> serial constants do NOT "
                 "transfer; re-microbench that box before concluding")
    elif t < 10:
        v.append(f"lru {t} < 10 -> memory pressure / crowded box; "
                 "rerun idle")
    else:
        v.append(f"lru {t} in the gap between pre-registered "
                 "bands (18.5, 25) -> inconclusive, investigate")
    if prior["rec"]["tps"] > t + 1:
        v.append(f"prior {prior['rec']['tps']} beats lru {t} by >1 tps "
                 "-> serial ranking flip is WRONG; falsifies BOTH models' "
                 "ranking logic (bandwidth model also ranks lru first)")
    else:
        v.append(f"prior {prior['rec']['tps']} <= lru {t}+1 -> "
                 "serial ranking (lru > prior) holds")
    out["verdicts"] = v
    return out


def test_b(a):
    b = _load(a.measured, "cap180")
    t = b["rec"]["tps"]
    out = dict(test="B", kind="36GB cap180 point",
               prediction=dict(serial_band="13.8-15.7",
                               note="tps alone does NOT discriminate "
                                    "S vs Q (same median)"),
               measured=dict(tps=t))
    e = eligibility(b, 180)
    out["eligibility"] = e
    if e:
        out["verdicts"] = _refused(e)
        return out
    in_band = 13.8 <= t <= 15.7
    out["verdicts"] = [f"cap180 tps {t} "
                       f"{'IN' if in_band else 'OUTSIDE'} the "
                       "pre-registered 13.8-15.7 band"]
    return out


def test_c(a):
    b = _load(a.measured, "collector n256")
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
    # Test C readouts are shape/admissibility, not mean scoring: require
    # known actual counts, but a single run with gaps is admissible input
    e = eligibility(b, cap=None, need_runs=1)
    out["eligibility"] = e
    if e:
        out["verdicts"] = _refused(e)
        return out
    v = []
    tg = b["rec"].get("tok_gap_ms")
    if tg:
        sig = signature_from_silicon_record(b["rec"])
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
    b = _load(a.measured, "cap224")
    t = b["rec"]["tps"]
    out = dict(test="D", kind="compute-term separation, cap224",
               prediction=dict(tps_at_compute_18_1=17.4,
                               tps_at_compute_24_1=15.8,
                               delta=1.6,
                               breakdown="io 22.9 + install 10.5 + sync 5.9"),
               measured=dict(tps=t))
    e = eligibility(b, 224)
    out["eligibility"] = e
    if e:
        out["verdicts"] = _refused(e)
        return out
    d18 = abs(t - 17.4)
    d24 = abs(t - 15.8)
    closer = "18.1 ms (DRAM_EFF@546)" if d18 <= d24 else "24.1 ms (DRAM_EFF@410)"
    out["verdicts"] = [f"measured {t} tps is closer to the "
                       f"{closer} prediction (|d| {min(d18, d24):.1f} vs "
                       f"{max(d18, d24):.1f}) -> compute term follows "
                       f"{closer}; pin every tier prediction by the "
                       "same offset"]
    return out


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
