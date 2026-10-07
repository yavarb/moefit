#!/usr/bin/env python3
"""Score T3's LIP admission A/B/A against the pre-registered rules — as code.

Prereg chain (all committed BEFORE any run data):
  results/T3_ADMISSION_SILICON_PREREG.md   (eb85d02 era: -0.05..+0.25 WASH)
  results/t3_admission_prereg_amendment1.json (34e7523: DB-ON basis,
     single-prompt band [-0.04, +0.17] tps; rules unchanged)
  results/t3_admission_power_analysis.json   (2822e66: 3-run protocol
     resolves >= 0.21 tps -> a WASH is statistically forced at any true
     effect <= +0.17 and must NOT be read as 'effect absent')

ARMS (both on the restored, patched server; default env, DB machinery ON):
  OFF = OMLX_ADMISSION unset/0  (stock insertion, flag-off)
  ON  = OMLX_ADMISSION=1 at process start (LIP insertion)
  Env arms do NOT need separate restarts only if the executor runs two
  restore restarts; if a single restart is used, ON/OFF must still be
  separate PROCESSES (env is read at import). A/B/A = OFF, ON, OFF.

DECISION RULES (pre-registered; unchanged by either amendment):
  d = median(ON tps) - median(OFF tps), >= 3 runs/arm, same prompt
  d >= +0.30  -> TRANSFERS (band check vs [+0.04, +0.17] DB-ON;
                 above-band flagged as un-modeled, still TRANSFERS)
  |d| < 0.30  -> WASH (expected; per power analysis reads as
                 'below the 0.21 tps resolution floor', NOT 'absent')
  d <= -0.30  -> FALSIFIES single-prompt transfer

SECOND READOUT (decisive, per cycle-40 power analysis): per-arm
hits/misses from the T2 stats snapshot — each snapshot is a dict with
summed counters over all 48 caches. Deltas over each arm's window give
miss/token; sim predicts ON-OFF = -1.291 miss/tok in the multi-prompt
steady state (regime the single-prompt bench does not exercise) and
~0 in the single-prompt regime. Snapshot math is reported; no verdict
is emitted from counters alone (their regime validity is limited).

usage:
  score_lip_aba.py --off off.json [off2.json ...] --on on.json [...] \
      [--stats-off t0.json t1.json] [--stats-on s0.json s1.json] \
      [--accept-unhashed-prompt]
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.score_dbuf_aba import load_arm, _median  # noqa: E402

STATS_KEYS = ("hits", "misses", "tokens", "t")


def stats_delta(snaps, label):
    """Counter delta over an arm's window from >= 2 snapshots."""
    if len(snaps) < 2:
        return None, f"{label}: need >=2 stats snapshots for a delta"
    a, b = snaps[0], snaps[-1]
    for k in STATS_KEYS:
        if k not in a or k not in b:
            return None, (f"{label}: stats snapshot missing '{k}' — "
                          "not a moe_offload_stats record")
    hits, misses = b["hits"] - a["hits"], b["misses"] - a["misses"]
    toks = max(b["tokens"] - a["tokens"], 1)
    return dict(hits=hits, misses=misses, miss_per_tok=round(misses / toks, 3),
                hits_per_tok=round(hits / toks, 3)), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", nargs="+", required=True)
    ap.add_argument("--on", nargs="+", required=True)
    ap.add_argument("--stats-off", nargs="*", default=[])
    ap.add_argument("--stats-on", nargs="*", default=[])
    ap.add_argument("--accept-unhashed-prompt", action="store_true")
    a = ap.parse_args()

    MIN_TOK = 1024
    off = load_arm(a.off, "OFF(LIP=0)", MIN_TOK)
    on = load_arm(a.on, "ON(LIP=1)", MIN_TOK)
    e = off["errs"] + on["errs"]
    if on["host"] != off["host"] or on["model"] != off["model"]:
        e.append(f"arms not matched: {on['host']}/{on['model']} vs "
                 f"{off['host']}/{off['model']}")
    hashes = on["prompt_hashes"] + off["prompt_hashes"]
    if any(h is None for h in hashes):
        if a.accept_unhashed_prompt:
            note = ("prompt identity NOT mechanically verified "
                    "(pre-prompt_sha256 blobs; operator attestation)")
        else:
            e.append("prompt_sha256 missing on >=1 blob and "
                     "--accept-unhashed-prompt not given - FAIL CLOSED")
            note = None
    else:
        u = set(hashes)
        note = (f"prompt_sha256 {u.pop()} (verified identical)" if len(u) == 1
                else None)
        if note is None:
            e.append(f"arms used DIFFERENT prompts: {sorted(set(hashes))}")

    d = None
    counters = {}
    if not e:
        d = round(_median(on["tpss"]) - _median(off["tpss"]), 3)
        if d >= 0.30:
            verdict = "TRANSFERS"
        elif d <= -0.30:
            verdict = "FALSIFIES single-prompt transfer"
        else:
            verdict = ("WASH (expected; per the pre-registered power "
                       "analysis this reads 'below the 0.21 tps 3-run "
                       "resolution floor', NOT 'effect absent')")
        if a.stats_off:
            counters["off"], err = stats_delta(
                [json.loads(Path(p).read_text()) for p in a.stats_off],
                "OFF")
            if err:
                e.append(err)
        if a.stats_on:
            counters["on"], err = stats_delta(
                [json.loads(Path(p).read_text()) for p in a.stats_on],
                "ON")
            if err:
                e.append(err)
        if "off" in counters and "on" in counters:
            counters["on_minus_off_miss_per_tok"] = round(
                counters["on"]["miss_per_tok"]
                - counters["off"]["miss_per_tok"], 3)
            counters["note"] = ("sim predicts -1.291 miss/tok in "
                                "multi-prompt steady state; ~0 expected "
                                "in the single-prompt regime — counters "
                                "are a readout, not a verdict")
    else:
        verdict = "SCORING REFUSED (fail-closed eligibility)"

    out = dict(
        test="T3 LIP admission A/B/A",
        kind="ON (OMLX_ADMISSION=1, env at process start) vs OFF (unset), "
             "default DB-ON env, restored patched server; Chief mandate",
        prereg_chain=["results/T3_ADMISSION_SILICON_PREREG.md",
                      "results/t3_admission_prereg_amendment1.json (34e7523)",
                      "results/t3_admission_power_analysis.json (2822e66)"],
        prediction=dict(single_prompt_band_tps=[-0.04, 0.17],
                        decision_thresholds=dict(transfers=0.30,
                                                 falsifies=-0.30),
                        power=dict(three_run_floor_tps=0.21,
                                   wash_is_forced_below=0.17)),
        median_off_tps=(round(_median(off["tpss"]), 3) if off["tpss"] else None),
        median_on_tps=(round(_median(on["tpss"]), 3) if on["tpss"] else None),
        delta_tps=d, verdict=verdict, prompt_note=note,
        counters=counters, errors=e)
    print(json.dumps(out, indent=1))
    return 1 if e else 0


if __name__ == "__main__":
    sys.exit(main())
