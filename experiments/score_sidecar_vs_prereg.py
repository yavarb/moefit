#!/usr/bin/env python3
"""Score T2's SIDECAR silicon A/B/A against T5's pre-registered prediction
(results/t5_sidecar_silicon_prereg.json, commit 81a6483 + conditional
amendment results/t5_iobatch1_conditional_prereg.json, commit 088913d) —
as code, so the sidecar result gets a mechanical verdict, not a post-hoc
reading. Registered BEFORE any sidecar B-arm data existed.

ARMS (T2's t2_sidecar_aba.py): sidecar OFF = A0/A/A2/A3 (default env, flag
absent); sidecar ON = B/B2 (/tmp/omlx_sidecar_on present). The sidecar
deploys the T5 cancel-on-wrong-route primitive (exact-replay signal, held
bytes, cancel-on-wrong-route), so this is the T5 breakeven framework's
first PROSPECTIVE silicon test.

DECISION RULES (pre-registered, t5_sidecar_silicon_prereg.json):
  band(m) = [1000/(ms_on - m*0.52 + m*0.198), 1000/(ms_on - m*0.82 + m*0.073)]
  with ms_on = 1000/15.72 (measured DB-ON baseline, 75cdc0f) and m banded
  30-50 when no measured m is available.
  B_median inside [floor, ceiling] -> framework TRANSFERS prospectively;
  below floor -> saving overstated OR coverage m_cov << m_iss (arbitrated by
      T2's sidecar counters: issued/used per token);
  above ceiling -> live contention cheaper than the 0.073 ms/expert floor,
      revising the contention term DOWN for all background-read designs.

REFINEMENT (conditional prereg 088913d): if T9's IO_BATCH=1 arm landed
BEFORE this scoring, pass --iobatch1-tps to replace the (0.52, 0.82) saving
bracket with the singleton S' = 0.52 + D/m, D = 1000/tps_b1 - 1000/15.72,
subject to sanity gates D >= 0 and D/m <= 0.30. If it landed AFTER, the
original bracket is used and S' is reported as post-hoc sensitivity.

ELIGIBILITY: reuse score_dbuf_aba.load_arm (fail-closed: >=3 contributing
runs/arm, actual runs[*].tokens >= 1024, integrity clean, host/model match,
prompt_sha256 identity or --accept-unhashed-prompt attestation).

usage:
  score_sidecar_vs_prereg.py --off a0.json a.json a2.json --on b.json b2.json \
      [--miss-per-tok M] [--sidecar-stats stats.json] \
      [--iobatch1-tps TPS] [--accept-unhashed-prompt]
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.score_dbuf_aba import load_arm, _median

MS_ON = 1000.0 / 15.72          # measured DB-ON baseline median (75cdc0f)
SAVE = (0.52, 0.82)             # ms per confirmed spec (DB-ON .. serial)
CONT = (0.073, 0.198)            # ms per issued background read (measured)
M_BAND = (30.0, 50.0)            # logical miss/tok band (measured anchors)
NOISE = 0.21                     # 3-run power floor tps (T3 2822e66)


def band(m, save=SAVE, cont=CONT, base_ms=MS_ON):
    """Predicted B-arm tps (floor, ceil) at miss/tok m on base_ms."""
    pess = base_ms - (m * save[0] - m * cont[1])   # least saving, most contention
    opt = base_ms - (m * save[1] - m * cont[0])    # most saving, least contention
    return 1000.0 / pess, 1000.0 / opt             # (floor_tps, ceil_tps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", nargs="+", required=True,
                    help="sidecar-OFF blobs (A0/A/A2/A3)")
    ap.add_argument("--on", nargs="+", required=True,
                    help="sidecar-ON blobs (B/B2)")
    ap.add_argument("--miss-per-tok", type=float, default=None,
                    help="measured logical miss/tok for the ON arm "
                         "(from stats diffs); else banded 30-50")
    ap.add_argument("--sidecar-stats", default=None,
                    help="optional JSON with issued_per_tok / "
                         "used_per_tok (arbitrates below-floor rule 2)")
    ap.add_argument("--iobatch1-tps", type=float, default=None,
                    help="T9 IO_BATCH=1 arm median tps, IF it landed "
                         "before this scoring (conditional prereg 088913d)")
    ap.add_argument("--accept-unhashed-prompt", action="store_true")
    a = ap.parse_args()

    off = load_arm(a.off, "OFF", 1024)
    on = load_arm(a.on, "ON", 1024)
    e = off["errs"] + on["errs"]
    if off["host"] != on["host"] or off["model"] != on["model"]:
        e.append(f"arms not matched: {off['host']}/{off['model']} vs "
                 f"{on['host']}/{on['model']}")
    hashes = off["prompt_hashes"] + on["prompt_hashes"]
    if any(h is None for h in hashes):
        if a.accept_unhashed_prompt:
            note = "prompt identity NOT mechanically verified (attestation)"
        else:
            e.append("prompt_sha256 missing on >=1 blob and "
                     "--accept-unhashed-prompt not given - FAIL CLOSED")
            note = None
    else:
        u = set(hashes)
        note = (f"prompt_sha256 {u.pop()} (verified identical)"
                if len(u) == 1 else None)
        if note is None:
            e.append(f"arms used DIFFERENT prompts: {sorted(u)} - FAIL CLOSED")

    save, save_note = SAVE, "original bracket (IO_BATCH=1 not landed first)"
    sanity = None
    if a.iobatch1_tps is not None:
        m_ref = a.miss_per_tok if a.miss_per_tok else 37.1
        D = 1000.0 / a.iobatch1_tps - MS_ON
        if D < 0 or D / m_ref > 0.30:
            sanity = (f"sanity gate FAILED: D={D:.2f} ms/tok, D/m="
                      f"{D / m_ref:.3f} outside [0, 0.30] - no re-price, "
                      "flag attribution to T9/T4")
        else:
            save = (0.52 + D / m_ref, 0.52 + D / m_ref)
            save_note = (f"singleton S'={save[0]:.3f} from IO_BATCH=1 "
                         f"tps {a.iobatch1_tps} (D={D:.2f} ms/tok, "
                         f"m={m_ref})")

    out = dict(
        test="T5 sidecar prereg — breakeven framework's first prospective "
             "silicon test",
        kind="T2 sidecar OFF vs ON (deploys the T5 cancel-on-wrong-route "
             "primitive)",
        prereg_source="results/t5_sidecar_silicon_prereg.json (+ "
                      "results/t5_iobatch1_conditional_prereg.json)",
        saving_basis=dict(save=save, note=save_note, sanity_gate=sanity),
        measured=dict(off_tps=[round(t, 2) for t in off["tpss"]],
                      on_tps=[round(t, 2) for t in on["tpss"]],
                      off_median=round(_median(off["tpss"]), 2),
                      on_median=round(_median(on["tpss"]), 2)),
        eligibility=e,
        prompt_identity=note,
    )
    if e:
        out["verdicts"] = ["SCORING REFUSED - INELIGIBLE: "
                           + "; ".join(e) + ". No verdict emitted."]
        print(json.dumps(out, indent=1))
        return

    d = round(_median(on["tpss"]) - _median(off["tpss"]), 2)
    out["measured"]["delta_on_minus_off"] = d
    # PRIMARY rule (pre-registered): absolute B-arm median vs the band on
    # the prereg's 15.72 baseline. SECONDARY readout: band re-anchored to
    # the measured OFF median (within-run drift control).
    if a.miss_per_tok:
        fl, ce = band(a.miss_per_tok, save)
        out["band_basis"] = f"measured m={a.miss_per_tok}"
    else:
        fl = min(band(m, save)[0] for m in M_BAND)
        ce = max(band(m, save)[1] for m in M_BAND)
        out["band_basis"] = "m banded 30-50 (no measured m supplied)"
    off_ms = 1000.0 / _median(off["tpss"])
    if a.miss_per_tok:
        fl2, ce2 = band(a.miss_per_tok, save, base_ms=off_ms)
    else:
        fl2 = min(band(m, save, base_ms=off_ms)[0] for m in M_BAND)
        ce2 = max(band(m, save, base_ms=off_ms)[1] for m in M_BAND)
    out["predicted_band_tps"] = [round(fl, 1), round(ce, 1)]
    out["band_reanchored_to_measured_off"] = [round(fl2, 1), round(ce2, 1)]

    b_med = round(_median(on["tpss"]), 2)
    if fl <= b_med <= ce:
        v = (f"B median {b_med} tps INSIDE the pre-registered band "
             f"[{fl:.1f}, {ce:.1f}] -> the T5 breakeven framework TRANSFERS "
             "PROSPECTIVELY to silicon; the DB-ON saving constant 0.52 "
             "gains measured support.")
    elif b_med < fl:
        arb = ""
        if a.sidecar_stats:
            try:
                s = json.loads(Path(a.sidecar_stats).read_text())
                iss, use = s.get("issued_per_tok"), s.get("used_per_tok")
                if iss and use:
                    arb = (" Coverage arbitration: issued "
                           f"{iss}/tok vs used {use}/tok -> "
                           + ("coverage m_cov << m_iss (replay misses the "
                              "warm-cache demand set)."
                              if use < 0.7 * iss else
                              "coverage ~1, so the SAVING constant is "
                              "overstated."))
            except Exception as ex:
                arb = f" sidecar-stats unreadable ({ex})."
        v = (f"B median {b_med} tps BELOW the pre-registered floor "
             f"{fl:.1f} -> either the DB-ON saving constant is overstated "
             "OR sidecar coverage m_cov << m_iss; T2's issued/used "
             f"counters arbitrate which.{arb} NOTE: if |delta| = |{d}| < "
             f"{NOISE} the arm is also within the 3-run power floor "
             "(WASH-class): the sidecar bought ~nothing on this prompt.")
    else:
        v = (f"B median {b_med} tps ABOVE the pre-registered ceiling "
             f"{ce:.1f} -> live-pipeline contention is CHEAPER than the "
             "measured 0.073 ms/expert floor (the 12-worker pool absorbs "
             "the 4-thread sidecar lane): revise the contention term DOWN "
             "for all background-read designs. This also loosens the T5 "
             "DROP's breakeven floor - recorded honestly.")
    v += (f" Secondary (band re-anchored to the measured OFF median "
          f"{_median(off['tpss']):.2f}: [{fl2:.1f}, {ce2:.1f}]; measured "
          f"delta {d}) reported as the within-run drift control, not the "
          "primary rule.")
    out["verdicts"] = [v]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
