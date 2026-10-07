"""Score silicon_dbuf's (T9) DBUF A/B/A against the pre-registered
prediction + AMENDMENT 2 (results/t4_dbuf_silicon_prereg.json, commits
ee44fde + 0af86e7), as code — so T9's measured result gets a mechanical
verdict, not a post-hoc reading.

ARMS (Amendment 2 — the box's staged-install machinery is ON by
default, T9 source read of md5 d420b305):
  ON  = stock default env (the 15.55 tok/s n=1024 baseline IS this arm)
  OFF = OMLX_MOE_OFFLOAD_IO_WORKERS=1 (serial path)
The A/B/A measures the DB's realized contribution by REMOVAL.

ELIGIBILITY IS FAIL-CLOSED (same discipline as score_silicon_run):
  - >= 3 contributing runs PER ARM (pooled across that arm's blobs)
  - actual runs[*].tokens >= 1024 (the REQUEST max_tokens never gates;
    T8: a 1024 request masking 128-token runs must not pass)
  - finish_reason + stream_integrity clean on every contributing run
  - matched arms: same host + model across ALL blobs
  - SAME PROMPT across arms: blob-level prompt_sha256 (collector now
    emits it) must be present and equal. Older blobs without it are
    scored ONLY with --accept-unhashed-prompt (the collector's default
    prompt is a fixed literal, so identity-by-construction is an
    operator attestation, not a mechanical check).

DECISION RULES (pre-registered, thresholds unchanged by Amendment 2):
  delta = median(ON runs' tps) - median(OFF runs' tps)
  d >= +0.30  -> TRANSFERS (report which install band it matches)
  |d| < 0.30  -> WASH (Amendment 2 wash_reading applies: install was
                 never on the DB-ON critical path -> re-price the
                 serial model's install term toward inst*missing_steps)
  d <= -0.30  -> CONTENTION (staged stream fights compute/demand IO)

usage:
  score_dbuf_aba.py --on on1.json [on2.json ...] --off off.json \
      [--prereg results/t4_dbuf_silicon_prereg.json]
      [--accept-unhashed-prompt]
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# reuse the fail-closed loader + eligibility of the protocol scorer
from experiments.score_silicon_run import _load, eligibility


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def load_arm(paths, label, min_tokens):
    """Pool one arm's blobs; enforce arm-level eligibility."""
    blobs = [_load(p, f"{label}:{Path(p).name}") for p in paths]
    tpss, errs = [], []
    for p, b in zip(paths, blobs):
        errs += eligibility(b, cap=None, need_runs=1)   # per-blob integrity
        runs = json.loads(Path(p).read_text()).get("runs") or []
        for r in runs:
            t, tok = r.get("decode_tps"), r.get("tokens")
            if t is None or tok is None:
                errs.append(f"{Path(p).name}: run with unknown tps/tokens")
                continue
            if tok < min_tokens:
                errs.append(f"{Path(p).name}: actual run tokens {tok} < "
                            f"{min_tokens} (request max_tokens does NOT "
                            "gate) - FAIL CLOSED")
            tpss.append(t)
    if len(tpss) < 3:
        errs.append(f"{label} arm has {len(tpss)} contributing run(s); "
                     "protocol requires >= 3")
    host = blobs[0]["host"]
    model = blobs[0]["model"]
    for p, b in zip(paths, blobs):
        if b["host"] != host or b["model"] != model:
            errs.append(f"{Path(p).name}: host/model mismatch within the "
                        f"{label} arm ({b['host']}/{b['model']} vs "
                        f"{host}/{model})")
    hashes = [json.loads(Path(p).read_text()).get("prompt_sha256")
              for p in paths]
    return dict(blobs=blobs, tpss=tpss, errs=errs, host=host, model=model,
                prompt_hashes=hashes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", nargs="+", required=True,
                    help="ON arm blob(s) (default env; A/B/A: first + "
                         "restore ON blobs)")
    ap.add_argument("--off", nargs="+", required=True,
                    help="OFF arm blob(s) (IO_WORKERS=1)")
    ap.add_argument("--prereg",
                    default=str(ROOT / "results/t4_dbuf_silicon_prereg.json"))
    ap.add_argument("--accept-unhashed-prompt", action="store_true",
                    help="score blobs lacking prompt_sha256 (operator "
                         "attests same-prompt; collector default prompt "
                         "is a fixed literal)")
    ap.add_argument("--off-label", default="IO_WORKERS=1 serial",
                    help="semantic label of the OFF arm for the record "
                         "(e.g. 'IO_BATCH=1 (DB off, slab+overlap on)' "
                         "per prereg amendment3 — scoring the BATCH1 "
                         "arm against the original install band)")
    a = ap.parse_args()

    prereg = json.loads(Path(a.prereg).read_text())
    MIN_TOK = 1024
    on = load_arm(a.on, "ON", MIN_TOK)
    off = load_arm(a.off, "OFF", MIN_TOK)

    e = on["errs"] + off["errs"]
    # cross-arm match
    if on["host"] != off["host"] or on["model"] != off["model"]:
        e.append(f"arms not matched: {on['host']}/{on['model']} vs "
                 f"{off['host']}/{off['model']}")
    # prompt identity
    hashes = on["prompt_hashes"] + off["prompt_hashes"]
    if any(h is None for h in hashes):
        if a.accept_unhashed_prompt:
            note = ("prompt identity NOT mechanically verified "
                    "(pre-prompt_sha256 blobs + --accept-unhashed-prompt; "
                    "operator attests same prompt)")
        else:
            e.append("prompt_sha256 missing on >=1 blob and "
                     "--accept-unhashed-prompt not given - FAIL CLOSED")
            note = None
    else:
        u = set(hashes)
        note = (f"prompt_sha256 {u.pop()} (verified identical)" if len(u) == 1
                else None)
        if note is None:
            e.append(f"arms used DIFFERENT prompts: {sorted(set(hashes))} "
                     "- matched-prompt design violated; FAIL CLOSED")

    out = dict(
        test="DBUF A/B/A (Amendment 2)",
        kind="T4 staged-install ON(default) vs OFF(%s), "
             "T9 silicon execution" % a.off_label,
        prereg_source=a.prereg,
        prediction=dict(
            on_is_measured_baseline=15.55,
            off_tps_band_0_18=prereg["predictions"]["eval_each_0.18"]
                                       ["amendment2_off_tps_band"],
            on_minus_off_band=prereg["predictions"]["eval_each_0.18"]
                                               ["delta_tps_band"],
            wash_reading=prereg["amendment2"]["wash_reading"]),
        measured=dict(on_tps=[round(t, 2) for t in on["tpss"]],
                      off_tps=[round(t, 2) for t in off["tpss"]],
                      on_median=round(_median(on["tpss"]), 2),
                      off_median=round(_median(off["tpss"]), 2)),
        eligibility=e,
        prompt_identity=note,
    )
    if e:
        out["verdicts"] = [
            f"SCORING REFUSED - INELIGIBLE: {'; '.join(e)}. No verdict is "
            "emitted; fix eligibility (more/longer runs, matched arms, "
            "same prompt) and rerun."]
        print(json.dumps(out, indent=1))
        return

    d = round(_median(on["tpss"]) - _median(off["tpss"]), 2)
    out["measured"]["delta_on_minus_off"] = d
    band = out["prediction"]["on_minus_off_band"]
    if d >= 0.30:
        v = (f"delta {d} >= +0.30 -> staged install (DB) TRANSFERS to "
             f"silicon. Pre-registered band @install 0.18: {band[0]}.."
             f"{band[1]}. "
             + ("INSIDE the band." if band[0] <= d <= band[1]
                else "OUTSIDE the band - constrains the install constant "
                     "or the steps-fraction (either is informative)."))
    elif abs(d) < 0.30:
        v = (f"|delta| = {d} < 0.30 -> WASH. AMENDMENT 2 READING: removing "
             "the read-ahead window cost nothing, so install was never "
             "on the DB-ON critical path - the 0.797 ms/miss slope is "
             "fetch-dominated and the serial model's install charge "
             "(inst x misses = 17.5 ms/tok @0.30) must be re-priced "
             "toward inst x missing_steps. NOT a falsification of the "
             "mechanism; under sidecar-prefetch bursts the same "
             "mechanism is predicted +7.6 tps (commit 4475d05).")
    else:
        v = (f"delta {d} <= -0.30 -> the staged stream CONTENTS with "
             "compute/demand IO on this box: falsifies the "
             "zero-contention assumption of the staged model (T2 "
             "measured 0.07-0.20 ms/expert interference for background "
             "reads).")
    out["verdicts"] = [v]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
