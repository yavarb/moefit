"""Canonical run-record metrics for moefit-lab.

Every sim or silicon (Santa Cruz) result should be reducible to a
`RunRecord` dict so sim↔silicon gaps are compared field-to-field instead
of by eyeballing prose in result JSONs.

Record schema (all keys optional except kind, tps, source):
  kind        "sim" | "silicon"
  source      short provenance string (file path, label, or run id)
  tps         decode tok/s (median for multi-run silicon)
  tps_stats   optional {min, max, median, n_runs}
  ttft_s      time to first token, seconds
  residency   fraction of experts resident per layer (0..1)
  experts_per_layer_resident   int, same as residency*E when known
  footprint_gib        model+expert DRAM footprint actually resident
  floor_gib            non-expert resident floor (attn/shared/lm_head)
  expert_gib_resident  resident expert table bytes
  full_model_gib       full on-disk model size
  sync_mb_per_tok      SSD sync reads (blocking misses), MiB/token
  async_mb_per_tok     SSD prefetch reads, MiB/token
  compute_ms_per_tok   DRAM-bound term
  stream_ms_per_tok    SSD-bound term
  limiter              "compute" | "stream" | None
  config      free-form dict: policy/mode/cap/tier/throttle or omlx settings
  host        e.g. "santacruz", "local-sim"
  measured_at timestamp string
  git_commit  commit of the code (sim) or of the runtime (silicon)

Geometry constants mirror experiments/sim_paging.py (single source would
be nicer; kept in sync deliberately until sim_paging imports these).
"""
from __future__ import annotations

EXPERT_MIB = 2.69
L, K, E = 48, 10, 512

REQUIRED = ("kind", "tps", "source")
NUMERIC = (
    "tps", "ttft_s", "residency", "experts_per_layer_resident",
    "footprint_gib", "floor_gib", "expert_gib_resident",
    "full_model_gib", "sync_mb_per_tok", "async_mb_per_tok",
    "compute_ms_per_tok", "stream_ms_per_tok",
)


def validate_record(rec: dict) -> list[str]:
    """Return a list of schema violations (empty list = valid)."""
    errs = []
    for k in REQUIRED:
        if k not in rec:
            errs.append(f"missing required key {k!r}")
    if rec.get("kind") not in ("sim", "silicon"):
        errs.append(f"kind must be 'sim' or 'silicon', got {rec.get('kind')!r}")
    for k in NUMERIC:
        v = rec.get(k)
        if v is not None and not isinstance(v, (int, float)):
            errs.append(f"{k} must be numeric, got {type(v).__name__}")
    if rec.get("tps") is not None and rec["tps"] <= 0:
        errs.append("tps must be > 0")
    if rec.get("residency") is not None and not 0 < rec["residency"] <= 1:
        errs.append("residency must be in (0, 1]")
    return errs


def sim_record_from_row(row: dict, tier: str, source: str) -> dict:
    """Convert one sim_paging.json row (per-tier columns) to a RunRecord."""
    cap = row["cap"]
    served = row[f"served_{tier}"]
    rec = dict(
        kind="sim",
        source=source,
        tps=row[f"tps_{tier}"],
        compute_ms_per_tok=row[f"ct_{tier}"],
        stream_ms_per_tok=row[f"st_{tier}"],
        limiter="compute" if row[f"ct_{tier}"] >= row[f"st_{tier}"]
                else "stream",
        residency=cap / E,
        experts_per_layer_resident=cap,
        floor_gib=round(4.6, 2),
        expert_gib_resident=round(cap * L * EXPERT_MIB / 1024, 2),
        config=dict(mode=row["mode"], cap=cap,
                    probe_picks=row.get("probe_picks"),
                    throttle=row.get("throttle"), tier=tier),
    )
    ssd_mb = row.get(f"ssdMB_{tier}")
    if ssd_mb is not None:
        # legacy rows only carry the combined SSD traffic; split unknown
        rec["config"]["ssd_mb_per_tok_total"] = ssd_mb
    rec["footprint_gib"] = round(
        rec["floor_gib"] + rec["expert_gib_resident"], 2)
    errs = validate_record(rec)
    assert not errs, f"bad sim record: {errs}"
    return rec


def silicon_record_from_measured(path_or_dict, source: str) -> dict:
    """Normalize a measured_santa_cruz_*.json blob to a RunRecord."""
    import json
    from pathlib import Path
    d = path_or_dict if isinstance(path_or_dict, dict) \
        else json.loads(Path(path_or_dict).read_text())
    runs = d.get("runs", [])
    mem = d.get("memory_during_run", {})
    setg = d.get("omlx_model_settings", {})
    tpss = [r["decode_tps"] for r in runs]
    import statistics
    rec = dict(
        kind="silicon",
        source=source,
        tps=d.get("decode_tps_median",
                  statistics.median(tpss) if tpss else None),
        tps_stats=dict(median=d.get("decode_tps_median"),
                       min=d.get("decode_tps_min"),
                       max=d.get("decode_tps_max"),
                       n_runs=len(runs)),
        ttft_s=d.get("ttft_s_median"),
        residency=setg.get("moe_expert_offload_resident_fraction"),
        experts_per_layer_resident=mem.get("resident_experts_per_layer_approx"),
        footprint_gib=mem.get("omlx_actual_gb"),
        expert_gib_resident=mem.get("expert_tables_resident_gb"),
        full_model_gib=mem.get("omlx_full_model_gb"),
        config=dict(model=d.get("model"), host=d.get("host"),
                    ram_gib=d.get("ram_gib"),
                    omlx_version=d.get("omlx_version"),
                    max_tokens=d.get("max_tokens"),
                    mtp_enabled=setg.get("mtp_enabled"),
                    ple_ssd_offload=setg.get("qwen4_ple_ssd_offload")),
        host=d.get("host"),
        measured_at=d.get("timestamp"),
        git_commit=d.get("git_commit"),
    )
    if rec["tps"] is None:
        raise ValueError(f"no decode_tps in measured record {source}")
    errs = validate_record(rec)
    assert not errs, f"bad silicon record: {errs}"
    return rec


def gap_report(sim: dict, silicon: dict) -> dict:
    """Field-by-field sim↔silicon comparison + roofline attribution.

    Returns a dict (also printable) with per-field deltas, the tok/s gap
    ratio, and a reconciliation term: the effective DRAM GB/s silicon
    would need for the sim's compute-time model to reproduce the
    measured tok/s (only meaningful when the sim is compute-bound).
    """
    assert sim["kind"] == "sim" and silicon["kind"] == "silicon"
    rep = dict(sim_source=sim["source"], silicon_source=silicon["source"])
    rep["tps"] = dict(sim=sim["tps"], silicon=silicon["tps"],
                      ratio=round(sim["tps"] / silicon["tps"], 3),
                      gap_pct=round(100 * (sim["tps"] / silicon["tps"] - 1),
                                    1))
    for f in ("residency", "footprint_gib", "expert_gib_resident",
              "ttft_s"):
        s, x = sim.get(f), silicon.get(f)
        rep[f] = None if (s is None or x is None) else dict(
            sim=s, silicon=x,
            delta=round(s - x, 3) if isinstance(s, (int, float)) else None)
    # roofline attribution from the sim's two terms
    ct, st = sim.get("compute_ms_per_tok"), sim.get("stream_ms_per_tok")
    rep["sim_terms_ms"] = dict(compute=ct, stream=st, limiter=sim.get("limiter"))
    rec = {}
    if ct and st and sim.get("floor_gib") is not None:
        # DRAM bytes/token implied by the sim compute term at its own eff BW
        # eff_bw was calibrated at 293 GB/s (sim_paging DRAM_EFF on 546 peak)
        rep["sim_derived"] = dict(
            dram_eff_gbs=293.0,
            implied_ms_per_tok=max(ct, st),
            implied_tps=round(1000.0 / max(ct, st), 2))
        # what eff BW silicon would need, holding sim byte traffic fixed:
        # ms_needed = 1000/tps_silicon ; scale ct by that ratio
        ms_needed = 1000.0 / silicon["tps"]
        rec["dram_eff_gbs_silicon_needed_if_compute_bound"] = round(
            293.0 * ct / ms_needed, 1)
        rec["silicon_ms_per_tok"] = round(ms_needed, 1)
        rec["unexplained_ms_per_tok"] = round(
            ms_needed - max(ct, st), 1)
        rep["reconciliation"] = rec
    return rep


def format_gap_report(rep: dict) -> str:
    lines = [f"sim↔silicon gap report",
             f"  sim:     {rep['sim_source']}",
             f"  silicon: {rep['silicon_source']}", ""]
    t = rep["tps"]
    lines.append(f"  tok/s:    sim {t['sim']}  vs  silicon {t['silicon']}"
                 f"   ratio {t['ratio']}  (sim {t['gap_pct']:+}% off)")
    for f in ("residency", "footprint_gib", "expert_gib_resident", "ttft_s"):
        v = rep.get(f)
        if v:
            lines.append(f"  {f:20s} sim {v['sim']}  silicon {v['silicon']}"
                         f"  delta {v['delta']:+}")
    st_ = rep.get("sim_terms_ms")
    if st_:
        lines.append(f"  sim roofline: compute {st_['compute']} ms/tok"
                     f" vs stream {st_['stream']} ms/tok"
                     f" -> limiter {st_['limiter']}")
    rc = rep.get("reconciliation", {})
    if rc:
        lines.append(f"  silicon needs {rc['silicon_ms_per_tok']} ms/tok;"
                     f" sim's limiting term is {max(st_['compute'], st_['stream'])}"
                     f" ms -> {rc['unexplained_ms_per_tok']} ms/tok"
                     " unexplained by sim terms")
        lines.append(f"  if compute-bound with sim bytes, silicon implies"
                     f" {rc['dram_eff_gbs_silicon_needed_if_compute_bound']}"
                     " GB/s eff DRAM (sim assumes 293)")
    return "\n".join(lines)
