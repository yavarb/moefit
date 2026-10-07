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
    # SSD-counter blob (measure_ssd_per_token.py): flat keys, no runs
    if not d.get("runs"):
        rec = dict(
            kind="silicon", source=source, tps=d["decode_tps"],
            ttft_s=d.get("ttft_s"),
            config=dict(host=d.get("host"), max_tokens=d.get("max_tokens"),
                        model=d.get("model"),
                        n_decode_samples=d.get("n_decode_samples")),
            host=d.get("host"), measured_at=d.get("timestamp"),
        )
        if d.get("decode_disk_MBps") and d.get("decode_disk_MB_per_token"):
            rec["ssd_meas_mbps"] = d["decode_disk_MBps"]
            rec["ssd_meas_mb_per_tok"] = d["decode_disk_MB_per_token"]
            rec["ssd_meas_iops"] = d.get("decode_iops")
            rec["ssd_io_kb"] = d.get("decode_avg_KB_per_io")
            rec["disk_ms_per_tok"] = round(
                d["decode_disk_MB_per_token"]
                / d["decode_disk_MBps"] * 1000.0, 1)
        errs = validate_record(rec)
        assert not errs, f"bad silicon record: {errs}"
        return rec
    runs = d.get("runs", [])
    mem = d.get("memory_during_run", {})
    setg = d.get("omlx_model_settings", {})
    tpss = [r["decode_tps"] for r in runs]
    import statistics
    lat = {}
    run_co = {}
    all_gaps = []
    for i, r in enumerate(runs):
        gaps = r.get("per_token_ms")
        if gaps is None and r.get("chunk_ts_s") and r.get("chunk_tokens"):
            ts, cts = r["chunk_ts_s"], r["chunk_tokens"]
            gaps = [1000.0 * (b - a) / max(t, 1)
                    for a, b, t in zip(ts, ts[1:], cts[1:])]
        if gaps:
            lat[r.get("run", i)] = latency_stats_from_deltas(gaps)
            all_gaps.extend(gaps)
            co = detect_coalescing(gaps)
            multi = (r.get("chunk_tokens")
                     and any(t > 1 for t in r["chunk_tokens"]))
            if multi and not co["coalesced"]:
                co = dict(co, coalesced=True,
                          note="direct evidence: chunk(s) carry >1 token")
            run_co[r.get("run", i)] = co
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
    if all_gaps:
        rec["tok_gap_ms"] = latency_stats_from_deltas(all_gaps)
        rec["tok_gap_ms_per_run"] = lat
        if any(c.get("coalesced") for c in run_co.values()):
            # flag with the aggregate over all runs' gaps
            rec["tok_gap_coalesced"] = detect_coalescing(all_gaps)
            rec["tok_gap_coalesced"]["runs_flagged"] = sorted(run_co)
    if rec["tps"] is None:
        raise ValueError(f"no decode_tps in measured record {source}")
    errs = validate_record(rec)
    assert not errs, f"bad silicon record: {errs}"
    return rec


def latency_stats_from_deltas(delta_ms) -> dict:
    """Per-token decode latency distribution from streamed deltas.

    `delta_ms` = list of inter-chunk gaps in ms (first delta after TTFT
    through the last). SSE servers coalesce chunks, so divide by the
    tokens per chunk when known; the caller supplies per-TOKEN gaps.
    Returns p50/p90/p95/p99/mean/max in ms, usable as RunRecord fields
    `tok_gap_ms_*` so silicon records carry a distribution, not just a
    median tok/s.
    """
    import statistics
    d = sorted(float(x) for x in delta_ms if x >= 0)
    if not d:
        raise ValueError("empty delta list")
    n = len(d)

    def pct(p):
        i = min(int(round(p / 100.0 * (n - 1))), n - 1)
        return round(d[i], 2)
    return dict(n=n, p50=pct(50), p90=pct(90), p95=pct(95), p99=pct(99),
                mean=round(statistics.fmean(d), 2), max=round(d[-1], 2))


# Measured on Santa Cruz by lead_silicon (results/gap_santa_cruz.json,
# commit 463c856): per-layer-step miss-resolution latency is
# A + B*k ms for k missed experts; install is per-expert; sync is per
# layer-step. Drive microbench: 3.8-5.6 GB/s on the same pattern, so
# these SERIAL constants describe oMLX's resolve discipline, not the
# SSD ceiling. compute_ms is the one ASSUMED term (18.1 = sim DRAM_EFF
# at 546 GB/s; glm_fidelity: 12.1-12.9 tps spread is within run noise).
SERIAL_CONSTANTS_MEASURED = dict(
    io_A_ms=0.20, io_B_ms=0.52, install_ms=0.30, sync_ms=0.122,
    expert_mb=2.765)


def serial_model_from_misses(M, constants=None, compute_ms=18.1):
    """Serial-latency time model (lead_silicon, measured constants).

    M: (T, L) per-token per-layer SYNC miss counts, as returned by
    sim_paging.simulate(..., want_misses=True)[3]. Returns the ms/token
    breakdown (compute / io / install / sync), serial-model tps, and
    the miss stats the breakdown rests on. SIMULATED prediction built
    from MEASURED latency constants.
    """
    import numpy as np
    c = dict(SERIAL_CONSTANTS_MEASURED)
    c.update(constants or {})
    M = np.asarray(M)
    T, L = M.shape
    per_layer = M.sum(axis=0) / T                # misses/tok per layer
    steps = (M > 0).sum(axis=0) / T              # layer-steps w/ >=1 miss
    io_ms = sum(c["io_A_ms"] * steps[li] + c["io_B_ms"] * per_layer[li]
                for li in range(L))
    install_ms = c["install_ms"] * float(per_layer.sum())
    sync_ms = c["sync_ms"] * L
    total = compute_ms + io_ms + install_ms + sync_ms
    return dict(
        breakdown_ms=dict(compute=round(compute_ms, 1),
                          io=round(io_ms, 1),
                          install=round(install_ms, 1),
                          sync=round(sync_ms, 1)),
        total_ms=round(total, 1),
        tps=round(1000.0 / total, 2),
        miss_experts_per_tok=round(float(per_layer.sum()), 1),
        frac_layer_steps_with_miss=round(
            float(steps.mean()), 3),
        constants=c, compute_ms_assumed=compute_ms,
    )


def detect_coalescing(delta_ms):
    """Flag SSE chunk coalescing in per-token gap data.

    Coalescing servers (vllm-style; NOT omlx, which streams ~1
    token/chunk) emit bursts of near-zero inter-chunk gaps followed by
    one large gap. Per-token gaps derived from such streams (dividing
    each inter-chunk gap by the LATER chunk's token count) misattribute
    the burst time and can distort the S-vs-Q signature (astra_local_exp
    probe, T8). Detection: >= 20% of gaps < 2 ms AND >= 10% of gaps
    > 5x median (bimodal burst signature). Direct evidence also counts:
    any chunk carrying > 1 token.
    """
    d = [float(x) for x in delta_ms if x >= 0]
    n = len(d)
    if n < 10:
        return dict(coalesced=False, frac_sub2ms=0.0,
                    frac_gt_5x_median=0.0, note="too few gaps")
    import statistics
    med = statistics.median(d)
    frac_sub = sum(1 for x in d if x < 2.0) / n
    frac_big = sum(1 for x in d if x > 5 * max(med, 1.0)) / n
    return dict(coalesced=(frac_sub >= 0.20 and frac_big >= 0.10),
                frac_sub2ms=round(frac_sub, 3),
                frac_gt_5x_median=round(frac_big, 3), median_ms=round(med, 2))


def serial_signature_check(tok_gap_ms):
    """S-vs-Q discriminator (T7's falsifiable signature, 2026-10-07).

    Model S (serial-latency miss resolution) predicts a bursty per-token
    gap: p95/mean ~ 1.47-1.51 (locked synth, cap143 true-LRU). A
    smoothed byte-backlog model (Q) with the same mean predicts
    p95/mean ~ 1.0. Accepts a tok_gap_ms stats dict (from
    latency_stats_from_deltas) or a raw list of per-token gaps.
    Returns dict(ratio, verdict). Verdicts: "serial (S)", "byte-backlog
    (Q)", or "ambiguous". Threshold 1.3 per
    results/MEASURED_VS_SIM_36GB.md. For silicon records use
    signature_from_silicon_record, which marks coalesced-stream inputs
    inadmissible (T8 probe).
    """
    if isinstance(tok_gap_ms, dict):
        mean, p95 = tok_gap_ms["mean"], tok_gap_ms["p95"]
    else:
        s = latency_stats_from_deltas(tok_gap_ms)
        mean, p95 = s["mean"], s["p95"]
    if mean <= 0:
        raise ValueError("mean must be > 0")
    ratio = round(p95 / mean, 3)
    if ratio >= 1.3:
        verdict = "serial (S)"
    elif ratio <= 1.1:
        verdict = "byte-backlog (Q)"
    else:
        verdict = "ambiguous"
    return dict(p95_over_mean=ratio, verdict=verdict,
                s_threshold=1.3, q_threshold=1.1)


def signature_from_silicon_record(rec: dict) -> dict:
    """S-vs-Q signature for a silicon record, honoring coalescing.

    If the record carries tok_gap_coalesced (normalizer flag: SSE chunk
    coalescing detected in the per-token gaps), the verdict becomes
    "inadmissible (chunk coalescing ...)" — a coalesced stream
    misattributes per-token time and can flip the S-vs-Q thresholds
    (astra_local_exp probe, T8); rerun with per-token timestamps
    (omlx streams ~1 token/chunk, so collector runs there are clean).
    """
    tg = rec.get("tok_gap_ms")
    if not tg:
        raise ValueError("record has no tok_gap_ms")
    sig = serial_signature_check(tg)
    co = rec.get("tok_gap_coalesced")
    if co and co.get("coalesced"):
        sig = dict(sig,
                   verdict="inadmissible (chunk coalescing detected: "
                           f"frac<2ms {co.get('frac_sub2ms')}, "
                           f"frac>5xmed {co.get('frac_gt_5x_median')}) - "
                           "rerun with per-token timestamps; omlx streams "
                           "~1 token/chunk")
    return sig


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
    # measured-SSD accounting (iostat blob): the stream term evaluated
    # with MEASURED concurrent disk MB/s and MEASURED physical MB/tok.
    # NOTE: measured MBps is a duty-cycle average under oMLX's serial
    # per-layer miss resolution (lead_silicon microbench: the drive
    # does 3.8-5.6 GB/s on the same pattern) - not a drive ceiling.
    m_mbpt = silicon.get("ssd_meas_mb_per_tok")
    m_mbps = silicon.get("ssd_meas_mbps")
    if m_mbpt and m_mbps:
        rep["ssd_measured"] = dict(
            mb_per_tok=m_mbpt, mbps=m_mbps,
            iops=silicon.get("ssd_meas_iops"),
            io_kb=silicon.get("ssd_io_kb"),
            disk_ms_per_tok=silicon.get("disk_ms_per_tok"))
        sim_mbpt = (sim.get("config") or {}).get("ssd_mb_per_tok_total")
        if sim_mbpt:
            rep["ssd_measured"]["sim_logical_mb_per_tok"] = sim_mbpt
            rep["ssd_measured"]["traffic_ratio_meas_over_sim"] = round(
                m_mbpt / sim_mbpt, 3)
        ms_needed = 1000.0 / silicon["tps"]
        rep["ssd_measured"]["silicon_ms_per_tok"] = round(ms_needed, 1)
        rep["ssd_measured"]["residual_ms_per_tok"] = round(
            ms_needed - silicon["disk_ms_per_tok"], 1)
    # per-token latency distribution, when the silicon run has it
    tg = silicon.get("tok_gap_ms")
    if tg:
        rep["tok_gap_ms"] = tg
        rep["s_vs_q_signature"] = signature_from_silicon_record(silicon)
        if silicon.get("tok_gap_coalesced"):
            rep["tok_gap_coalesced"] = silicon["tok_gap_coalesced"]
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
                     " unexplained by sim terms (sim-spec accounting)")
        lines.append(f"  if compute-bound with sim bytes, silicon implies"
                     f" {rc['dram_eff_gbs_silicon_needed_if_compute_bound']}"
                     " GB/s eff DRAM (sim assumes 293)")
    sm = rep.get("ssd_measured")
    if sm:
        lines.append(f"  MEASURED SSD: {sm['mb_per_tok']} MB/tok at"
                     f" {sm['mbps']} MB/s ({sm['iops']} IOPS,"
                     f" {sm['io_kb']} KB/IO)"
                     f" = {sm['disk_ms_per_tok']} ms/tok disk time"
                     f" vs {sm['silicon_ms_per_tok']} ms/tok measured"
                     f" -> residual {sm['residual_ms_per_tok']} ms")
        if "sim_logical_mb_per_tok" in sm:
            lines.append(f"  traffic: sim logical"
                         f" {sm['sim_logical_mb_per_tok']} MB/tok vs"
                         f" measured physical {sm['mb_per_tok']}"
                         f" (ratio {sm['traffic_ratio_meas_over_sim']});"
                         " measured MB/s is a duty-cycle average under"
                         " oMLX serial miss resolution, NOT the drive"
                         " ceiling (microbench: 3.8-5.6 GB/s)")
    tg = rep.get("tok_gap_ms")
    if tg:
        lines.append(f"  tok-gap ms: p50 {tg['p50']} p90 {tg['p90']}"
                     f" p95 {tg['p95']} p99 {tg['p99']}"
                     f" max {tg['max']} (n={tg['n']})")
    sig = rep.get("s_vs_q_signature")
    if sig:
        lines.append(f"  S-vs-Q signature: p95/mean {sig['p95_over_mean']}"
                     f" -> {sig['verdict']}"
                     f" (S predicts ~1.5, byte-backlog ~1.0;"
                     f" thresholds {sig['s_threshold']}/{sig['q_threshold']})")
    return "\n".join(lines)
