"""Tests for moefit/metrics.py record schema and gap report."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moefit.metrics import (validate_record, sim_record_from_row,
                            silicon_record_from_measured, gap_report,
                            format_gap_report)

SILICON_FIXTURE = dict(
    kind="measured", host="santacruz", ram_gib=36.0, model="m",
    timestamp="t", git_commit="c", decode_tps_median=13.0,
    decode_tps_min=12.2, decode_tps_max=13.1, ttft_s_median=0.68,
    omlx_version="0.7.0",
    runs=[dict(decode_tps=13.0, ttft_s=0.7)],
    memory_during_run=dict(
        omlx_actual_gb=22.61, expert_tables_resident_gb=18.98,
        omlx_full_model_gb=103.94,
        resident_experts_per_layer_approx=143),
    omlx_model_settings=dict(
        moe_expert_offload_resident_fraction=0.28),
)

SIM_ROW = dict(cap=143, mode="lru", probe_picks=0, throttle="legacy",
               **{"served_48GB-M4M": 0.84, "tps_48GB-M4M": 36.6,
                  "ct_48GB-M4M": 18.1, "st_48GB-M4M": 27.3,
                  "ssdMB_48GB-M4M": 207.0})


def test_validate_record():
    assert validate_record(dict(kind="sim", tps=10, source="x")) == []
    errs = validate_record(dict(kind="bad", source="x"))
    assert any("tps" in e for e in errs) and any("kind" in e for e in errs)


def test_sim_record_from_row():
    rec = sim_record_from_row(SIM_ROW, "48GB-M4M", "fixture")
    assert rec["kind"] == "sim"
    assert abs(rec["residency"] - 143 / 512) < 1e-9
    assert rec["footprint_gib"] == 22.63
    assert rec["limiter"] == "stream"
    assert rec["config"]["ssd_mb_per_tok_total"] == 207.0


def test_silicon_record_and_gap():
    rec = silicon_record_from_measured(SILICON_FIXTURE, "fixture")
    assert rec["kind"] == "silicon" and rec["tps"] == 13.0
    assert rec["residency"] == 0.28
    sim = sim_record_from_row(SIM_ROW, "48GB-M4M", "fixture")
    rep = gap_report(sim, rec)
    assert abs(rep["tps"]["ratio"] - 36.6 / 13.0) < 0.01
    assert rep["reconciliation"]["silicon_ms_per_tok"] == 76.9
    # 1000/13 - max(18.1, 27.3) = 76.9 - 27.3 = 49.6
    assert rep["reconciliation"]["unexplained_ms_per_tok"] == 49.6
    txt = format_gap_report(rep)
    assert "unexplained" in txt and "13.0" in txt


SILICON_SSD_FIXTURE = dict(
    kind="measured", host="santacruz", timestamp="t", max_tokens=256,
    decode_tps=12.71, ttft_s=6.67, n_decode_samples=19,
    decode_disk_MBps=1785.7, decode_disk_MB_per_token=140.5,
    decode_iops=10746.0, decode_avg_KB_per_io=171.4,
)


def test_silicon_ssd_blob_and_measured_accounting():
    rec = silicon_record_from_measured(SILICON_SSD_FIXTURE, "ssd")
    assert rec["kind"] == "silicon" and rec["tps"] == 12.71
    assert rec["ssd_meas_mb_per_tok"] == 140.5
    assert rec["ssd_meas_mbps"] == 1785.7
    # 140.5 / 1785.7 * 1000 = 78.7 ms/tok
    assert rec["disk_ms_per_tok"] == 78.7
    sim = sim_record_from_row(SIM_ROW, "48GB-M4M", "fixture")
    rep = gap_report(sim, rec)
    sm = rep["ssd_measured"]
    assert sm["sim_logical_mb_per_tok"] == 207.0
    assert abs(sm["traffic_ratio_meas_over_sim"] - 140.5 / 207.0) < 0.001
    assert sm["silicon_ms_per_tok"] == 78.7
    # measured accounting closes the gap to within rounding
    assert abs(sm["residual_ms_per_tok"]) <= 0.1
    txt = format_gap_report(rep)
    assert "MEASURED SSD" in txt and "duty-cycle" in txt
    # sim-spec accounting still reported and labeled as such
    assert "sim-spec accounting" in txt


if __name__ == "__main__":
    for fn in [test_validate_record, test_sim_record_from_row,
               test_silicon_record_and_gap,
               test_silicon_ssd_blob_and_measured_accounting]:
        fn()
        print("ok", fn.__name__)
