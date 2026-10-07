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


if __name__ == "__main__":
    for fn in [test_validate_record, test_sim_record_from_row,
               test_silicon_record_and_gap]:
        fn()
        print("ok", fn.__name__)
