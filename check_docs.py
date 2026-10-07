#!/usr/bin/env python3
"""check_docs.py - verify every number in README.md / SETUP.md / FAQ.md that
comes from the simulator against results/sim_paging.json.

Stdlib only. Exit 0 when every documented number matches the shipped table.

    python3 check_docs.py

Checks:
  * the results table in SETUP.md (tok/s for LRU, pinned hot-set, routing
    sidecar at each RAM tier / capacity) cell by cell;
  * the DRAM ceilings quoted in prose (derived from the LRU compute time);
  * the LRU SSD traffic range quoted in prose;
  * the "48 GB at ~N% of a 128 GB Mac" claim.
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TIER_OF = {"24 GB": "24GB-M4", "32 GB": "32GB-M4P", "48 GB": "48GB-M4M",
           "64 GB": "64GB-M4X"}
COL_MODE = {"LRU": "lru", "pinned hot-set": "prior",
            "routing sidecar": "sidecar"}
MEASURED_128GB_TPS = 57.4


def load_rows():
    rows = json.load(open(ROOT / "results/sim_paging.json"))
    return {(r["cap"], r["mode"]): r for r in rows if r["probe_picks"] == 6}


def parse_table(md):
    """first markdown table whose header contains 'LRU' -> header, rows."""
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|") and "LRU" in line:
            header = [c.strip() for c in line.strip("|").split("|")]
            body = []
            for l2 in lines[i + 2:]:
                if not l2.startswith("|"):
                    break
                body.append([c.strip() for c in l2.strip("|").split("|")])
            return header, body
    raise SystemExit("no results table found")


def num(cell):
    m = re.search(r"\d+(?:\.\d+)?", cell.replace("**", ""))
    return float(m.group()) if m else None


def main():
    rows = load_rows()
    problems = []
    setup = (ROOT / "SETUP.md").read_text()
    header, body = parse_table(setup)
    for cells in body:
        tier = TIER_OF[re.match(r"\d+ GB", cells[0]).group()]
        cap = int(num(cells[1]))
        for col, cell in zip(header[2:], cells[2:]):
            mode = COL_MODE[col]
            want = rows[(cap, mode)][f"tps_{tier}"]
            got = num(cell)
            if got != want:
                problems.append(f"SETUP table {cells[0]} cap={cap} {col}: "
                                f"doc says {got}, JSON says {want}")
            ssd = re.search(r"(\d+) MB/tok", cell)
            if ssd:
                want_ssd = rows[(cap, mode)][f"ssdMB_{tier}"]
                if int(ssd.group(1)) != int(want_ssd):
                    problems.append(f"SETUP table SSD note {cells[0]} cap={cap}"
                                    f" {col}: doc {ssd.group(1)} JSON {want_ssd}")
    print(f"SETUP.md table: {len(body)} rows checked")

    # ceilings quoted in prose: the best LRU tok/s per tier (DRAM-bound rows)
    ceilings = {t: max(r[f"tps_{t}"] for (c, m), r in rows.items()
                       if m == "lru" and f"tps_{t}" in r)
                for t in ("24GB-M4", "32GB-M4P", "48GB-M4M")}
    prose = "\n".join((ROOT / f).read_text()
                      for f in ("README.md", "SETUP.md", "FAQ.md"))
    for tier, c in ceilings.items():
        if f"{c}" not in prose:
            problems.append(f"ceiling {c} tok/s for {tier} not quoted anywhere")
    print("ceilings from JSON:", ceilings)

    # LRU SSD traffic range quoted as "about 140 to 340 MB per token"
    lo, hi = rows[(128, "lru")]["ssdMB_32GB-M4P"], rows[(32, "lru")]["ssdMB_24GB-M4"]
    m = re.search(r"about (\d+) to (\d+) MB per token", prose)
    if not m:
        problems.append("LRU SSD traffic range sentence missing")
    elif abs(int(m.group(1)) - lo) > 5 or abs(int(m.group(2)) - hi) > 5:
        problems.append(f"LRU SSD range doc {m.groups()} vs JSON {lo},{hi}")

    # 48 GB vs 128 GB claim
    pct = round(100 * rows[(128, "lru")]["tps_48GB-M4M"] / MEASURED_128GB_TPS)
    pct_prior = round(100 * rows[(128, "prior")]["tps_48GB-M4M"] / MEASURED_128GB_TPS)
    m = re.search(r"about (\d+)% of (?:the|a) 128 GB", prose)
    if not m:
        problems.append("48 GB vs 128 GB percentage sentence missing")
    elif int(m.group(1)) not in (pct, pct_prior, round((pct + pct_prior) / 2)):
        problems.append(f"48GB/128GB claim {m.group(1)}% vs JSON LRU {pct}% "
                        f"/ pinned {pct_prior}%")
    print(f"48 GB cap 128 vs measured 128 GB: LRU {pct}% pinned {pct_prior}%")

    if problems:
        print("\n".join("MISMATCH: " + p for p in problems))
        sys.exit(1)
    print("all documented numbers match results/sim_paging.json")


if __name__ == "__main__":
    main()
