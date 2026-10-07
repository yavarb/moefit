# Changelog

## 2026-10-07 — Public tip cleanup

Public docs now carry only product pitch, install/run, and measured keeps.
Removed private lab ledgers (silicon-proof tables, design ledgers, host /
location labels). Measured artifacts renamed to hardware labels
(`measured_m4max_36gb*.json`). Headline keeps:

- M4 Max 128 GB full-fit: **57.4 tok/s** decode measured
- M4 Max Studio 36 GB paging: **15.55 tok/s** steady (n≥1024); short-run 13.0
- Exact routing-replay on repeated prompts: **~+40%** (repeat-only)

Simulator default remains the serial miss model; see SETUP.md for the
tier table used by `check_docs.py`.

## Earlier

See git history for prior experiment notes. They are no longer summarized
on the public tip.
