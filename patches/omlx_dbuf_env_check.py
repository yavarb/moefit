#!/usr/bin/env python3
"""T4 handoff to silicon_dbuf (T9): staged-install INVENTORY + TOGGLE.

Run THIS FIRST on the target box (Santa Cruz, before any restart):

  python3 omlx_dbuf_env_check.py            # inventories the live server's file
  python3 omlx_dbuf_env_check.py --file PATH  # any copy of moe_expert_offload.py

What it decides (in seconds, zero risk — read-only):
  CASE B: the file already contains the upstream staged-install machinery
          (io pool, read-ahead window, install-on-completion), gated by
          env vars. The double-buffer ON/OFF arms are then PURE
          ENVIRONMENT on restart — no file edit, no code patch.
  AMENDED (T9's read of the INSTALLED d420b305 file, cycle 26): the
          machinery's DEFAULT on the box is ON (IO_WORKERS default 12,
          window 4*12=48, decode-overlap default 1; workers<=1 = serial
          path). So the arms are:
            ON  (staged install): the DEFAULT environment — no vars set;
                                 the measured 15.55 baseline IS this arm
            OFF (serial):          OMLX_MOE_OFFLOAD_IO_WORKERS=1
  CASE A: the file predates the machinery (fully-serial _ensure_ids:
          fetch->install->fetch->install per miss). The staged loop must
          be back-ported — the exact upstream block to insert, with
          anchors, is in results/T4_DBUF_SILICON_PATCH.md (Case A).

NOTE this script detects PRESENCE, not the DEFAULT VALUE: if the box's
file matches d420b305, trust T9's source read (default ON) over any
generic assumption. Lead_silicon's microbench measured A + B*k ms per
layer fully exposed on the box — on a DB-ON-default file that means the
exposed term is fetch latency + trailing install, not serial install
per expert (see prereg Amendment 2's wash_reading).
"""
import argparse
import hashlib
import os
import re
import sys

# Markers of the upstream staged-install machinery (present in
# omlxenv/omlx-ref moe_expert_offload.py, md5s 61ead257/a12bba2a).
MARKERS = {
    "io_pool_env": "OMLX_MOE_OFFLOAD_IO_WORKERS",
    "io_batch_env": "OMLX_MOE_OFFLOAD_IO_BATCH",
    "pool_def": "_IO_WORKERS",
    "read_ahead": "def _read_ahead",
    "windowed_ensure": "def prefetch(",  # inner fn of windowed _ensure_ids
    "install_on_payload": "payload: list | None = None",
}


def inventory(path):
    src = open(path, "rb").read()
    txt = src.decode("utf-8", errors="replace")
    md5 = hashlib.md5(src).hexdigest()
    found = {k: (v in txt) for k, v in MARKERS.items()}
    has_machinery = (found["pool_def"] and found["read_ahead"]
                     and found["windowed_ensure"])
    return md5, found, has_machinery


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None,
                    help="path to moe_expert_offload.py; default: "
                         "auto-locate the file the live omlx-server "
                         "process has loaded (via lsof), falling back "
                         "to the Homebrew Cellar path")
    a = ap.parse_args()
    path = a.file
    if path is None:
        import subprocess
        # 1) what the running server actually loaded
        try:
            pid = subprocess.run(["pgrep", "-x", "omlx-server"],
                                 capture_output=True, text=True
                                 ).stdout.split()
            for p in pid:
                out = subprocess.run(
                    ["lsof", "-p", p], capture_output=True, text=True
                ).stdout
                for line in out.splitlines():
                    if "moe_expert_offload.py" in line:
                        path = line.split()[-1]
                        break
                if path:
                    break
        except Exception:
            pass
        # 2) Homebrew Cellar default (design_inventor's source-read path)
        if path is None:
            import glob
            hits = sorted(glob.glob(
                "/opt/homebrew/Cellar/omlx/*/libexec/lib/python*/"
                "site-packages/omlx/patches/moe_expert_offload.py"))
            path = hits[-1] if hits else None
    if path is None or not os.path.exists(path):
        sys.exit("could not locate moe_expert_offload.py — pass --file")
    md5, found, has_machinery = inventory(path)
    print(f"file: {path}")
    print(f"md5:  {md5}")
    for k, v in found.items():
        print(f"  {'FOUND ' if v else 'missing'}  {k}: {MARKERS[k]!r}")
    print()
    if has_machinery:
        print("CASE B — staged-install machinery IS present in the file.")
        print("ON/OFF is pure environment at restart; NO file edit needed.")
        print("AMENDED ARM SEMANTICS (prereg Amendment 2): if this file is")
        print("the box's d420b305, its DEFAULT is ALREADY ON (workers 12):")
        print("  ON  arm (staged install): DEFAULT env, no vars set "
              "(= the 15.55 baseline arm)")
        print("  OFF arm (serial):          OMLX_MOE_OFFLOAD_IO_WORKERS=1")
        print("Score A/B/A against results/t4_dbuf_silicon_prereg.json "
              "amendment2 (ON pinned at 15.55; OFF band 14.58-15.24 "
              "@install 0.18).")
    else:
        print("CASE A — the file predates the staged-install machinery "
              "(stock serial resolve).")
        print("Back-port required: see results/T4_DBUF_SILICON_PATCH.md "
              "Case A (verbatim upstream block + anchors).")
    print()
    print("Reference md5s known to the lab:")
    print("  d420b305...  Homebrew 0.7.0 on Santa Cruz (true-LRU "
          "ExpertCache, design_inventor source-read)")
    print("  61ead257...  omlxenv local copy (windowed staged install, "
          "this file's mechanism source)")
    print("  a12bba2a...  omlx-ref HEAD 79f4488 (decayed-count variant)")


if __name__ == "__main__":
    main()
