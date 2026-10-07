#!/usr/bin/env python3
"""T4 handoff to silicon_dbuf (T9): staged-install INVENTORY + TOGGLE.

Run THIS FIRST on the target box (M4 Max 36 GB, before any restart):

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

# Markers of the upstream staged-install machinery. Naming across variants
# differs: the omlxenv reference (md5 61ead257) has `def _read_ahead`; the
# T2-sidecar+T3-LIP MERGED file (214e8823) implements the same windowed
# read-ahead as an inner `prefetch()` closure in _ensure_ids. Detection is
# therefore STRUCTURAL: an IO pool definition + the env gate + EITHER
# read-ahead form. A fully-serial file (true CASE A) has none of these.
MARKERS = {
    "io_pool_env": "OMLX_MOE_OFFLOAD_IO_WORKERS",
    "io_batch_env": "OMLX_MOE_OFFLOAD_IO_BATCH",
    "pool_def": "_IO_WORKERS",
    "read_ahead_named": "def _read_ahead",
    "windowed_ensure": "def prefetch(",  # inner fn of windowed _ensure_ids
    "install_on_payload": "payload: list | None = None",
    "overlap_env": "OMLX_MOE_OFFLOAD_OVERLAP",
}


def inventory(path):
    src = open(path, "rb").read()
    txt = src.decode("utf-8", errors="replace")
    md5 = hashlib.md5(src).hexdigest()
    found = {k: (v in txt) for k, v in MARKERS.items()}
    read_ahead = found["read_ahead_named"] or found["windowed_ensure"]
    has_machinery = (found["io_pool_env"] and found["pool_def"]
                     and read_ahead)
    return md5, found, has_machinery


def live_env():
    """Probe what CAN be verified about the RUNNING omlx-server.

    macOS does NOT expose another process's environment to ps/launchctl
    on this box (verified: ps eww, ps -E, and launchctl procinfo all
    return nothing) — so the live server's OMLX_* arm config (IO_WORKERS,
    IO_BATCH, OVERLAP, ADMISSION) is UNVERIFIABLE from outside. A
    restart wrapper can leave the server in a non-default env with no
    visible trace; arms collected against it under a default-env
    assumption would be contaminated.

    FAIL-CLOSED guidance: arms must set their own env via their own
    restart (as t9_batch1_arm.sh does) rather than trust the ambient
    server. What this probe DOES verify: the sidecar flag file, and a
    multiple-omlx-server hazard (port conflict / wrong server on :8000).
    """
    import os
    import subprocess
    pids = subprocess.run(["pgrep", "-x", "omlx-server"],
                          capture_output=True, text=True).stdout.split()
    if not pids:
        print("no omlx-server process found")
        return
    print(f"omlx-server process(es): {pids}")
    if len(pids) > 1:
        print("HAZARD: multiple omlx-server processes — port conflict / "
              "the :8000 server may not be the one you think; kill the "
              "stale one before any arm")
    # env: attempt the read, but NEVER report 'default' from absence
    for p in pids:
        ps = subprocess.run(["ps", "eww", "-o", "command=", "-p", p],
                            capture_output=True, text=True).stdout
        vis = sorted(set(re.findall(r"(OMLX_[A-Z_]+)=\S*", ps)))
        if vis:
            print(f"pid {p}: OMLX_* env VISIBLE: {vis} "
                  "(non-default config — arm accordingly)")
        else:
            print(f"pid {p}: OMLX_* env NOT READABLE on this macOS "
                  "(ps eww / launchctl procinfo all empty) — arm config "
                  "UNKNOWN; FAIL-CLOSED: do NOT assume default env")
    # sidecar flag: the one arm state that IS verifiable
    flag = "/tmp/omlx_sidecar_on"
    print(f"sidecar flag {flag}: "
          + ("PRESENT (T2 sidecar arm ACTIVE)" if os.path.exists(flag)
             else "absent (sidecar off)"))
    print()
    print("Arm defaults on this file: IO_WORKERS=12 (pool ON), "
          "IO_BATCH=48 (window ON), OVERLAP=1, ADMISSION=0.")
    print("RULE: every arm restarts omlx with its OWN env (the "
          "t9_batch1_arm.sh restart() pattern) — never collect against "
          "a server whose env you did not set in this arm's own restart.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None,
                    help="path to moe_expert_offload.py; default: "
                         "auto-locate the file the live omlx-server "
                         "process has loaded (via lsof), falling back "
                         "to the Homebrew Cellar path")
    ap.add_argument("--live", action="store_true",
                    help="probe the RUNNING omlx-server's OMLX_* env "
                         "(ps eww) and report which arm config is live; "
                         "run this before/after every arm to catch "
                         "servers left in a non-default env")
    a = ap.parse_args()
    if a.live:
        live_env()
        return
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
    print("  d420b305...  Homebrew 0.7.0 on M4 Max 36 GB (true-LRU "
          "ExpertCache, design_inventor source-read; machinery ON "
          "by default per T9)")
    print("  61ead257...  omlxenv local copy (windowed staged install, "
          "named _read_ahead)")
    print("  214e8823...  MERGED T2-sidecar+T3-LIP drop-in (repo "
          "patches/omlx_t2_sidecar_plus_lip.py; machinery VERIFIED "
          "INTACT — windowed read-ahead is an inner prefetch() "
          "closure; DB env gates + OMLX_ADMISSION + sidecar flag all "
          "present; all three patches compose)")
    print("  a12bba2a...  omlx-ref HEAD 79f4488 (decayed-count variant)")


if __name__ == "__main__":
    main()
