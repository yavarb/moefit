"""Sidecar persistence must survive a process restart.

Repro for two bugs in moefit_prefetch.Sidecar:
  1. record() keys entries as "<prefix>:<layer>" but __init__ reloads them
     keyed by the bare prefix hash, so after a restart lookup() never hits.
  2. The default sidecar path lives under ~/Library/Application Support/
     moefit/, and record() appends to the file without creating the
     directory, so the first record() on a fresh Mac raises.

run: python3 tests/test_sidecar.py
"""
import os, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from moefit_prefetch import Sidecar  # noqa: E402


def main():
    fails = 0
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sidecar.jsonl")
        s1 = Sidecar(path)
        s1.record("abc", 3, [1, 2, 3])
        s1.record("abc", 4, [7, 8, 9])
        live = s1.lookup("abc", 3)
        print("same-process lookup:", live)
        s2 = Sidecar(path)            # simulate a restart
        after = s2.lookup("abc", 3)
        print("after-restart lookup:", after)
        if after != [1, 2, 3]:
            print("FAIL: sidecar entries do not survive restart")
            fails += 1
        if s2.lookup("abc", 4) != [7, 8, 9]:
            print("FAIL: second layer entry lost")
            fails += 1

        nested = os.path.join(d, "Library", "Application Support",
                              "moefit", "sidecar.jsonl")
        try:
            Sidecar(nested).record("p", 0, [1])
            print("record() into a missing directory: OK")
        except FileNotFoundError as e:
            print("FAIL: record() into a missing directory raised:", e)
            fails += 1
    if fails:
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
