"""Per-process disk IO + page-in accounting for the oMLX server during a decode.

proc_pid_rusage(RUSAGE_INFO_V4) on the server pid (same user, no sudo):
ri_diskio_bytesread = bytes this process caused to be read from storage
(pread misses + mmap page faults), ri_pageins = page-ins. Compared with the
whole-disk iostat number from measure_ssd_per_token.py, this tells whether the
decode-time SSD traffic is the server's own (expert slabs + PLE rows) or
something else on the box.

usage: python3 proc_diskio.py PID --max-tokens 256 --out f.json
"""
import argparse, ctypes, ctypes.util, json, os, subprocess, sys, time
from pathlib import Path

libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)


class RUsageV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
        "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
        "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
        "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
        "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
        "ri_diskio_byteswritten", "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance",
        "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
        "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive",
        "ri_billed_system_time", "ri_serviced_system_time", "ri_logical_writes",
        "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles",
        "ri_billed_energy", "ri_serviced_energy", "ri_interval_max_phys_footprint",
        "ri_runnable_time")]


def usage(pid):
    r = RUsageV4()
    if libproc.proc_pid_rusage(int(pid), 4, ctypes.byref(r)) != 0:
        raise OSError(ctypes.get_errno(), "proc_pid_rusage")
    return dict(diskio_read=r.ri_diskio_bytesread, pageins=r.ri_pageins,
                user_ns=r.ri_user_time, sys_ns=r.ri_system_time,
                footprint=r.ri_phys_footprint)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pid")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    here = Path(__file__).resolve().parent
    tmp = "/tmp/proc_diskio_ssd.json"
    cmd = [sys.executable, str(here / "measure_ssd_per_token.py"),
           "--max-tokens", str(a.max_tokens), "--out", tmp]
    if a.prompt:
        cmd += ["--prompt", a.prompt]
    u0 = usage(a.pid); t0 = time.time()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    u1 = usage(a.pid); dt = time.time() - t0
    s = json.load(open(tmp))
    n = s["tokens"]
    # iostat MB/token is decode-window only; whole-window disk MB approximated
    disk_window_MB = sum(x[3] for x in s["samples"] if 0 <= x[0] <= dt)
    d = {k: u1[k] - u0[k] for k in ("diskio_read", "pageins", "user_ns", "sys_ns")}
    res = dict(kind="measured", host=os.uname().nodename,
               timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"), pid=int(a.pid),
               tokens=n, decode_tps=s["decode_tps"], ttft_s=s["ttft_s"],
               window_s=round(dt, 1),
               proc_diskio_read_MB=round(d["diskio_read"] / 1e6, 1),
               proc_pageins=d["pageins"],
               proc_pagein_MB=round(d["pageins"] * os.sysconf("SC_PAGE_SIZE") / 1e6, 1),
               proc_cpu_s=round((d["user_ns"] + d["sys_ns"]) / 1e9, 2),
               disk_iostat_window_MB=round(disk_window_MB, 1),
               iostat_decode_MB_per_token=s["decode_disk_MB_per_token"],
               proc_share_of_disk=round(d["diskio_read"] / 1e6 / disk_window_MB, 3)
               if disk_window_MB else None,
               note="window = prefill+decode+~3.5s tail; ri_* times in Mach absolute units "
                    "(1 ns on Apple Silicon? verify); proc_cpu_s is indicative only")
    print(json.dumps(res, indent=1))
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
