"""Real-checkpoint coalescer ON/OFF with physical APPLE SSD byte deltas + F_NOCACHE.
Shorter than t6_ssd_proof (48 batches x 2 reps) so it fits the silicon slot.
"""
import fcntl, json, os, plistlib, random, statistics, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moefit.expert_pack import components
from moefit.coalesced_read import read_exact
from moefit.expert_batch_read import read_experts


def disk():
    data = plistlib.loads(subprocess.check_output(
        ['ioreg', '-a', '-r', '-c', 'IOBlockStorageDriver', '-d', '2']))
    best = None
    for item in data:
        stats = item.get('Statistics') or {}
        nbytes = stats.get('Bytes (Read)')
        if nbytes is None:
            continue
        kids = item.get('IORegistryEntryChildren') or []
        names = ' '.join(str(c.get('IORegistryEntryName') or '') for c in kids)
        if 'APPLE SSD' in names.upper():
            return int(nbytes)
        if best is None or int(nbytes) > best:
            best = int(nbytes)
    if best:
        return best
    raise RuntimeError('physical counter unavailable')


def main():
    model, out = map(Path, sys.argv[1:3])
    out.parent.mkdir(parents=True, exist_ok=True)
    cs = [components(model, li) for li in range(48)]
    paths = {c['file'] for group in cs for c in group}
    fds = {p: os.open(p, os.O_RDONLY) for p in paths}
    rows = []
    try:
        for fd in fds.values():
            fcntl.fcntl(fd, 48, 1)  # F_NOCACHE
            fcntl.fcntl(fd, 45, 0)  # F_RDAHEAD off
        a = disk(); t = time.perf_counter(); time.sleep(1); b = disk()
        idle = (b - a) / (time.perf_counter() - t)
        rng = random.Random(6025)
        # 64 random batches, 10 experts — ~4.6 GB useful per arm (10*9*819200*64)
        random_batches = [(i % 48, rng.sample(range(512), 10)) for i in range(64)]
        with ThreadPoolExecutor(4) as pool:
            def fetch(arg):
                e, c = arg
                return read_exact(fds[c['file']], c['size'], c['offset'] + e * c['size'])
            for rep in range(2):
                arms = [('OFF',), ('ON',)]
                if rep:
                    arms = list(reversed(arms))
                for (arm,) in arms:
                    # Drop UBC for these fds again
                    for fd in fds.values():
                        fcntl.fcntl(fd, 48, 1)
                    time.sleep(0.5)
                    p0 = disk(); start = time.perf_counter(); useful = 0; check = 0
                    for li, ids in random_batches:
                        group = cs[li]
                        if arm == 'OFF':
                            blocks = list(pool.map(fetch, [(e, c) for e in ids for c in group]))
                        else:
                            got, _ = read_experts(group, fds, ids, executor=pool, lanes=4)
                            blocks = [x[c['name']] for x in got for c in group]
                        useful += sum(len(x) for x in blocks)
                        check += sum(x[0] for x in blocks)
                    seconds = time.perf_counter() - start
                    p1 = disk()
                    r = dict(rep=rep, order='random', arm=arm, seconds=seconds,
                             useful_bytes=useful, useful_GBps=useful / seconds / 1e9,
                             physical_bytes=p1 - p0,
                             physical_over_useful=(p1 - p0) / useful if useful else None,
                             physical_GBps=(p1 - p0) / seconds / 1e9, checksum=check)
                    rows.append(r)
                    print(json.dumps(r), flush=True)
        assert len({r['checksum'] for r in rows}) == 1
        med = {}
        for arm in ('OFF', 'ON'):
            med[arm] = dict(
                useful=statistics.median(r['useful_GBps'] for r in rows if r['arm'] == arm),
                physical=statistics.median(r['physical_GBps'] for r in rows if r['arm'] == arm),
                phys_ratio=statistics.median(r['physical_over_useful'] for r in rows if r['arm'] == arm),
            )
        phys_ok = all(0.5 <= (r['physical_over_useful'] or 0) <= 2.0 for r in rows)
        speedup = med['ON']['physical'] / med['OFF']['physical'] if med['OFF']['physical'] else None
        clear = speedup is not None and speedup >= 1.25 and med['ON']['physical'] > 0.5
        verdict = 'PASS' if (phys_ok and clear) else ('FAIL_CACHE_ONLY' if not phys_ok else 'FAIL_NO_MARGIN')
        report = dict(
            kind='REAL_CHECKPOINT_READS_WITH_PHYSICAL_APPLE_SSD_COUNTERS',
            host=os.uname().nodename, model=str(model),
            idle_read_Bps=idle, workers=4, experts_per_batch=10, batches=64, layers=48,
            cache_control='F_NOCACHE=1,F_RDAHEAD=0; no global purge',
            baseline_random_chunk_peak_GBps=5.02,
            rows=rows, median=med,
            speedup_physical_ON_vs_OFF=speedup,
            vs_baseline_peak={a: med[a]['physical'] / 5.02 for a in med},
            physical_approx_useful=phys_ok, verdict=verdict,
            caveat='Physical counters are whole-disk; controller cache uncontrolled. Decode idle required.',
        )
        out.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: report[k] for k in ('verdict', 'median', 'speedup_physical_ON_vs_OFF',
                                                 'physical_approx_useful', 'vs_baseline_peak')}, indent=2))
    finally:
        for fd in fds.values():
            os.close(fd)


if __name__ == '__main__':
    main()
