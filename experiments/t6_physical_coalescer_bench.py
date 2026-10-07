"""M4 Max 36 GB physical-SSD coalescer ON vs OFF. Darwin F_NOCACHE + disk0 byte deltas.
Synthetic full-materialized fixture (no page-cache claim). NOT LLM decode.
"""
import fcntl, hashlib, json, os, plistlib, platform, random, statistics, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moefit.coalesced_read import read_batch, read_exact


def disk_read_bytes():
    """Prefer internal APPLE SSD Bytes(Read); BSD Name is often None on Darwin."""
    data = plistlib.loads(subprocess.check_output(
        ['ioreg', '-a', '-r', '-c', 'IOBlockStorageDriver', '-d', '2']))
    best = None
    for item in data:
        stats = item.get('Statistics') or {}
        nbytes = stats.get('Bytes (Read)')
        if nbytes is None:
            continue
        kids = item.get('IORegistryEntryChildren') or []
        names = ' '.join(str(c.get('IORegistryEntryName') or c.get('BSD Name') or '') for c in kids)
        # Prefer internal Apple SSD; skip external T7 / zero counters
        if 'APPLE SSD' in names.upper() or 'APPLE SSD' in str(item.get('IORegistryEntryName') or '').upper():
            return int(nbytes)
        if best is None or int(nbytes) > best[0]:
            best = (int(nbytes), names)
    if best and best[0] > 0:
        return best[0]
    raise RuntimeError('disk0 physical counter unavailable')


def main():
    assert sys.platform == 'darwin'
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'results/t6_physical_coalescer.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(6501)
    E, C, size, n_batches, batch_e = 512, 9, 65536, 96, 10
    batches = []
    for _ in range(n_batches):
        ids = rng.sample(range(E), batch_e)
        batches.append([(component * E * size + e * size, size) for e in ids for component in range(C)])
    methods = ['individual', 'lanes']
    rows = []
    scratch = Path(os.environ.get('TMPDIR', '/tmp'))
    with tempfile.TemporaryDirectory(prefix='t6-phys-', dir=str(scratch)) as td:
        path = Path(td) / 'fixture.bin'
        with path.open('wb', buffering=0) as f:
            for _ in range(E * C):
                f.write(rng.randbytes(size))
            os.fsync(f.fileno())
            fcntl.fcntl(f.fileno(), 51)  # F_FULLFSYNC
        fd = os.open(path, os.O_RDONLY)
        try:
            fcntl.fcntl(fd, 48, 1)  # F_NOCACHE
            fcntl.fcntl(fd, 45, 0)  # F_RDAHEAD off
            a = disk_read_bytes(); t0 = time.perf_counter(); time.sleep(1.0); b = disk_read_bytes()
            idle_bps = (b - a) / (time.perf_counter() - t0)
            with ThreadPoolExecutor(max_workers=4) as pool:
                def execute(method, extents):
                    if method == 'individual':
                        blocks = list(pool.map(lambda x: read_exact(fd, x[1], x[0]), extents))
                        return blocks, {'planned_reads': len(extents), 'issued_bytes': sum(n for _, n in extents)}
                    return read_batch(fd, extents, executor=pool, lanes=4, max_gap=0,
                                      max_span=1024 * 1024, max_amplification=1.125)
                for extents in batches[:2]:
                    exp, _ = execute('individual', extents)
                    got, _ = execute('lanes', extents)
                    assert [bytes(x) for x in got] == [bytes(x) for x in exp]
                for rep in range(3):
                    order = methods[rep:] + methods[:rep]
                    for method in order:
                        time.sleep(0.3)
                        p0 = disk_read_bytes()
                        start = time.perf_counter()
                        useful = issued = checksum = 0
                        for batch in batches:
                            got, st = execute(method, batch)
                            checksum += sum(x[0] for x in got)
                            useful += sum(n for _, n in batch)
                            issued += st['issued_bytes']
                        seconds = time.perf_counter() - start
                        p1 = disk_read_bytes()
                        phys = p1 - p0
                        row = dict(rep=rep, method=method, seconds=seconds,
                                   useful_bytes=useful, issued_bytes=issued,
                                   useful_GBps=useful / seconds / 1e9,
                                   physical_bytes=phys,
                                   physical_GBps=phys / seconds / 1e9,
                                   physical_over_useful=(phys / useful) if useful else None,
                                   checksum=checksum)
                        rows.append(row)
                        print(json.dumps(row), flush=True)
            assert len({r['checksum'] for r in rows}) == 1
        finally:
            os.close(fd)
    med_u = {m: statistics.median(r['useful_GBps'] for r in rows if r['method'] == m) for m in methods}
    med_p = {m: statistics.median(r['physical_GBps'] for r in rows if r['method'] == m) for m in methods}
    ratio_phys = (med_p['lanes'] / med_p['individual']) if med_p['individual'] else None
    phys_ok = all(0.7 <= (r['physical_over_useful'] or 0) <= 1.5 for r in rows)
    clear_margin = ratio_phys is not None and ratio_phys >= 1.25
    verdict = 'PASS' if (phys_ok and clear_margin) else ('FAIL_CACHE_ONLY' if not phys_ok else 'FAIL_NO_MARGIN')
    report = dict(
        kind='MEASURED M4 Max 36 GB physical-SSD coalescer microbench (synthetic fixture)',
        host=platform.node(), platform=platform.platform(),
        cache_controls='F_NOCACHE=1, F_RDAHEAD=0, fsync+F_FULLFSYNC write',
        physical_source='IOBlockStorageDriver disk0 Bytes (Read)',
        idle_read_Bps=idle_bps, fixture_bytes=E * C * size,
        experts=E, components=C, component_bytes=size, batches=n_batches,
        workers=4, rows=rows,
        median_useful_GBps=med_u, median_physical_GBps=med_p,
        speedup_physical_lanes_vs_individual=ratio_phys,
        speedup_useful_lanes_vs_individual=med_u['lanes'] / med_u['individual'],
        baseline_random_chunk_peak_GBps=5.02,
        vs_baseline_peak={m: med_p[m] / 5.02 for m in methods},
        physical_approx_useful=phys_ok, verdict=verdict,
        workload_sha256=hashlib.sha256(json.dumps(batches).encode()).hexdigest(),
        caveat='Whole-disk physical counters include metadata/background; idle_read_Bps recorded. Synthetic extents not live expert layout.',
    )
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('verdict', 'median_physical_GBps', 'median_useful_GBps',
                                             'speedup_physical_lanes_vs_individual', 'physical_approx_useful')}, indent=2))


if __name__ == '__main__':
    main()
