"""T6 fixed-cap matched-window replay and physical/logical evidence audit.
SIMULATED only. Reuses T2 policy implementations without changing them.
Run: python3 experiments/analysis_t6_policy_windows.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from design_omlx_exact import run_omlx
from design_admission_cache import run_lru
from analysis_t6_censoring import replay

ROOT = Path(__file__).resolve().parents[1]
CAP, L = 143, 48


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    files = ['results/traces_synth/holdout.npz',
             'experiments/design_omlx_exact.py',
             'experiments/design_admission_cache.py',
             'results/measured_santa_cruz_36gb_ssd.json',
             'results/t8_pagecache_capacity_probe.json']
    hashes = {p: sha(ROOT / p) for p in files}
    d = np.load(ROOT / files[0])
    tags = sorted({k.rsplit('|', 1)[0] for k in d.files if k.endswith('|ids')})
    lengths = [len(d[f'{tag}|L0_idx']) for tag in tags]
    n = sum(lengths)
    matrices = {}
    for name, policy in [('lru', run_lru), ('omlx', run_omlx)]:
        retained = np.empty((n, L), int)
        reset = np.empty_like(retained)
        for li in range(L):
            parts = [d[f'{tag}|L{li}_idx'] for tag in tags]
            seq = np.concatenate(parts)
            retained[:, li] = policy(seq, CAP)
            reset[:, li] = np.concatenate([policy(part, CAP) for part in parts])
            if name == 'lru':
                _, m, _ = replay(seq, CAP)
                assert np.array_equal(retained[:, li], m.reshape(n, -1).sum(1))
        matrices[name] = {'retained': retained, 'reset': reset}
    masks = {'all': np.ones(n, bool),
             'global_drop200_T2': np.arange(n) >= 200,
             'each_prompt_drop128_T6': np.concatenate([np.arange(x) >= 128 for x in lengths])}
    rows = []
    for lifecycle in ('retained', 'reset'):
        for label, mask in masks.items():
            row = {'lifecycle': lifecycle, 'window': label, 'tokens': int(mask.sum())}
            for name in matrices:
                m = matrices[name][lifecycle][mask]
                row[name] = {'miss_per_token': float(m.sum(1).mean()),
                             'miss_layers_per_token': float((m > 0).sum(1).mean())}
            a, b = row['lru']['miss_per_token'], row['omlx']['miss_per_token']
            row['omlx_minus_lru_miss_per_token'] = b-a
            row['omlx_vs_lru_miss_percent'] = 100*(b/a-1)
            rows.append(row)
    # Reproduce published rounded T2 anchors, using exactly its global trim.
    anchor = next(r for r in rows if r['lifecycle']=='retained' and r['window']=='global_drop200_T2')
    assert round(anchor['lru']['miss_per_token'], 2) == 57.41
    assert round(anchor['omlx']['miss_per_token'], 2) == 56.91
    measured = json.loads((ROOT / files[3]).read_text())
    physical = measured['decode_disk_MB_per_token']
    expert_mb = 67.95e3 / (48*512)
    simulated = 57.4
    derived_ratio = physical / (simulated * expert_mb)
    recovered = physical / derived_ratio / expert_mb
    assert abs(recovered-simulated) < 1e-10
    per_prompt = []
    start = 0
    for tag, length in zip(tags, lengths):
        sl = slice(start+128, start+length)
        lr = float(matrices['lru']['retained'][sl].sum(1).mean())
        om = float(matrices['omlx']['retained'][sl].sum(1).mean())
        per_prompt.append({'tag': tag, 'suffix_tokens': length-128,
                           'lru_miss_per_token': lr, 'omlx_miss_per_token': om,
                           'omlx_vs_lru_miss_percent': 100*(om/lr-1)})
        start += length
    out = {'kind': 'SIMULATED matched-window replay plus evidence accounting; no new silicon',
           'cap': CAP, 'trace_tokens': n, 'hashes': hashes, 'rows': rows,
           'retained_per_prompt_suffix': per_prompt,
           'provenance_audit': {
               'physical_MB_per_token_measured': physical,
               'expert_MB_assumed_from_geometry': expert_mb,
               'physical_full_expert_equivalent_not_logical_misses': physical/expert_mb,
               'simulated_LRU_misses_used_to_derive_absorption': simulated,
               'derived_absorption_ratio_not_independently_measured': derived_ratio,
               'circular_recovered_misses': recovered,
               'raw_measured_blob_keys': sorted(measured),
               'conclusion': '57.4 is simulated, not an independent logical-miss counter. Throughput and IO microbench evidence remain independent.'},
           'checks': ['LRU mask cross-check on all 48 layers',
                      'T2 57.41/56.91 global-drop200 anchors reproduced',
                      'absorption inversion circularity identity',
                      'input/source hashes unchanged during execution']}
    for p, h in hashes.items():
        assert sha(ROOT / p) == h, p
    dest = ROOT / 'results/analysis_t6_policy_windows.json'
    dest.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps({'rows': rows, 'per_prompt': per_prompt,
                      'provenance_audit': out['provenance_audit'], 'checks': out['checks']}, indent=2))


if __name__ == '__main__':
    main()
