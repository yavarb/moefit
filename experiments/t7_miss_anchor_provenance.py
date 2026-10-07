"""T7 provenance/identifiability audit. Existing artifacts only, no silicon.
Run: python3 experiments/t7_miss_anchor_provenance.py
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / name).read_text())


def main():
    measured_path = 'results/measured_santa_cruz_36gb_ssd.json'
    gap_path = 'results/gap_santa_cruz.json'
    policy_path = 'results/design_omlx_exact.json'
    measured, gap, policy = map(load, (measured_path, gap_path, policy_path))
    row = next(r for r in gap['rows'] if r['cap'] == 143)
    physical = measured['decode_disk_MB_per_token']
    size = gap['constants']['expert_MB']
    misses = row['sim_miss_experts_per_token']
    absorption_ratio = physical / (size * misses)
    recovered = physical / (size * absorption_ratio)
    assert abs(recovered - misses) < 1e-12
    assert row['measured_ssd_MB_per_token'] == physical
    assert 'sim_miss_experts_per_token' in row
    assert not any('miss_experts' in key for key in measured)

    candidates = []
    for r in policy['rows']:
        if r['cap'] != 143 or r['policy'] not in ('lru', 'omlx'):
            continue
        m = r['miss_per_tok']
        q = physical / (size * m)
        assert 0 < q <= 1
        assert abs(size * m * q - physical) < 1e-10
        candidates.append(dict(policy=r['policy'], simulated_misses_per_token=m,
                               fitted_physical_fraction=q,
                               reconstructed_physical_MB_per_token=size*m*q))
    # No sweep: one no-absorption explanation plus two existing policies.
    no_absorption = physical / size
    assert abs(size * no_absorption - physical) < 1e-10
    output = dict(
        role='astra_analysis_b', model_tag='Astra gpt-6',
        kind='derived provenance audit, NOT new silicon measurement',
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                       for p in (measured_path, gap_path, policy_path)},
        measured=dict(physical_MB_per_token=physical,
                      decode_tps=measured['decode_tps'],
                      n_disk_samples=measured['n_decode_samples']),
        simulated_miss_anchor=misses,
        assumed_expert_MB=size,
        derived_physical_fraction=absorption_ratio,
        circular_recovered_misses=recovered,
        no_absorption_misses_per_token=no_absorption,
        existing_policy_explanations=candidates,
        identifiability=dict(equation='P = q * E * M', observed_scalars=1,
                            unknown_scalars=['q physical fraction', 'M logical misses'],
                            jacobian_rank=1,
                            caveat='Assumes all disk traffic is expert reads, consistent byte units, complete expert payloads, no read amplification or unrelated IO.'),
        provenance=[
            'gap_santa_cruz.py: lru_misses(gold, cap), then M.sum(1).mean() -> sim_miss_experts_per_token.',
            't8_pagecache_capacity_probe.json: 140.5 measured / 158.8 simulated -> 0.885, not an independent measured fraction.',
            'Inverting that derived fraction to claim measured ~57.4 misses returns the input simulation by construction.',
            'design_omlx_exact.json labels itself simulated; its measured_anchor has throughput and physical bytes, not logical miss counts.',
        ],
        verdict='57.4 misses/token is simulated; source-code policy identification survives, independent measured miss-count validation does not.',
        handoff='Measure logical miss/admission counts and payload bytes directly in the same decode window as physical IO. Host vm_stat/iostat alone cannot identify expert-cache hits.',
    )
    target = ROOT / 'results/t7_miss_anchor_provenance.json'
    target.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
