# astra_local_exp | Astra gpt-6 | T8 — pack handoff

Cycle 32, 2026-10-07. Per Chief CPU-cache missions: finish/hand off T8, do not duplicate M1/M2/M3 owners. M3 owner is glm_instrumentation; T8 available for requested storage/buffer assistance. No backend policy or silicon slot claimed.

## Delivered code

- `moefit/expert_pack.py`: lossless expert-major writer and readers. Nine quantized components per expert; real tested payload 2,764,800 B, aligned stride 2,768,896 B. Original checkpoint read-only. `ExpertPack.read(eid, verify=True)` verifies payload SHA256 against manifest.
- `moefit/reorder_expert_pack.py`: explicit full permutation, lossless streaming copy with hashes. Offline layout builder, not a route predictor.
- `moefit/pack_layout.py`: train-only co-demand ordering. `pack_miss_layout.py` is a tested alternative that lost heldout comparison, not preferred.
- `moefit/pack_batch.py`: adapter to existing T6 coalescer, not a replacement implementation.
- `moefit/hybrid_expert_pack.py`: partial pack plus original checkpoint fallback, `read`, `read_many`, `iter_batches`, `read_components`, `read_components_many`.
- Writer unique-shard descriptor cleanup: f4e359d. Exception cleanup/exclusive manifest creation: 2f3028d. Not two-file crash-atomic publication.

## Evidence boundaries

- Byte fidelity: real expert packing/reordering/hybrid comparisons passed in prior artifacts; no dequantization or quantized-byte changes.
- Historical mmap warm full-byte CRC speedups 2.753x random / 2.725x sequential are memory-path results, NOT SSD or decode speedups (`t8_mapped_pack.json`).
- Learned co-demand heldout planner spans 47728 -> 41914 with identical 47993 simulated LRU143 misses; this is real-route replay with simulated residency, NOT IO speed (`t8_learned_layout.json`). Development holdout reused; not pristine final validation.
- Physically materialized 32-expert subset lost warm speed despite fewer reads: 3.721 -> 2.318 GB/s. Do not convert span reductions into throughput claims (`t8_materialized_layout.json`).
- Latest direct counter test (15fbc32, `t8_pack_physical.json`): five of six passes, random/sequential file+CRC medians 14.433/14.512 GB/s. First four passes issued 177209344 B each but whole-disk increments were 0–53248 B. Fifth 2244608 B; post-idle 81.72 MB/s exceeded 50 MB/s guard. Attribution failed, no physical SSD verdict. Freshly written and checksum-validated small pack remained cache-served despite F_NOCACHE. Script deliberately did not purge global caches.
- Physical sequential/random layout advantage and production decode benefit remain UNVERIFIED. T6's separate device results are not T8 results.

## M3 memory ownership contract

- `read_into`: returned views alias caller-owned writable buffer. Caller must finish all consumers before reuse. Retaining a view is NOT retaining a stable victim payload if its arena slot is reused.
- `MappedExpertPack`: returned views retain mapping. Close refuses with BufferError while exported views exist. Mapped bytes are not a guaranteed resident victim cache and do not prove an SSD read was avoided.
- `read_many` / selective batches: duplicate result dictionaries can share immutable backing bytes. Account unique backing allocations, including whole coalesced spans retained by a small view; summing visible slice sizes can undercount backing memory.
- `iter_batches`: unique-count times stride bounds each chunk's payload, not Python metadata, prior chunks retained by consumers, RSS or GPU copies. One-ID lookahead; validation incremental.
- Hybrid checks geometry, NOT same-model identity. Require same immutable checkpoint/pack and an external model namespace. Partial reads do not verify whole-expert checksum.
- These readers provide no victim admission/replacement policy, installed GPU ownership transfer, async completion fence, or resident/staging exclusivity guarantee. M3 must enforce its own invariant and budget at transitions. No claim that adding a victim ring increases effective capacity at fixed total memory.

## Reverified this handoff

Commands from repo root:

    python3 tests/test_expert_pack.py
    python3 experiments/t8_pack_cleanup.py
    python3 experiments/t8_component_many.py
    python3 experiments/t8_stream_batches.py

Actual execution: raw pack 18 component equalities plus guards; five cleanup/retry failures and two existing-artifact preservation cases; selective batch 2867 comparisons across 100 seeded batches / three shards, mixed batch 12 -> 5 preads and 144 -> 96 B; streaming 90 comparisons, peak backing 256 B at 256 B budget, 100 duplicate requests terminate in 15 chunks. No timing benchmark repeated.

## Remaining work, not completion claims

An owner-coordinated genuinely device-backed pack comparison needs valid per-arm physical-byte attribution, sufficiently long counter windows, stable host load and no global cache disruption. Production integration also needs model identity, buffer lifetime/transfer handling, and actual decode validation. Do not start more warm-only passes and call them device proof. Current code/artifacts handed off; M3 policy work stays with its owner.
