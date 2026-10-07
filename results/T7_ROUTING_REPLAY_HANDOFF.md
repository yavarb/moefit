# astra_analysis_b | Astra gpt-6 | T7 — routing replay handoff

Cycle 24 closes the current reference implementation per the CPU-cache mission reassignment. M1/M2/M3 ownership stays with the assigned agents. No inference-server restart or new silicon slot is requested.

## Implemented artifact

`moefit/routing_replay.py` provides exact full-prefix routing memoization, optional packed uint32 prefix keys, immutable route IDs/weights, byte-accounted LRU eviction, ordered layered execution, restored-prefix resume, completed-token fork, targeted namespace invalidation, runtime resize, and bounded current-token read planning (`plan_reads`, `plan_complete_layers`). Reuse defaults OFF unless the caller opts into deterministic execution.

## Integration contract for T2 / M2

1. Create a cache and a nonempty bytes namespace identifying immutable model/adapters, tokenizer, positions/attention policy, numerical execution policy, tenant, and initial hidden/KV state. Token equality alone does not establish that two executions have equivalent states.
2. Start `layered_session(namespace, layers, deterministic=True)` only for a backend whose relevant execution is deterministic under that namespace.
3. Call `begin_token(actual_token_id)` before layer 0. A token mismatch prevents reuse of the old continuation, even if later token IDs reconverge.
4. At this boundary, optionally call a planner with a stable per-layer residency mapping, exact expert-byte callback, and hard budget. The output is metadata only. It is not permission to exceed the backend queue depth, install early, or wait on speculative work.
5. At each layer call `route(layer, gate_callback)`. Misses run the callback at the current hidden state. Hits skip ONLY routing; attention, expert computation, hidden updates and KV writes still execute.
6. A restored KV prefix must be paired with `resume_layered(..., prefix_tokens)` for exactly that state. `fork()` clones metadata only; the caller must clone or safely share KV separately.
7. On execution failure discard the session and restore the caller's matching model state. On model/context changes stop affected execution and choose a NEW namespace. Invalidation removes entries but does not revoke returned arrays or prevent old sessions from repopulating old namespaces.

M2 is a probabilistic temporal predictor for non-repeat traffic. Do not feed its predictions into this exact route-hit path. It may use exact-prefix plans as a separate high-confidence source, with independent backend demand validation. No T7 claim that a non-repeat predictor is exact.

## Evidence and limits

Historical local CPU measurements, not rerun for timing in cycle 24:
- `t7_direct_baseline.json`: constructed recurrent MoE, direct 4778 vs replay 5358 tokens/s, 1.121x at 50% route hits. Disabled-wrapper baseline is slower and must not substitute for direct execution.
- `t7_attention_replay.json`: actual causal attention + tiny random MoE, direct 6939 vs replay 7146 tokens/s, 1.030x; uncontrolled-host small delta. All logits and KV arrays bit-exact.
- These are not production LLM or silicon paging gains. T2's measured sidecar gain is its own implementation and is not attributed to T7.

Cycle 24 re-executed correctness:
- `python3 experiments/t7_attention_plans.py`: 144/192 route hits per planner, partial 216 and complete 172 nonresident planned requests, zero false requests, exact logits/KV. Simulated residency; no IO issued.
- `python3 experiments/t7_complete_layers.py`: 200 seeded reference comparisons plus guards pass. Complete-layer planning is greedy, not optimal, and can skip urgent large early groups.
- `python3 experiments/t7_resize.py`: tuple/packed shrink, pause, grow and exact accounting pass; 8/8 repeat hits after regrowth.
- `python3 experiments/t7_namespace_invalidation.py`: 9 target entries removed and 9 unrelated hits retained in each representation; guards pass.

Remaining production gates: deterministic backend capture and namespace definition; GPU/CPU route conversion costs; batched/prefill numerical equivalence; consumer-specific read scheduling and deadlines; end-to-end measured ON/OFF gains. No oMLX adapter, async IO scheduler, or production deployment has been completed.

Capacity is accounted bytes, not RSS. External arrays and session prefixes retain memory. Full-prefix representation has quadratic history storage growth. The cache is not thread-safe. An exact-prefix route cache does not generalize to paraphrases/new topics by itself.
