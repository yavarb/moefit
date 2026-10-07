# T8 eligibility probe — astra_local_exp | Astra gpt-6

This is an offline code probe, not a silicon measurement. Inputs are deliberately
fabricated fixtures. Reproduce with:

    python3 experiments/t8_scorer_eligibility_probe.py

Source hashes in results/t8_scorer_eligibility_probe.json lock the scorer and
normalizer that produced the findings. The script aborts if either changes during
execution. Production files are not modified.

## Observed failures

The real scorer CLI completed all 10 cases. Nine inputs violate eligibility or
lack evidence required to establish eligibility, and all nine still receive a
substantive conclusion. The length-valid control also receives a conclusion.

- A requested max_tokens=1024 overrides actual runs[*].tokens=128. No warning.
- Explicit top-level tokens=128 emits FORBIDS warnings but also TRANSFERS.
- Missing all token counts is silently accepted.
- A mixed 1024/128/1024 triplet is silently accepted.
- One run instead of the protocol's three is silently accepted.
- Different host/model/RAM/cap across the paired A inputs is silently accepted.
- Explicit memory_during_run.free_percent=5 is silently accepted. This is a
  fixture field, not a claim that the current collector emits this spelling;
  the scorer currently does not validate memory eligibility at all.
- B and D also accept requested1024/actual128 with no warning.

The control is length-valid, not proof of complete protocol compliance: this
probe does not establish matched prompts, policy provenance or real idle state.

## Handoff to T4/T3 and silicon executor

Do not change numeric prediction thresholds. Separate eligibility from scoring:

1. Read actual authoritative completion counts for EACH contributing run.
   Requested max_tokens is a budget, never evidence of actual completion length.
2. Unknown counts => eligibility unknown, not eligible. A short run => ineligible.
3. Enforce the registered repetition requirement before aggregation; do not
   average short runs into an otherwise valid triplet or silently filter them.
4. Validate matching recorded host/model/policy/cap and prompt identity for A.
   Establish a canonical idle-memory field; missing evidence is not idle proof.
5. When ineligible, preserve descriptive measurements/predictions but suppress
   TRANSFERS, ranking falsification, band-membership inference, and compute-term
   selection. A warning next to the conclusion is not mechanical enforcement.
6. Expose structured eligibility state, reasons, actual counts and provenance so
   callers can distinguish an invalid comparison from a model disagreement.

T4 owns production scorer/collector changes. T3 owns protocol clarification.
No silicon run, network call, inference load, or parameter sweep was performed.
