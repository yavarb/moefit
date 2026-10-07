# T8 stream integrity — astra_local_exp | Astra gpt-6

Offline synthetic fixtures, not silicon results. Run:

    python3 experiments/t8_stream_integrity_probe.py

The actual collector is exercised with mocked network iteration and a virtual
clock. No endpoint is contacted. The strict integrity validator is a probe-only
prototype, not production plumbing and not a universal SSE protocol rule.

Observed execution:

    complete control                    12.50 tps, 16 deltas, finish=length
    abrupt EOF                          12.50 tps, 16 deltas, finish=None
    usage then EOF, no finish           12.50 tps, 16 deltas, finish=None
    finish/DONE but no usage            12.50 tps, 16 deltas, finish=length
    malformed data before finish        12.50 tps, 16 deltas, finish=length
    combined usage + final text         13.39 tps, 15 deltas, finish=None
    text after DONE                      6.82 tps, 17 deltas, finish=length

Five streams violate the strict benchmark completeness contract, but each
returns positive throughput. Missing usage is labeled streamed_deltas, so that
fallback is not hidden; it still cannot establish authoritative generated-token
counts. Abrupt EOF is not recorded as a completion error. Malformed data is
silently ignored. DONE is ignored as a JSON parse failure, so trailing data is
processed. Combined usage and choice content is valid input structure for this
probe, but the collector's usage branch continues before processing choices,
losing both final text timestamp and finish reason.

Acceptance prototype:

- Record explicit terminal marker, recognized finish, authoritative positive
  integer completion count, malformed-data count, and data-after-terminal.
- Under this benchmark contract, require all completion evidence before admitting
  throughput to model scoring. Preserve partial data as descriptive diagnostics.
- Parse usage independently of choices; a single event can carry both.
- Stop interpreting stream content at DONE. Terminal garbage must never change
  the first/last text-event throughput interval.
- Missing usage is unknown token count, not validated completion count.

All seven expected eligibility classifications pass. Production-source hash
consistency also passes. Integrity alone does not establish >=1024 tokens,
three repetitions, matching hardware, or verified token-level timing.

T4 owns production integration. Existing T4 coalescing/eligibility edits were
left untouched. Thresholds and pre-registered model predictions are unchanged.
