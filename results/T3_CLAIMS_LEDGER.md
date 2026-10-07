# T3 claims ledger — glm_fidelity (final status of every headline claim)

Why this exists: WINS.md and the notebook are append-only. Several of my
early headlines were later corrected or retracted after astra/T7/T6/T8
audits. This table is the authoritative status of each claim, with the
correction artifact that supersedes the original headline. If a WINS
entry conflicts with this ledger, the LEDGER wins.

| # | Claim (first posted) | Status | Superseded by / stands on |
|---|---------------------|--------|---------------------------|
| 1 | Sim↔silicon gap decomposed; single-knob reconciliation implies effective SSD BW ~2.6–3.0 GB/s; PLE traffic competing explanation (c1) | SUPERSEDED (framing) | The serial model + measured microbench replaced the bandwidth framing: 2.6–3.0 was a duty-cycle/latency blend. T8-queue and lead's constants explain the same data. results/fidelity_miss_model.json (feasibility table) remains valid as conditional analysis. |
| 2 | Serial-latency model validated at 3 measured points (cold warmup 5.0 vs 6.43; steady 12.1 vs 12.7/13.0; cap92 8.9 vs 7.8) | STANDS (means-based) | results/fidelity_serial_validate.json. Caveats: synth traces; hit_refresh=True is an approximation of oMLX's decayed-count policy (<1% on synth, T6 window-specific); anchors are means, immune to the Test-C identifiability critiques. |
| 3 | Serial model IN the simulator; POLICY RANKING FLIP (true-LRU > pinned-prior at every cap; "prior@≥128" carry-forward is a bandwidth artifact) | STANDS | Commit 9cd94bd; robust to the omlx-exact identification (0.6–2.3% at all caps). results/sim_serial_policy_ranking.json. |
| 4 | Equivalence guardrail restored (stale sim_reference constants, not algorithm drift) | STANDS | Commit 4a7ebb8; 45/45 PASS. |
| 5 | 48GB milestone expectations (~14.5–15.9 tok/s @192, not 54.8) + tier table | STANDS as SIM | results/fidelity_tier_predictions.json — all rows labeled simulated; non-36GB rows extrapolated. 36GB anchor row now interval-based (see #9). 54.8 = ideal-pipelining ceiling only. |
| 6 | Pre-registered silicon protocol (Tests A–D) + Amendment 1 (Test C downgraded to shape compatibility) | STANDS | Commit 2ce6a32 + 9640761. Amendment registered openly after independently reproducing the T7 counterexample (1.827 vs their 1.831). Scorer aligned by T4 (4ec1d63); A/B/D fail-close on ACTUAL per-run counts still pending (T8 d778463; my position: anti-rules are hard gates). |
| 7 | "omlx-exact matches the MEASURED miss count (~57.4) within 1%" (c9) | RETRACTED — CIRCULAR | 57.4 is a sim-replay output; q=0.885 was measured-phys/simulated-logical (inverts to recover 57.4). T7 fb927fc + T6 audit; my retraction: 3376266. Corrected anchor is an interval (see #9). Policy identification (decayed-count) rests on the source read + throughput agreement — stands. |
| 8 | "COMPUTE TERM RESOLVED to 24.1 ms; every constant measured or measured-confirmed" (c12) | CORRECTED ×2 → CONDITIONAL | (a) T7's identifiability audit: the regression cannot statistically resolve 18.1 vs 24.1; "resolved" → best-supported (09565e9, af10b16). (b) T6's residency audit invalidated the mincore absorption bound; corrected miss interval [50.8, 54.1]/tok, and the 24.1 preference FLIPS at absorption ~6.6% — currently conditional on absorption ≤ ~6%, bounded only by a future-knowledge static-set argument (5a009a3). Settlers: Tests B/D, matched-window logical-miss counters. |
| 9 | "mincore rejects page-cache absorption → real misses 50.8/tok; synth overstates 15%" (c12–13) | CORRECTED | The mincore byte fraction is a footprint average, NOT a request-weighted bound (T6 analysis_t6_residency_weighting.json; same-size sets span 0.5–5.9% coverage). Corrected: misses ∈ [50.8, 54.1]/tok conditional on absorption 0–5.9%; synth overstatement 8–15% across the interval. 5a009a3. |
| 10 | "(T4's detect_coalescing guard covers this)" re the 256x2 collector run (c12) | RETRACTED — FALSE | T8 6e18d77: the artifact escapes the guard; its p95/mean=1.094 is chunk-gap-derived and inadmissible per Amendment 1 regardless of the detector verdict. Adopted principle: a detector's False is not proof of unbuffered delivery; the 256-tokens-vs-86-chunks mismatch is direct evidence. |
| 11 | Miss-cost slope 0.797 ms/expert "measured-confirmed" | DOWNGRADED to descriptive | The fitted slope (0.288±0.053 ms/MB) is a measured physical-MB association; the per-expert conversion is conditional on the absorption premise (see #9). Value cut stands: slope sits above all pure byte-rates and within 3% of serial io+install (09565e9). |
| 12 | First real-trace validation (persistence calibration holds: 0.365 vs 0.350; omlx-exact ≈ true-LRU on real routing) | STANDS with caveats | results/fidelity_real_traces_firstlook.json — cold-only, single prompt, likely prefill positions; flagged. |
| 13 | Design-band arbitration (T2 optimistic vs T7 pessimistic overlap bounds; quote as bracket) | STANDS | results/DESIGN_BAND_ARBITRATION.md; sidecar-pread silicon test is the decider. |
| 14 | Protocol-owner positions (scorer fail-close; anti-rules are hard gates on actual counts) | STANDS | Recorded in 3376266 + 5a009a3; T4 implementing. |

Open gates on T3 (all settle via existing priority silicon work, no new
requests): (a) absorption ≤ ~6% — currently argument-bounded; (b) the
overlap bracket — sidecar-pread test; (c) 48GB bin (Tests A/D); (d)
real-trace completion; (e) scorer A/B/D fail-close before any sweep is
scored.

## Update 2026-10-08 (cycles 49–52): the silicon arm, and the ledger-relevant aftermath

| # | Claim (cycle) | Status | Notes |
|---|--------------|--------|-------|
| 15 | "T3 LIP v3 silicon arm: +0.34 tps = TRANSFERS by letter" (c49, bea1d24) | STANDS with explicit qualification | Measured ON median 15.98 vs OFF 15.64 (3x1024/arm, same-prompt family 87eb8e91, all finish=length). TRANSFERS is the mechanical pre-registered verdict (>=+0.30) and stands as recorded — but the SAME entry carried the honest statistical reading (bootstrap CI95 [-1.16,+0.80], P(d<=0)=0.45, observed sd ~0.43 -> 3-run floor ~0.7 tps): INCONCLUSIVE in substance. Do not cite "+0.34 TRANSFERS" without the CI. Artifacts: results/t3_lip_scored.json (incl. follow-ups: 507 provenance = pre-v3 attempt; demotion overhead <=0.17 ms/tok = -0.04 tps upper bound; T4's flags-off probe 15.78 in-band; T9's measured install exposure re-price below). |
| 16 | KeyError root cause + freeze compliance (c47-49) | STANDS — CLOSED | Root cause confirmed by 3 independent agents (T3/T9/T4): per-install front-insert evicts before gather. v3 (post-call-tail demotion, d9850d99) is the fix the freeze specified; arm ran under the guarded runbook. Full autopsy: results/T3_DIP_HANDOFF.md (8182706, updated 72b8076). |
| 17 | "Per-miss value under DB-ON bounded [0.58, 0.71] ms/miss" (T5's bound, cited in my c52) | SETTLED — MEASURED ~0.65 | T9's corrected BATCH1 window (b6bd72f: m=38.2 from the server's lifetime, not the cross-process 9.58 artifact) gives D/m = 0.124–0.134 ms/miss install exposure -> S = 0.52 + 0.13 = ~0.65 ms/miss, mid-interval of the bound. Consequence for #3/#15-scale claims: any online-policy miss-rate gain prices at ~+0.10 tps per miss/tok saved (warm-suffix); LIP's -1.291 miss/tok = +0.21 tps expected = exactly the 3-run power floor; real short-prompt regime ~0. The eviction lane remains dead in tps terms under MEASURED constants — every DROP verdict (#3 family, c36-38 idle-decay, admission v1-v3) is now measured-priced, not model-priced. |
| 18 | "8-run/arm LIP resolving session would settle +0.34 vs 0" (c49) | RETIRED — SUPERSEDED by M1 | Registered in the DIP handoff: if M1's DIP ships code-green, score DIP; re-running LIP v3 at n=8 has no independent value. |

Net ledger state after the arm: no standing T3 claim has been weakened by the silicon
result; the ranking-flip/DROP family (#3, #17) is now grounded in MEASURED constants
(S ~0.65 ms/miss), and the only conditional claims (#8/#9, absorption/compute) are
unchanged — their settlers (matched-window logical counters, Tests B/D) remain queued.
