# C1 campaign bench results — 2026-11-15

All three C1 items benched serially behind the rig lock in one ~4.5 h
window (prod + CPU sidecar stopped for the lane, both restored after;
the A55 config change is live in production).

## At-a-glance verdicts

| Item | Attempted | Verdict |
|---|---|---|
| A55 spec/MTP tuning | draft n_max=4 gate + p_min early-stop sweep | **n4 REJECTED; n3 + p_min 0.50 ADOPTED** (live in prod config) |
| A44 b_shift implicit causal mask | PP +3-7% + mask-tensor VRAM | **FAILED (correctness)** — degenerate temp-0 output; branch kept for fix cycle |
| A47 cross-device scheduler events | PP +2-4% via sync elimination | **REJECTED** — syncs eliminated (17.2 -> 1.22 avg/cycle) but PP +0%, TG -6% |

Per-item in-branch reports (verdict on line 1 of each):
- `a44-bshift` branch: `docs/a44-bshift-bench-report-20261115.md`
- `a47-events` branch: `docs/a47-events-bench-report-20261115.md`

## A55 — n_max=4 rejected, p_min early-stop adopted

- Gate legs (n4, p_min 0.00, reps 3) vs a same-bench n3/p0 control
  (added on the spot; the prepared 14-leg config had no n3-p0 arm):
  - Q6 118k: n3 median 9.41 / n4 9.10 (-3.3%)
  - Q5 134k: n3 median 9.51 / n4 8.19 (-13.8%)
  - n4 rate/pos on filler: (0.550, 0.270, 0.138, 0.074) Q6,
    (0.504, 0.220, 0.088, 0.038) Q5
  - The row's absolute gate (r3 >= 0.33, tg >= 9.8) is in serving
    units (anchors from real-traffic daily-log rounds). The bench
    filler's acceptance curve decays far faster (first position 0.55
    vs 0.768 serving), so the absolute gate is serving-unit-only;
    scored relatively, n4 loses on both quants.
  - Extra strike: the ONLY config whose temp-0 identity diverged from
    the counting baseline was n4-p0 on Q6 (echo variant). n4-p0 is
    both slower and the only output-shifting config.
  - Risk of the reject: if serving r4 really is ~0.37 (row
    extrapolation), n4 would win ~+9-11% on serving text. Recoverable
    later by a temporary daily-INI n_max=4 flip + daily-log read.
- p_min sweep (p_min {0.50, 0.70, 0.90} x n_max {3,4}, both quants,
  reps 2), mean-of-reps vs n3-p0 baseline (9.27 Q6 / 9.73 Q5):
  - **n3-p50: 9.87 (+6.5%) / 11.12 (+14.3%) — winner on both quants**
  - n4-p50: 9.64 / 10.85; p70: +3.3% / +9.0% class; p90: ~-5% class
  - Mechanism: at p_min>0 low-confidence rounds SKIP drafting entirely
    (accept-calls << draft-calls); draft ms/round 39.6-40.7 (n4-p0)
    -> 19.4-19.8 (n3-p50) -> ~14 (p90), mean acc len held/rose on
    drafted rounds.
  - Output-neutrality: temp-0 identity IDENTICAL across all p_min and
    both n_max values (except the n4-p0-Q6 outlier above).
- Adoption: `spec-draft-p-min = 0.50` added to all eight 27B daily
  INI blocks (the 9B compaction block untouched); load log verified
  at prod restart: `n_max=3, n_min=0, p_min=0.50, backend_sampling=1`.
  Revert = remove the added lines. Serving-unit soak verdict (rate/pos
  curve, mean acc len, tg vs the 9.09 baseline) reads from the daily
  log after ~1-2 days of real traffic.

## A44 — gate FAILED (correctness), no merge

- A/B on the feature binary, `--b-shift off|on`, Q6 118k, reps 3.
- OFF clean (identity counting, pp rep1 226.7, tg mean 9.39); ON
  DEGENERATE — identity `1,2,3,4,5,5,6,7,8,5,7,8,1,2,3,4,...` (matches
  OFF ~5 tokens then repetition loops), pp 225.7 (+0% vs +3-7% gate),
  tg mean 8.70, VRAM peak 9945 -> 9771 MB (mask tensor gone — the path
  WAS exercised).
- Diagnosis: short-context / K-tile-boundary mask bug class (probe
  prompt ~15 tok < K-tile; divergence starts at output token ~5-6).
  The impl report's "bit-identity" was analytic-only; the bench was
  the first empirical test and it caught the bug.
- Process defect noted: the `b_shift path taken` verification line is
  below the visible log level at `-lv 4` — raise it.
- Fix path before re-bench: empirical kernel validation at short n_kv
  (both MMA_F16 and TILE), a visible-level path-taken line, a tiny
  identity check. Branch and build dir intact.

## A47 — mechanism proven, impact nil, rejected

- A/B on the feature binary via env kill switch (OFF:
  `LLAMA_SCHED_EVENTS=0`; ON default-on with the debug counter), Q6
  118k, reps 3 each.
- ON counted syncs avg 1.22 / max 2 over 4,486 compute cycles
  (n_splits=34, n_copies=1) vs OFF avg 17.2 / max 47 over 4,219 — the
  ~34/ubatch baseline eliminated. No deadlock (A12 hang check passed:
  3 full reps + identity clean). Identity identical.
- PP 227.9 vs 227.8 (+0%); TG mean 9.05 vs 9.64 (-6%, ~1 sigma run
  variance, direction negative).
- Why: prefill is P40 MMQ GEMM-bound (syncs were ~1.2-2.5% of the
  ubatch, inside noise); decode's fixed 24% gap is handoff-kernel
  spin-syncs + weight streaming, which scheduler-level events do not
  touch. The syncs were never the bottleneck on this rig.
- Disposition: REJECTED (no throughput win; adds pinned-RAM cost, a
  formal staging-reuse race, complexity). Event infra + double-
  buffered staging stay on the branch for a future A27-class
  pipelining item; kill switch `LLAMA_SCHED_EVENTS=0`.

## Methodological keeper (generalizable)

The harness filler prompts produce a much lower, faster-decaying MTP
acceptance curve than production traffic (first position ~0.55 vs
0.768 serving). Any future gate whose thresholds were derived from
serving-log data must either be scored RELATIVELY on the bench and
confirmed on real traffic, or use a serving-like prompt source. Bench
units != serving units.

## Evidence

- Result CSVs: `a1bench_results_20261114_220257.csv` (A55 gate),
  `a1bench_results_20261114_223124.csv` (A55 control),
  `a1bench_results_20261114_225736.csv` (A55 sweep),
  `a1bench_results_20261115_011656.csv` (A44),
  `a1bench_results_20261115_014013.csv` (A47 off),
  `a1bench_results_20261115_015125.csv` (A47 on).
- Per-leg server logs + run logs alongside (naming:
  `a1bench_<stamp>_<tag>.log`, `c1_*_run.log`).
- A55 soak verification line: `spec common_specu: - n_max=3,
  n_min=0, p_min=0.50, n_embd=5120, backend_sampling=1`.
- Sanitize: quad-anchored IP + drive-path + username sweep on this
  report -> zero hits.
