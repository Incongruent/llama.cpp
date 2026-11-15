# A47 scheduler-events bench report — 2026-11-15

**VERDICT: REJECTED — mechanism proven, no throughput impact; NO MERGE
to prod.** Event-based cross-device sync works exactly as designed
(full stream syncs eliminated) but buys nothing on this rig. The
event infra + double-buffered pinned staging stay on this branch for a
future pipelining item (A27-class) if per-sync granularity ever
matters.

## Summary

- Replaces the scheduler's full cross-device stream syncs with CUDA
  event wait/record + double-buffered 32 MiB pinned staging; kill
  switch `LLAMA_SCHED_EVENTS=0`.
- Row gate: PP +2-4% at ub1024, <5 counted syncs per compute cycle
  (vs the ~34/ubatch baseline), no A12-style deadlock.
- Sync criterion: **PASS decisively** (avg 1.22 / max 2 per cycle vs
  avg 17.2 / max 47 baseline).
- No deadlock (3 full reps, identity clean), output-neutral.
- Impact: **FAIL** — PP +0%, TG -6% (~1 sigma, direction negative).

## Bench setup

- Harness: a1b direct-launch, port 8901, fill = 0.9*ctx, n_predict 512,
  reps 3, temp 0, identity gate per leg.
- Geometry: Q6_K 27B, r_ctx 118000, ub 1024, b 2048, kvc q8_0,
  layer-split-hybrid attn:16, tensor-split 0.22/0.78 (3080 / P40).
- Feature binary built from this branch.
- A/B via env: OFF leg `LLAMA_SCHED_EVENTS=0` (leg `a47_off_q6k118`),
  ON leg default-on with `GGML_SCHED_DEBUG=1` for per-cycle sync
  counters (leg `a47_on_q6k118`).

## Results

| metric | off (kill switch) | on (events) | gate |
|---|---|---|---|
| counted syncs per compute cycle | avg 17.21, max 47 (4,219 cycles) | **avg 1.22, max 2** (4,486 cycles) | <5 |
| pp rep1 (real fill 104,474 tok) | 227.8 t/s | 227.9 t/s (+0%) | +2-4% |
| tg mean (3 reps) | 9.64 (9.39/9.48/10.04) | 9.05 (8.61/9.33/9.20) | >= off |
| identity (temp 0) | counting | counting (identical) | unchanged |
| deadlock | - | none (3 full reps) | none |

Counter line format (timestamp prefix stripped):
`I ggml_backend_sched_compute_splits: A47 sync_count=2  n_splits=34
n_copies=1  events=on` — the n_splits=34 shape is the 34-way PP-off
split path; events=off/on shows which arm each line belongs to.

## Why the mechanism wins nothing here

- Prefill (where the ~34 syncs/ubatch lived) is P40 MMQ GEMM-bound:
  the syncs cost ~1.2-2.5% of the ubatch, inside run-to-run noise.
- Decode's fixed gap vs single-GPU is handoff-kernel spin-syncs +
  weight streaming — scheduler-level stream syncs are not on that
  critical path.
- Conclusion: full stream syncs were never the bottleneck on this rig;
  eliminating them correctly yields ~zero.

## Disposition

- REJECTED for prod (no win; adds pinned-RAM cost, a formal
  staging-reuse race, and complexity).
- Revert / kill switch for anyone building this branch:
  `LLAMA_SCHED_EVENTS=0`.
- Keep on this branch: the event-based sync primitives and the
  double-buffered pinned staging are reusable building blocks for an
  A27-class ubatch pipelining item, where per-sync granularity (not
  sync elimination) is the lever.

## Evidence

- Results CSVs: `a1bench_results_20261115_014013.csv` (off),
  `a1bench_results_20261115_015125.csv` (on)
- Per-leg server logs: `a1bench_20261115_014013_a47_off_q6k118.log`,
  `a1bench_20261115_015125_a47_on_q6k118.log`
- Run logs: `c1_a47_off_run.log`, `c1_a47_on_run.log`
- Sync counter totals: 4,219 lines (off) / 4,486 lines (on) extracted
  from the leg logs with max/avg computed over the full run.
- Sanitize: quad-anchored IP + drive-path + username sweep -> zero hits.
